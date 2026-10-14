"""Đánh giá truy xuất của chatbot RAG trên bộ câu hỏi knowledge/eval/questions.json.

Chạy:  .venv/Scripts/python scripts/eval_rag.py [--embedder e5|hash|both] [--sweep] [--answers]

- Nạp toàn bộ knowledge/*.md vào một kho trong bộ nhớ (không cần MongoDB).
- Câu trong phạm vi: Hit@1, Hit@5 (tài liệu đúng nằm trong 5 đoạn lấy ra), MRR, số câu bị bỏ sót (không lấy ra gì).
- Câu ngoài phạm vi: tỉ lệ bị từ chối đúng (không đoạn nào vượt ngưỡng liên quan).
- --sweep: thử nhiều ngưỡng để chọn relevance_floor cho embedder.
- --answers: gọi LLM thật (tốn vài cent) để kiểm câu trả lời có trích [Nguồn N] và câu ngoài phạm vi bị từ chối.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.application.dto.request.models import ChatbotQuery  # noqa: E402
from app.application.service.rag_chatbot_service import NOT_FOUND_ANSWER, RagChatbotService  # noqa: E402
from app.application.service.service_config import RagConfig  # noqa: E402
from app.application.service.vector_search import HashingEmbedder  # noqa: E402


class InMemoryKnowledgeRepository:
    def __init__(self) -> None:
        self.documents, self.chunks, self.models = {}, {}, {}

    def save_knowledge_document(self, document):
        self.documents[document.document_id] = document
        return document

    def find_knowledge_documents(self):
        return {key: value for key, value in self.documents.items() if value.active}

    def replace_knowledge_chunks(self, document_id, chunks):
        self.chunks = {key: value for key, value in self.chunks.items() if value.document_id != document_id}
        self.chunks.update({chunk.chunk_id: chunk for chunk in chunks})

    def find_knowledge_chunks(self, limit: int = 2000):
        return list(self.chunks.values())[:limit]

    def delete_knowledge_document(self, document_id):
        self.documents.pop(document_id, None)
        self.replace_knowledge_chunks(document_id, [])

    def delete_embeddings(self, source_type):
        return 0

    def find_active_model(self, capability):
        return self.models.get(capability)

    def save_model_version(self, model):
        self.models[model.capability] = model
        return model


class _NoVenues:
    def get_recommendations(self, **_):
        return []


def build(embedder, llm=None) -> RagChatbotService:
    service = RagChatbotService(
        InMemoryKnowledgeRepository(), _NoVenues(), llm, embedder, RagConfig(max_context_chunks=5)
    )
    service.sync_directory(ROOT / "knowledge")
    return service


def expected_docs(item: dict) -> set[str]:
    value = item["doc"]
    if not value:
        return set()
    return set(value) if isinstance(value, list) else {value}


def retrieved_files(service: RagChatbotService, question: str) -> list[str]:
    return [entry.document.source_url.removeprefix("knowledge/") for entry in service.retrieve(question)]


def evaluate(service: RagChatbotService, questions: list[dict], verbose: bool = True) -> dict:
    in_scope = [item for item in questions if item["doc"]]
    out_scope = [item for item in questions if not item["doc"]]
    hit1 = hit5 = missed = 0
    reciprocal = 0.0
    for item in in_scope:
        files = retrieved_files(service, item["q"])
        wanted = expected_docs(item)
        rank = next((index for index, name in enumerate(files, 1) if name in wanted), None)
        hit1 += rank == 1
        hit5 += rank is not None
        reciprocal += 1 / rank if rank else 0.0
        missed += not files
        if verbose and rank != 1:
            label = f"top{rank}" if rank else "MISS"
            print(f"    [{label}] {item['q']} -> {files[:3]}")
    leaked = [item["q"] for item in out_scope if retrieved_files(service, item["q"])]
    if verbose:
        for question in leaked:
            print(f"    [LEAK] {question}")
    return {
        "hit@1": hit1 / len(in_scope),
        "hit@5": hit5 / len(in_scope),
        "mrr": reciprocal / len(in_scope),
        "missed": missed,
        "out_of_scope_rejected": (len(out_scope) - len(leaked)) / len(out_scope),
    }


def sweep(service: RagChatbotService, questions: list[dict]) -> None:
    embedder_type = type(service.embedder)
    original = embedder_type.relevance_floor
    for step in range(13):
        floor = round(original - 0.06 + 0.01 * step, 2)
        embedder_type.relevance_floor = floor
        with contextlib.redirect_stdout(io.StringIO()):
            result = evaluate(service, questions, verbose=False)
        print(
            f"  floor={floor:.2f}  hit@1={result['hit@1']:.2f}  hit@5={result['hit@5']:.2f}  "
            f"missed={result['missed']}  out_rejected={result['out_of_scope_rejected']:.2f}"
        )
    embedder_type.relevance_floor = original


# Câu từ chối của LLM không cố định chữ; các dấu hiệu này đủ để đếm (đọc kèm phần in ra khi cần chắc chắn).
REFUSAL_MARKS = ("chưa có thông tin", "không có thông tin", "không thể tư vấn", "không thể trả lời",
                 "chỉ có thể trả lời", "mời bạn hỏi về goatsports", "hỏi về goatsports",
                 "mình là trợ lý của goatsports")


def check_answers(service: RagChatbotService, questions: list[dict], verbose_answers: bool = True) -> None:
    in_scope = [item for item in questions if item["doc"]]
    out_scope = [item for item in questions if not item["doc"]]
    cited = sum("[Nguồn" in service.answer_query(ChatbotQuery(message=item["q"])).answer for item in in_scope)
    refused = 0
    for item in out_scope:
        answer = service.answer_query(ChatbotQuery(message=item["q"])).answer
        refused += answer == NOT_FOUND_ANSWER or any(mark in answer.lower() for mark in REFUSAL_MARKS)
        if verbose_answers:
            print(f"    Q: {item['q']}")
            print(f"    A: {answer[:160]}")
    print(f"  answers citing [Nguồn N]: {cited}/{len(in_scope)}   out-of-scope refused: {refused}/{len(out_scope)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", choices=["e5", "hash", "both"], default="both")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--answers", action="store_true")
    args = parser.parse_args()
    questions = json.loads((ROOT / "knowledge" / "eval" / "questions.json").read_text(encoding="utf-8"))

    embedders = []
    if args.embedder in {"hash", "both"}:
        embedders.append(HashingEmbedder())
    if args.embedder in {"e5", "both"}:
        from app.infrastructure.embedding.e5_onnx import E5OnnxEmbedder

        embedders.append(
            E5OnnxEmbedder(ROOT / "models" / "e5-small-model.onnx", ROOT / "models" / "e5-small-tokenizer.json")
        )

    llm = None
    if args.answers:
        from app.infrastructure.config.settings import settings
        from app.infrastructure.external.client.service_clients import OpenAiCompatibleLanguageModelClient

        llm = OpenAiCompatibleLanguageModelClient(settings)

    for embedder in embedders:
        service = build(embedder, llm)
        print(f"== {embedder.model_name} (floor {embedder.relevance_floor}) · {len(service.repository.chunks)} đoạn")
        result = evaluate(service, questions)
        print("  " + "  ".join(f"{key}={value:.2f}" if isinstance(value, float) else f"{key}={value}"
                               for key, value in result.items()))
        if args.sweep:
            sweep(service, questions)
        if args.answers:
            check_answers(service, questions)


if __name__ == "__main__":
    main()
