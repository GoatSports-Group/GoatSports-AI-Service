"""Đánh giá trợ lý GOAT AI.

Chạy:  .venv/Scripts/python scripts/eval_rag.py [--embedder e5|hash|both] [--sweep] [--answers] [--tools]

- Truy xuất (mặc định, không gọi LLM): Hit@1, Hit@5, MRR trên knowledge/eval/questions.json; câu lạc đề bị chặn
  ngay ở bước tìm. --sweep thử nhiều ngưỡng relevance_floor.
- --answers (gọi LLM thật, tốn vài cent): câu trong phạm vi có trích [Nguồn N]; câu lạc đề bị từ chối.
- --tools (gọi LLM thật): trên knowledge/eval/tools.json, công cụ được chọn đúng và điền đúng tham số; câu chính
  sách / lạc đề không gọi công cụ; câu hỏi nối tiếp nhiều lượt. Dữ liệu sân, giải, đơn là dữ liệu giả cố định,
  "hôm nay" cố định 2026-10-15, nên kết quả lặp lại được.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.application.dto.request.models import ChatbotQuery  # noqa: E402
from app.application.port.client import CallerIdentity  # noqa: E402
from app.application.service.assistant_service import AssistantService  # noqa: E402
from app.application.service.assistant_tools import AssistantTools  # noqa: E402
from app.application.service.rag_chatbot_service import RagChatbotService  # noqa: E402
from app.application.service.service_config import RagConfig  # noqa: E402
from app.application.service.vector_search import HashingEmbedder  # noqa: E402

EVAL_TODAY = date(2026, 10, 15)
EVAL_USER = CallerIdentity(UUID("11111111-1111-1111-1111-111111111111"), "nguoichoi@goat.test", "nguoichoi", "PLAYER")
REFUSAL_MARKS = ("chưa có thông tin", "không có thông tin", "không thể tư vấn", "không thể trả lời",
                 "chỉ có thể trả lời", "hỏi về goatsports", "mình là trợ lý của goatsports", "liên hệ hỗ trợ")


# ---- Bộ nhớ giả (dùng chung cho test) ----

class InMemoryKnowledgeRepository:
    def __init__(self) -> None:
        self.documents, self.chunks, self.models, self.unanswered = {}, {}, {}, []

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

    def record(self, question, reason, bad_answer=None):
        """Đóng vai KnowledgeGapService (chỉ ghi lại) để đếm câu chưa trả lời được."""
        self.unanswered.append((question, reason))

    def find_active_model(self, capability):
        return self.models.get(capability)

    def save_model_version(self, model):
        self.models[model.capability] = model
        return model


class InMemoryConversationStore:
    def __init__(self, limit_reached: bool = False) -> None:
        self.history: dict[str, list[dict[str, str]]] = {}
        self.limit_reached = limit_reached

    def load(self, key):
        return list(self.history.get(key, []))

    def append(self, key, question, answer):
        self.history.setdefault(key, []).extend(
            [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        )

    def allow(self, key, limit_per_minute):
        return not self.limit_reached


class FakePlatformData:
    """Dữ liệu cố định của venue-service / club-service; ghi lại mọi lời gọi để kiểm."""

    VENUES = [
        {"venueId": "v-goat", "name": "Goat Arena", "address": "12 Quang Trung", "district": "Gò Vấp",
         "minPrice": 80000, "maxPrice": 150000, "averageRating": 4.6, "totalReviews": 23,
         "sportTypes": ["BADMINTON", "PICKLEBALL"]},
        {"venueId": "v-td", "name": "Sân bóng Thủ Đức", "address": "5 Võ Văn Ngân", "district": "Thủ Đức",
         "minPrice": 250000, "maxPrice": 400000, "averageRating": 4.2, "totalReviews": 9, "sportTypes": ["FOOTBALL"]},
    ]
    CLUB = {"clubId": "c-1", "name": "CLB Cầu Lông Phú Nhuận", "sportType": "BADMINTON", "city": "Hồ Chí Minh",
            "memberCount": 11, "matchCount": 6, "winRate": 0.67}

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def search_venues(self, keyword, sport_type, district, max_price, sort, limit):
        self.calls.append(("search_venues", keyword, sport_type, district, max_price, sort))
        rows = [venue for venue in self.VENUES
                if (not keyword or keyword.lower() in venue["name"].lower())
                and (not sport_type or sport_type in venue["sportTypes"])
                and (not district or district.lower() in venue["district"].lower())
                and (not max_price or venue["minPrice"] <= max_price)]
        if sort == "price":
            rows.sort(key=lambda venue: venue["minPrice"])
        if sort == "rating":
            rows.sort(key=lambda venue: -venue["averageRating"])
        return rows[:limit], len(rows)

    def free_slots(self, venue_keyword, play_date, sport_type):
        self.calls.append(("free_slots", venue_keyword, play_date, sport_type))
        venue = next((v for v in self.VENUES if venue_keyword.lower() in v["name"].lower()), None)
        if venue is None:
            return None
        slots = [{"start": "18:00:00", "end": "19:00:00", "price": 120000}, {"start": "20:00:00", "end": "21:00:00", "price": 120000}]
        return {"venue_id": venue["venueId"], "venue_name": venue["name"],
                "courts": [{"court": "Sân 1", "slots": slots}, {"court": "Sân 2", "slots": []}]}

    def search_tournaments(self, sport_type, status, keyword, limit):
        self.calls.append(("search_tournaments", sport_type, status, keyword))
        if sport_type not in (None, "PICKLEBALL") or status != "REGISTRATION_OPEN":
            return []
        return [{"tournamentId": "t-1", "name": "Goat Pickleball Open 2026", "sportType": "PICKLEBALL",
                 "playFormatLabel": "Đánh đôi", "status": "REGISTRATION_OPEN", "entryFee": 200000,
                 "currentParticipants": 10, "maxParticipants": 16, "startDate": "2026-11-01",
                 "endDate": "2026-11-02", "registrationCloseDate": "2026-10-25"}]

    def my_clubs(self, identity, activity_limit):
        self.calls.append(("my_clubs", identity.user_id))
        return ([{"membershipId": "m-1", "role": "OWNER", "status": "ACTIVE", "club": self.CLUB}],
                [{"activityId": "a-1", "clubId": "c-1", "title": "Tập kỹ thuật đập cầu", "startAt": "2026-10-16T19:00:00"}])

    def search_clubs(self, sport_type, city, keyword, limit):
        self.calls.append(("search_clubs", sport_type, city, keyword))
        rows = [self.CLUB] if sport_type in (None, "BADMINTON") else []
        return rows[:limit], len(rows)

    def my_bookings(self, identity, status, limit):
        self.calls.append(("my_bookings", identity.user_id, status))
        return [{"bookingId": "b-1", "bookingCode": "GS123456", "venueName": "Goat Arena", "courtName": "Sân 1",
                 "playDate": "2026-10-17", "startTime": "18:00:00", "endTime": "19:00:00", "status": "CONFIRMED",
                 "depositAmount": 36000, "remainingAmount": 84000,
                 "cancellation": {"status": "PENDING", "refundAmount": 36000}}]


def build_knowledge(embedder) -> RagChatbotService:
    knowledge = RagChatbotService(InMemoryKnowledgeRepository(), embedder, RagConfig(max_context_chunks=5))
    knowledge.sync_directory(ROOT / "knowledge")
    return knowledge


def build_assistant(knowledge, llm, data=None, store=None) -> AssistantService:
    return AssistantService(
        knowledge, AssistantTools(data or FakePlatformData()), llm, store or InMemoryConversationStore(),
        knowledge.repository, clock=lambda: datetime(2026, 10, 15, 9, 0, tzinfo=timezone(timedelta(hours=7))),
    )


# ---- Truy xuất ----

def expected_docs(item: dict) -> set[str]:
    value = item["doc"]
    if not value:
        return set()
    return set(value) if isinstance(value, list) else {value}


def retrieved_files(knowledge: RagChatbotService, question: str) -> list[str]:
    return [entry.document.source_url.removeprefix("knowledge/") for entry in knowledge.retrieve(question)]


def evaluate_retrieval(knowledge, questions, verbose=True) -> dict:
    in_scope = [item for item in questions if item["doc"]]
    out_scope = [item for item in questions if not item["doc"]]
    hit1 = hit5 = missed = 0
    reciprocal = 0.0
    for item in in_scope:
        files = retrieved_files(knowledge, item["q"])
        rank = next((index for index, name in enumerate(files, 1) if name in expected_docs(item)), None)
        hit1 += rank == 1
        hit5 += rank is not None
        reciprocal += 1 / rank if rank else 0.0
        missed += not files
        if verbose and rank != 1:
            print(f"    [{f'top{rank}' if rank else 'MISS'}] {item['q']} -> {files[:3]}")
    leaked = [item["q"] for item in out_scope if retrieved_files(knowledge, item["q"])]
    if verbose:
        for question in leaked:
            print(f"    [LEAK] {question}")
    # Chế độ không có LLM: chỉ trích nguyên văn đoạn đạt quote_floor (câu lạc đề không được lọt câu nào).
    quote_floor = knowledge.embedder.quote_floor

    def quotable(question):
        return any(entry.relevance >= quote_floor for entry in knowledge.retrieve(question))

    quoted_out = sum(quotable(item["q"]) for item in out_scope)
    return {"hit@1": hit1 / len(in_scope), "hit@5": hit5 / len(in_scope), "mrr": reciprocal / len(in_scope),
            "missed": missed, "out_of_scope_rejected": (len(out_scope) - len(leaked)) / len(out_scope),
            "fallback_in_quoted": sum(quotable(item["q"]) for item in in_scope) / len(in_scope),
            "fallback_out_rejected": (len(out_scope) - quoted_out) / len(out_scope)}


def sweep(knowledge, questions) -> None:
    embedder_type = type(knowledge.embedder)
    original = embedder_type.relevance_floor
    for step in range(13):
        floor = round(original - 0.06 + 0.01 * step, 2)
        embedder_type.relevance_floor = floor
        with contextlib.redirect_stdout(io.StringIO()):
            result = evaluate_retrieval(knowledge, questions, verbose=False)
        print(f"  floor={floor:.2f}  hit@1={result['hit@1']:.2f}  hit@5={result['hit@5']:.2f}  "
              f"missed={result['missed']}  out_rejected={result['out_of_scope_rejected']:.2f}")
    embedder_type.relevance_floor = original


# ---- Câu trả lời (LLM thật) ----

def is_refusal(answer: str) -> bool:
    return any(mark in answer.lower() for mark in REFUSAL_MARKS)


def check_answers(knowledge, llm, questions) -> None:
    in_scope = [item for item in questions if item["doc"]]
    out_scope = [item for item in questions if not item["doc"]]
    assistant = build_assistant(knowledge, llm)
    cited = sum("[Nguồn" in assistant.answer(ChatbotQuery(message=item["q"]), None).answer for item in in_scope)
    refused = 0
    for item in out_scope:
        answer = assistant.answer(ChatbotQuery(message=item["q"]), None).answer
        refused += is_refusal(answer)
        print(f"    Q: {item['q']}\n    A: {answer[:150]}")
    print(f"  answers citing [Nguồn N]: {cited}/{len(in_scope)}   out-of-scope refused: {refused}/{len(out_scope)}")


# ---- Công cụ (LLM thật, dữ liệu giả) ----

class RecordingLlm:
    """Bọc LLM thật, ghi lại công cụ nó chọn ở mỗi lượt."""

    def __init__(self, inner) -> None:
        self.inner, self.tool_calls = inner, []

    @property
    def model_name(self):
        return self.inner.model_name

    def complete(self, system, prompt, max_tokens=None):
        return self.inner.complete(system, prompt, max_tokens)

    def chat(self, messages, tools, max_tokens):
        turn = self.inner.chat(messages, tools, max_tokens)
        if turn:
            self.tool_calls.extend((call.name, json.loads(call.arguments or "{}")) for call in turn.tool_calls)
        return turn


def args_match(expected: dict, actual: dict) -> bool:
    def norm(value):
        return str(value).strip().lower() if not isinstance(value, (int, float)) else float(value)
    return all(norm(actual.get(key)) == norm(value) for key, value in expected.items())


def check_tools(knowledge, llm, spec: dict) -> None:
    right_tool = right_args = 0
    for item in spec["single"]:
        recorder = RecordingLlm(llm)
        assistant = build_assistant(knowledge, recorder)
        response = assistant.answer(ChatbotQuery(message=item["q"]), EVAL_USER)
        calls = [call for call in recorder.tool_calls if call[0] == item["tool"]]
        ok_tool = bool(calls)
        ok_args = ok_tool and any(args_match(item["args"], args) for _, args in calls)
        right_tool += ok_tool
        right_args += ok_args
        if not ok_args:
            print(f"    [{'ARGS' if ok_tool else 'TOOL'}] {item['q']} -> {recorder.tool_calls}")
        if not response.cards and ok_tool:
            print(f"    [NO CARD] {item['q']}")
    total = len(spec["single"])
    print(f"  tool chosen: {right_tool}/{total}   tool + args right: {right_args}/{total}")

    false_calls = 0
    for question in spec["no_tool"]:
        recorder = RecordingLlm(llm)
        build_assistant(knowledge, recorder).answer(ChatbotQuery(message=question), EVAL_USER)
        if recorder.tool_calls:
            false_calls += 1
            print(f"    [TOOL USED] {question} -> {recorder.tool_calls}")
    print(f"  policy / off-topic questions without tool calls: {len(spec['no_tool']) - false_calls}/{len(spec['no_tool'])}")

    follow_ok = 0
    for case in spec["multi_turn"]:
        store = InMemoryConversationStore()
        conversation = uuid4()
        recorder = RecordingLlm(llm)
        assistant = build_assistant(knowledge, recorder, store=store)
        for turn in case["turns"][:-1]:
            assistant.answer(ChatbotQuery(message=turn, conversation_id=conversation), EVAL_USER)
        recorder.tool_calls.clear()
        assistant.answer(ChatbotQuery(message=case["turns"][-1], conversation_id=conversation), EVAL_USER)
        ok = any(name == case["tool"] and args_match(case["args"], args) for name, args in recorder.tool_calls)
        follow_ok += ok
        if not ok:
            print(f"    [FOLLOW-UP] {case['turns']} -> {recorder.tool_calls}")
    print(f"  follow-up questions resolved: {follow_ok}/{len(spec['multi_turn'])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", choices=["e5", "hash", "both"], default="both")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--answers", action="store_true")
    parser.add_argument("--tools", action="store_true")
    args = parser.parse_args()
    questions = json.loads((ROOT / "knowledge" / "eval" / "questions.json").read_text(encoding="utf-8"))

    embedders = []
    if args.embedder in {"hash", "both"}:
        embedders.append(HashingEmbedder())
    if args.embedder in {"e5", "both"}:
        from app.infrastructure.embedding.e5_onnx import E5OnnxEmbedder

        embedders.append(E5OnnxEmbedder(ROOT / "models" / "e5-small-model.onnx", ROOT / "models" / "e5-small-tokenizer.json"))

    llm = None
    if args.answers or args.tools:
        from app.infrastructure.config.settings import settings
        from app.infrastructure.external.client.service_clients import OpenAiCompatibleLanguageModelClient

        llm = OpenAiCompatibleLanguageModelClient(settings)

    for embedder in embedders:
        knowledge = build_knowledge(embedder)
        print(f"== {embedder.model_name} (floor {embedder.relevance_floor}) · {len(knowledge.repository.chunks)} đoạn")
        result = evaluate_retrieval(knowledge, questions)
        print("  " + "  ".join(f"{key}={value:.2f}" if isinstance(value, float) else f"{key}={value}"
                               for key, value in result.items()))
        if args.sweep:
            sweep(knowledge, questions)
        if args.answers:
            check_answers(knowledge, llm, questions)
        if args.tools:
            spec = json.loads((ROOT / "knowledge" / "eval" / "tools.json").read_text(encoding="utf-8"))
            check_tools(knowledge, llm, spec)


if __name__ == "__main__":
    main()
