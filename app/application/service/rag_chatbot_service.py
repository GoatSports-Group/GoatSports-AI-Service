"""Kho tri thức (RAG) của trợ lý GOAT AI: chính sách, hướng dẫn sử dụng, FAQ. Việc hỏi đáp nằm ở AssistantService.

Kho tri thức là các file Markdown trong thư mục knowledge/, tự đồng bộ lúc khởi động. Mỗi đoạn nhớ đường đề mục
của nó ("Hủy sân › Sau 60 phút") và mang vector ngay trong đoạn. Truy xuất = độ gần nghĩa (embedding) cộng
một phần khớp từ khóa (TF-IDF); đoạn dưới ngưỡng gần nghĩa thì bị bỏ. Ngưỡng không tách sạch được mọi câu lạc
đề (đo bằng scripts/eval_rag.py), nên lớp chặn cuối là prompt của trợ lý.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

import numpy as np

from app.application.dto.request.models import KnowledgeDocumentRequest
from app.application.dto.response.models import KnowledgeIngestResponse
from app.application.port.persistence import AiDocumentRepository
from app.application.port.text_embedding import TextEmbedder
from app.application.service.model_registry import get_or_create_active_model
from app.application.service.service_config import RagConfig
from app.application.service.vector_search import tfidf_scores, tokenize
from app.domain.entity.models import KnowledgeChunk, KnowledgeDocument
from app.domain.enums import AiCapability, EmbeddingSourceType

logger = logging.getLogger(__name__)

KNOWLEDGE_SOURCE_PREFIX = "knowledge/"
HEADING_SEPARATOR = " › "
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

@dataclass(frozen=True)
class RetrievedChunk:
    chunk: KnowledgeChunk
    document: KnowledgeDocument
    score: float
    # Chỉ độ gần nghĩa (score còn cộng khớp từ khóa để xếp hạng).
    relevance: float = 0.0


@dataclass
class KnowledgeSyncReport:
    indexed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


class RagChatbotService:
    def __init__(
        self,
        repository: AiDocumentRepository,
        embedder: TextEmbedder,
        config: RagConfig,
    ) -> None:
        self.repository = repository
        self.embedder = embedder
        self.config = config

    # ---- Nạp tài liệu ----

    def ingest(self, request: KnowledgeDocumentRequest) -> KnowledgeIngestResponse:
        document_id = document_id_for(request.source_url or f"{request.title}:{request.version}")
        pieces = [
            (heading, text)
            for heading, section in split_markdown(request.content)
            for text in chunk_text(section)
        ]
        if not pieces:
            pieces = [(None, request.content.strip())]
        vectors = self.embedder.embed_passages(
            [_embedding_text(request.title, heading, text) for heading, text in pieces]
        )
        chunks = [
            KnowledgeChunk(
                # Mã cố định theo (tài liệu, vị trí): nạp lại thì ghi đè, không đẻ thêm đoạn mới.
                chunk_id=uuid5(document_id, str(index)),
                document_id=document_id,
                content=text,
                heading=heading,
                chunk_index=index,
                token_count=len(tokenize(text)),
                embedding=vector,
            )
            for index, ((heading, text), vector) in enumerate(zip(pieces, vectors))
        ]
        document = KnowledgeDocument(
            document_id=document_id,
            title=request.title,
            sport_type=request.sport_type,
            source_url=request.source_url,
            version=request.version,
            content_hash=content_hash(request.content),
            active=True,
            embedding_model=self.embedder.model_name,
        )
        self.repository.save_knowledge_document(document)
        self.repository.replace_knowledge_chunks(document_id, chunks)
        get_or_create_active_model(self.repository, AiCapability.RAG)
        return KnowledgeIngestResponse(document=document, chunks_indexed=len(chunks))

    def sync_directory(self, directory: Path) -> KnowledgeSyncReport:
        """Đồng bộ kho tri thức với các file .md: file mới / đã sửa / đổi model thì nạp lại.

        Tài liệu "knowledge/..." không còn file thì bị gỡ; câu trả lời bổ sung "faq/..." giữ nguyên.
        """
        report = KnowledgeSyncReport()
        existing = self.repository.find_knowledge_documents()
        seen: set[UUID] = set()
        for path in sorted(directory.glob("*.md")):
            content = path.read_text(encoding="utf-8")
            source_url = f"{KNOWLEDGE_SOURCE_PREFIX}{path.name}"
            document_id = document_id_for(source_url)
            seen.add(document_id)
            current = existing.get(document_id)
            if (
                current is not None
                and current.content_hash == content_hash(content)
                and current.embedding_model == self.embedder.model_name
            ):
                report.unchanged.append(path.name)
                continue
            self.ingest(
                KnowledgeDocumentRequest(title=markdown_title(content, path.stem), content=content, source_url=source_url)
            )
            report.indexed.append(path.name)
        for document in existing.values():
            # Chỉ quản tài liệu đến từ thư mục; câu trả lời bổ sung của admin (faq/...) sống ở Mongo, không có file.
            if (document.source_url or "").startswith(KNOWLEDGE_SOURCE_PREFIX) and document.document_id not in seen:
                self.repository.delete_knowledge_document(document.document_id)
                report.removed.append(document.source_url or document.title)
        # Vector đời cũ (trước khi vector nằm trong đoạn) không còn ai đọc.
        self.repository.delete_embeddings(EmbeddingSourceType.KNOWLEDGE_CHUNK)
        return report

    # ---- Truy xuất ----

    def retrieve(self, question: str, sport_type=None) -> list[RetrievedChunk]:
        # Chỉ độ gần nghĩa quyết định đoạn có liên quan; khớp từ khóa chỉ giúp xếp hạng, vì TF-IDF khớp cả các từ
        # hỏi phổ biến ("thế nào", "bao nhiêu") nên để nó tự mở cửa thì câu lạc đề lọt qua.
        floor = self.embedder.relevance_floor
        relevant = [item for item in self._scored(question, sport_type) if item.relevance >= floor]
        return relevant[: self.config.max_context_chunks]

    def top_passages(self, question: str, limit: int) -> list[RetrievedChunk]:
        """Các đoạn gần nhất, KHÔNG cắt theo ngưỡng: để soạn nháp cho câu trợ lý chưa trả lời được, khi cách hỏi
        khác tài liệu tới mức tìm thường bỏ lỡ."""
        return self._scored(question, None)[:limit]

    def _scored(self, question: str, sport_type) -> list[RetrievedChunk]:
        documents = self.repository.find_knowledge_documents()
        chunks = [
            chunk
            for chunk in self.repository.find_knowledge_chunks()
            if chunk.document_id in documents
            and (sport_type is None or documents[chunk.document_id].sport_type in {None, sport_type})
        ]
        if not chunks:
            return []
        semantic = self._semantic_scores(question, chunks)
        lexical = tfidf_scores(question, [f"{chunk.heading or ''}\n{chunk.content}" for chunk in chunks])
        weight = self.config.lexical_weight
        scored = [
            RetrievedChunk(
                chunk=chunk,
                document=documents[chunk.document_id],
                score=min(1.0, (1 - weight) * max(0.0, meaning) + weight * words),
                relevance=meaning,
            )
            for chunk, meaning, words in zip(chunks, semantic, lexical)
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored

    def _semantic_scores(self, question: str, chunks: list[KnowledgeChunk]) -> list[float]:
        query = np.asarray(self.embedder.embed_query(question), dtype=np.float32)
        scores = []
        for chunk in chunks:
            # Đoạn nhúng bằng model khác (khác số chiều) bị bỏ qua cho tới lần đồng bộ sau (lúc khởi động).
            if len(chunk.embedding) != len(query):
                scores.append(0.0)
                continue
            scores.append(float(np.dot(query, np.asarray(chunk.embedding, dtype=np.float32))))
        return scores


# ---- Markdown ----

def split_markdown(content: str) -> list[tuple[str | None, str]]:
    """Cắt theo đề mục. Đề mục # (cấp 1) là tên tài liệu nên không lặp lại trong đường đề mục của đoạn."""
    sections: list[tuple[str | None, str]] = []
    path: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            heading = HEADING_SEPARATOR.join(part for part in path[1:] if part) or None
            sections.append((heading, text))
        buffer.clear()

    for line in content.splitlines():
        match = _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            # path[i] là đề mục cấp i+1; nhảy cấp (# rồi ###) thì để trống cấp bị bỏ qua.
            path = path[: level - 1] + [""] * (level - 1 - len(path[: level - 1])) + [match.group(2).strip()]
            continue
        buffer.append(line)
    flush()
    return sections


def chunk_text(content: str, max_chars: int = 900, overlap_chars: int = 150) -> list[str]:
    """Mục dài thì chia theo đoạn văn, mỗi phần gối lên phần trước một chút để không cắt mất ngữ cảnh."""
    paragraphs = [paragraph.strip() for paragraph in content.split("\n\n") if paragraph.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs or [content.strip()]:
        candidate = f"{current}\n\n{paragraph}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = f"{current[-overlap_chars:]}\n\n{paragraph}".strip()
        else:
            for start in range(0, len(paragraph), max_chars - overlap_chars):
                chunks.append(paragraph[start : start + max_chars])
            current = ""
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk.strip()]


def markdown_title(content: str, fallback: str) -> str:
    for line in content.splitlines():
        match = _HEADING.match(line)
        if match and len(match.group(1)) == 1:
            return match.group(2).strip()[:300]
    return fallback


def document_id_for(identity: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"goatsports:knowledge:{identity}")


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _embedding_text(title: str, heading: str | None, text: str) -> str:
    return f"{title}{HEADING_SEPARATOR}{heading}\n{text}" if heading else f"{title}\n{text}"


