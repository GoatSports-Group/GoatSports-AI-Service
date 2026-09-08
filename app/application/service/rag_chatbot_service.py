from __future__ import annotations

import hashlib
from dataclasses import dataclass
from uuid import NAMESPACE_URL, UUID, uuid5

from app.application.dto.request.models import ChatbotQuery, KnowledgeDocumentRequest
from app.application.dto.response.models import (
    ChatbotResponse,
    KnowledgeIngestResponse,
    RagSource,
)
from app.application.port.client import LanguageModelClient
from app.application.port.persistence import AiDocumentRepository
from app.application.service.model_registry import get_or_create_active_model
from app.application.service.recommendation_service import RecommendationService
from app.application.service.service_config import RagConfig
from app.application.service.vector_search import hybrid_text_scores, local_embedding, tokenize
from app.domain.entity.models import (
    EmbeddingRecord,
    KnowledgeChunk,
    KnowledgeDocument,
)
from app.domain.enums import AiCapability, EmbeddingSourceType


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: KnowledgeChunk
    document: KnowledgeDocument
    score: float


class RagChatbotService:
    def __init__(
        self,
        repository: AiDocumentRepository,
        recommendation_service: RecommendationService,
        language_model_client: LanguageModelClient,
        config: RagConfig,
    ) -> None:
        self.repository = repository
        self.recommendation_service = recommendation_service
        self.language_model_client = language_model_client
        self.config = config

    def ingest(self, request: KnowledgeDocumentRequest) -> KnowledgeIngestResponse:
        identity = request.source_url or f"{request.title}:{request.version}"
        document_id = uuid5(NAMESPACE_URL, f"goatsports:knowledge:{identity}")
        content_hash = hashlib.sha256(request.content.encode("utf-8")).hexdigest()
        document = KnowledgeDocument(
            document_id=document_id,
            title=request.title,
            sport_type=request.sport_type,
            source_url=request.source_url,
            version=request.version,
            content_hash=content_hash,
            active=True,
        )
        chunks = [
            KnowledgeChunk(
                document_id=document_id,
                content=content,
                chunk_index=index,
                token_count=len(tokenize(content)),
            )
            for index, content in enumerate(self._chunk_text(request.content))
        ]
        self.repository.save_knowledge_document(document)
        self.repository.replace_knowledge_chunks(document_id, chunks)

        model = get_or_create_active_model(self.repository, AiCapability.RAG)
        for chunk in chunks:
            vector = local_embedding(chunk.content)
            self.repository.save_embedding(
                EmbeddingRecord(
                    source_type=EmbeddingSourceType.KNOWLEDGE_CHUNK,
                    source_id=chunk.chunk_id,
                    model_version_id=model.model_version_id,
                    dimensions=len(vector),
                    vector=vector,
                )
            )
        return KnowledgeIngestResponse(document=document, chunks_indexed=len(chunks))

    def answer_query(
        self,
        query: ChatbotQuery,
        user_id: UUID | None = None,
    ) -> ChatbotResponse:
        retrieved = self.retrieve(query.message, query.favorite_sport)
        venues = self._recommend_venues(query, user_id)
        sources = [
            RagSource(
                document_id=item.document.document_id,
                title=item.document.title,
                source_url=item.document.source_url,
                chunk_index=item.chunk.chunk_index,
                relevance_score=round(item.score, 6),
            )
            for item in retrieved
        ]

        if not retrieved:
            return ChatbotResponse(
                answer=(
                    "Mình chưa tìm thấy tài liệu đủ liên quan trong kho tri thức GoatSports để trả lời "
                    "chắc chắn. Bạn có thể diễn đạt cụ thể hơn hoặc chọn một nội dung gợi ý."
                ),
                suggested_actions=self._suggested_actions(query, bool(venues)),
                recommended_venues=venues,
                sources=[],
                grounded=False,
                model=None,
            )

        answer, model_name = self._generate_answer(query.message, retrieved)
        return ChatbotResponse(
            answer=answer,
            suggested_actions=self._suggested_actions(query, bool(venues)),
            recommended_venues=venues,
            sources=sources,
            grounded=True,
            model=model_name,
        )

    def retrieve(self, question: str, sport_type=None) -> list[RetrievedChunk]:
        documents = self.repository.find_knowledge_documents()
        chunks = []
        embeddings = {}
        if self.config.vector_search.enabled:
            hits = self.repository.vector_search_embeddings(
                EmbeddingSourceType.KNOWLEDGE_CHUNK,
                local_embedding(question),
                self.config.vector_search.index_name,
                max(self.config.max_context_chunks * 10, 50),
            )
            embeddings = {record.source_id: record for record, _ in hits}
            chunks = self.repository.find_knowledge_chunks_by_ids(list(embeddings))
        if not chunks:
            chunks = self.repository.find_knowledge_chunks()
            embeddings = self.repository.find_embeddings(
                EmbeddingSourceType.KNOWLEDGE_CHUNK,
                [chunk.chunk_id for chunk in chunks],
            )
        chunks = [
            chunk
            for chunk in chunks
            if chunk.document_id in documents
            and (sport_type is None or documents[chunk.document_id].sport_type in {None, sport_type})
        ]
        if not chunks and self.config.vector_search.enabled:
            chunks = [
                chunk
                for chunk in self.repository.find_knowledge_chunks()
                if chunk.document_id in documents
                and (
                    sport_type is None
                    or documents[chunk.document_id].sport_type in {None, sport_type}
                )
            ]
            embeddings = self.repository.find_embeddings(
                EmbeddingSourceType.KNOWLEDGE_CHUNK,
                [chunk.chunk_id for chunk in chunks],
            )
        vectors = [
            embeddings[chunk.chunk_id].vector
            if chunk.chunk_id in embeddings
            else local_embedding(chunk.content)
            for chunk in chunks
        ]
        scores = hybrid_text_scores(question, [chunk.content for chunk in chunks], vectors)
        ranked = sorted(zip(chunks, scores), key=lambda item: item[1][0], reverse=True)
        return [
            RetrievedChunk(chunk=chunk, document=documents[chunk.document_id], score=combined)
            for chunk, (combined, _, _) in ranked[: self.config.max_context_chunks]
            if combined >= self.config.minimum_relevance_score
        ]

    def _generate_answer(
        self,
        question: str,
        retrieved: list[RetrievedChunk],
    ) -> tuple[str, str | None]:
        context = "\n\n".join(
            f"[Nguồn {index}: {item.document.title}]\n{item.chunk.content}"
            for index, item in enumerate(retrieved, start=1)
        )
        content = self.language_model_client.generate(question, context)
        if content:
            return content, self.language_model_client.model_name

        best = retrieved[0]
        excerpt = best.chunk.content.strip()
        if len(excerpt) > 900:
            excerpt = excerpt[:900].rsplit(" ", 1)[0] + "…"
        return f"Theo tài liệu “{best.document.title}”: {excerpt} [Nguồn 1]", "extractive-fallback"

    @staticmethod
    def _chunk_text(content: str, max_chars: int = 1200, overlap_chars: int = 180) -> list[str]:
        paragraphs = [paragraph.strip() for paragraph in content.splitlines() if paragraph.strip()]
        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs or [content.strip()]:
            candidate = f"{current}\n{paragraph}".strip()
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                chunks.append(current)
                prefix = current[-overlap_chars:]
                current = f"{prefix}\n{paragraph}".strip()
            else:
                for start in range(0, len(paragraph), max_chars - overlap_chars):
                    chunks.append(paragraph[start : start + max_chars])
                current = ""
        if current:
            chunks.append(current)
        return chunks or [content.strip()]

    def _recommend_venues(self, query: ChatbotQuery, user_id: UUID | None):
        venue_terms = ("gần", "sân", "ở đâu", "địa chỉ", "giá", "lân cận")
        if not any(term in query.message.lower() for term in venue_terms):
            return []
        return self.recommendation_service.get_recommendations(
            user_id=user_id,
            user_lat=query.user_location_lat,
            user_lng=query.user_location_lng,
            sport_type=query.favorite_sport,
            max_results=3,
        )

    @staticmethod
    def _suggested_actions(query: ChatbotQuery, has_venues: bool) -> list[str]:
        actions = ["Hỏi về luật thi đấu", "Hỏi về kỹ thuật luyện tập"]
        if has_venues:
            actions.insert(0, "Xem sân phù hợp")
        if "ghép" in query.message.lower() or "đối thủ" in query.message.lower():
            actions.insert(0, "Tìm đối thủ")
        return actions[:3]

