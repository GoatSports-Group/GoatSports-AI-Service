from __future__ import annotations

import hashlib

from app.application.dto.request.models import SemanticDocumentRequest
from app.application.dto.response.models import SemanticSearchResult
from app.application.port.persistence import AiDocumentRepository
from app.application.service.model_registry import get_or_create_active_model
from app.application.service.service_config import VectorSearchConfig
from app.application.service.vector_search import hybrid_text_scores, local_embedding
from app.domain.entity.models import (
    EmbeddingRecord,
    SemanticDocument,
)
from app.domain.enums import AiCapability, EmbeddingSourceType, SearchDocumentType


class SemanticSearchService:
    def __init__(
        self,
        repository: AiDocumentRepository,
        vector_search_config: VectorSearchConfig,
    ) -> None:
        self.repository = repository
        self.vector_search_config = vector_search_config

    def index(self, request: SemanticDocumentRequest) -> SemanticDocument:
        current = self.repository.find_semantic_document(
            request.document_type,
            request.source_id,
        )
        values = {
            "document_type": request.document_type,
            "source_id": request.source_id,
            "content": request.content,
            "content_hash": hashlib.sha256(request.content.encode("utf-8")).hexdigest(),
            "metadata": request.metadata,
            "source_updated_at": request.source_updated_at,
        }
        if current:
            values["document_id"] = current.document_id
        document = SemanticDocument.model_validate(values)
        self.repository.save_semantic_document(document)
        model = get_or_create_active_model(self.repository, AiCapability.SEMANTIC_SEARCH)
        vector = local_embedding(document.content)
        self.repository.save_embedding(
            EmbeddingRecord(
                source_type=EmbeddingSourceType.SEMANTIC_DOCUMENT,
                source_id=document.document_id,
                model_version_id=model.model_version_id,
                dimensions=len(vector),
                vector=vector,
            )
        )
        return document

    def search(
        self,
        query: str,
        document_type: SearchDocumentType | None = None,
        max_results: int = 20,
        min_score: float = 0.05,
    ) -> list[SemanticSearchResult]:
        documents = []
        embeddings = {}
        if self.vector_search_config.enabled:
            hits = self.repository.vector_search_embeddings(
                EmbeddingSourceType.SEMANTIC_DOCUMENT,
                local_embedding(query),
                self.vector_search_config.index_name,
                max(max_results * 10, 100),
            )
            embeddings = {record.source_id: record for record, _ in hits}
            documents = self.repository.find_semantic_documents_by_ids(list(embeddings))
            if document_type:
                documents = [
                    document
                    for document in documents
                    if document.document_type == document_type
                ]
        if not documents:
            documents = self.repository.find_semantic_documents(document_type=document_type)
            embeddings = self.repository.find_embeddings(
                EmbeddingSourceType.SEMANTIC_DOCUMENT,
                [document.document_id for document in documents],
            )
        vectors = [
            embeddings[document.document_id].vector
            if document.document_id in embeddings
            else local_embedding(document.content)
            for document in documents
        ]
        scores = hybrid_text_scores(query, [document.content for document in documents], vectors)
        ranked = sorted(
            zip(documents, scores),
            key=lambda item: item[1][0],
            reverse=True,
        )
        return [
            SemanticSearchResult(
                document_id=document.document_id,
                document_type=document.document_type,
                source_id=document.source_id,
                content=document.content,
                score=round(combined * 100.0, 2),
                tfidf_score=round(tfidf, 6),
                embedding_score=round(embedding, 6),
                metadata=document.metadata,
            )
            for document, (combined, tfidf, embedding) in ranked[:max_results]
            if combined >= min_score
        ]
