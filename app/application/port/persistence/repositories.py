from contextlib import AbstractContextManager
from typing import Protocol
from uuid import UUID

from app.domain.entity.models import (
    AIModelVersion,
    DemandForecastResult,
    DemandObservation,
    EmbeddingRecord,
    KnowledgeChunk,
    KnowledgeDocument,
    MatchmakingQueueEntry,
    MatchmakingSession,
    MatchParticipantProfile,
    SemanticDocument,
    UserInteraction,
)
from app.domain.enums import (
    AiCapability,
    EmbeddingSourceType,
    RecommendationTargetType,
    SearchDocumentType,
)


class MatchmakingSessionRepository(Protocol):
    def save(self, session: MatchmakingSession) -> MatchmakingSession: ...
    def find_by_id(self, session_id: UUID) -> MatchmakingSession | None: ...
    def find_latest_by_participant(self, participant_id: UUID) -> MatchmakingSession | None: ...
    def find_active_by_participant(self, participant_id: UUID) -> MatchmakingSession | None: ...


class MatchmakingQueueRepository(Protocol):
    def lock(self) -> AbstractContextManager: ...
    def save(
        self,
        entry: MatchmakingQueueEntry,
        profile: MatchParticipantProfile,
    ) -> MatchmakingQueueEntry: ...
    def find_by_participant(self, participant_id: UUID) -> MatchmakingQueueEntry | None: ...
    def find_profile(self, profile_id: UUID) -> MatchParticipantProfile | None: ...
    def queued_entries(self, limit: int) -> list[MatchmakingQueueEntry]: ...
    def queue_size(self) -> int: ...


class AiDocumentRepository(Protocol):
    def save_model_version(self, model: AIModelVersion) -> AIModelVersion: ...
    def find_active_model(self, capability: AiCapability) -> AIModelVersion | None: ...
    def save_forecast(self, forecast: DemandForecastResult) -> DemandForecastResult: ...
    def save_embedding(self, embedding: EmbeddingRecord) -> EmbeddingRecord: ...
    def find_embeddings(
        self,
        source_type: EmbeddingSourceType,
        source_ids: list[UUID],
    ) -> dict[UUID, EmbeddingRecord]: ...
    def vector_search_embeddings(
        self,
        source_type: EmbeddingSourceType,
        query_vector: list[float],
        index_name: str,
        limit: int,
    ) -> list[tuple[EmbeddingRecord, float]]: ...
    def save_semantic_document(self, document: SemanticDocument) -> SemanticDocument: ...
    def find_semantic_document(
        self,
        document_type: SearchDocumentType,
        source_id: UUID,
    ) -> SemanticDocument | None: ...
    def find_semantic_documents(
        self,
        document_type: SearchDocumentType | None = None,
        limit: int = 1000,
    ) -> list[SemanticDocument]: ...
    def find_semantic_documents_by_ids(
        self,
        document_ids: list[UUID],
    ) -> list[SemanticDocument]: ...
    def save_knowledge_document(self, document: KnowledgeDocument) -> KnowledgeDocument: ...
    def find_knowledge_documents(self) -> dict[UUID, KnowledgeDocument]: ...
    def replace_knowledge_chunks(
        self,
        document_id: UUID,
        chunks: list[KnowledgeChunk],
    ) -> None: ...
    def find_knowledge_chunks(self, limit: int = 2000) -> list[KnowledgeChunk]: ...
    def find_knowledge_chunks_by_ids(self, chunk_ids: list[UUID]) -> list[KnowledgeChunk]: ...
    def save_demand_observation(self, observation: DemandObservation) -> DemandObservation: ...
    def find_demand_observations(
        self,
        venue_id: UUID,
        sport_type: str,
        limit: int = 10_000,
    ) -> list[DemandObservation]: ...
    def find_forecasts(
        self,
        venue_id: UUID,
        sport_type: str,
        limit: int = 500,
    ) -> list[DemandForecastResult]: ...
    def save_interaction(self, interaction: UserInteraction) -> UserInteraction: ...
    def find_interactions(
        self,
        user_id: UUID | None = None,
        target_type: RecommendationTargetType | None = None,
        limit: int = 20_000,
    ) -> list[UserInteraction]: ...
