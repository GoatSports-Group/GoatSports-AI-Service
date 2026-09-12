from uuid import UUID

from pydantic import Field

from app.domain.base import DomainModel
from app.domain.entity.models import IdentityVerification, KnowledgeDocument, MatchmakingSession
from app.domain.enums import (
    AiCapability,
    MatchSessionStatus,
    QueueStatus,
    RecommendationTargetType,
    SearchDocumentType,
)


class AiCapabilityStatus(DomainModel):
    capability: AiCapability
    algorithm: str
    storage: str
    status: str
    detail: str


class QueueJoinResponse(DomainModel):
    status: QueueStatus
    message: str
    session: MatchmakingSession | None = None


class QueueStatusResponse(DomainModel):
    status: QueueStatus | MatchSessionStatus | str
    queue_size: int | None = None
    session: MatchmakingSession | None = None


class ActionResponse(DomainModel):
    success: bool
    message: str


class VenueRecommendation(DomainModel):
    venue_id: UUID
    venue_name: str
    sport_types: list[str] = Field(default_factory=list)
    address: str
    distance_km: float
    rating: float
    min_price: float | None = None
    max_price: float | None = None
    score: float
    match_reason: str
    suggested_courts: list[str] = Field(default_factory=list)


class PersonalizedRecommendation(DomainModel):
    target_id: UUID
    target_type: RecommendationTargetType
    score: float = Field(ge=0, le=100)
    reason: str


class SemanticSearchResult(DomainModel):
    document_id: UUID
    document_type: SearchDocumentType
    source_id: UUID
    content: str
    score: float = Field(ge=0, le=100)
    tfidf_score: float = Field(ge=0, le=1)
    embedding_score: float = Field(ge=0, le=1)
    metadata: dict = Field(default_factory=dict)


class KnowledgeIngestResponse(DomainModel):
    document: KnowledgeDocument
    chunks_indexed: int = Field(ge=1)


class RagSource(DomainModel):
    document_id: UUID
    title: str
    source_url: str | None = None
    chunk_index: int = Field(ge=0)
    relevance_score: float = Field(ge=0, le=1)


class ChatbotResponse(DomainModel):
    answer: str
    suggested_actions: list[str] = Field(default_factory=list)
    recommended_venues: list[VenueRecommendation] = Field(default_factory=list)
    sources: list[RagSource] = Field(default_factory=list)
    grounded: bool = False
    model: str | None = None


class IdentityVerificationResponse(DomainModel):
    verification: IdentityVerification
