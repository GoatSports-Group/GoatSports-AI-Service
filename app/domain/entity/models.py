from __future__ import annotations

from datetime import date, datetime, time
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import Field, model_validator

from app.domain.base import DomainModel, utc_now
from app.domain.enums import (
    AcceptanceDecision,
    AiCapability,
    AiModelStatus,
    EmbeddingSourceType,
    IdentityDocumentType,
    IdentityVerificationStatus,
    InteractionType,
    MatchProposalStatus,
    MatchSessionStatus,
    ParticipantType,
    QueueStatus,
    RecommendationTargetType,
    SearchDocumentType,
    SkillLevel,
    SportType,
)


class MatchPreference(DomainModel):
    sport_type: SportType
    play_date: date
    start_time: time
    end_time: time
    timezone: str = Field(default="Asia/Ho_Chi_Minh", min_length=1, max_length=64)
    max_elo_difference: int = Field(default=300, ge=0, le=2000)
    max_distance_km: float = Field(default=10.0, gt=0, le=100)

    @model_validator(mode="after")
    def validate_time_range(self) -> MatchPreference:
        if self.end_time <= self.start_time:
            raise ValueError("endTime must be after startTime")
        return self


class MatchParticipantProfile(DomainModel):
    participant_profile_id: UUID = Field(default_factory=uuid4)
    participant_id: UUID
    participant_type: ParticipantType
    name: str = Field(min_length=1, max_length=150)
    avatar_url: str | None = Field(default=None, max_length=2048)
    sport_type: SportType
    elo_rating: int = Field(default=1200, ge=0, le=5000)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    snapshot_at: datetime = Field(default_factory=utc_now)


class PlayerProfile(MatchParticipantProfile):
    participant_type: Literal[ParticipantType.PLAYER] = ParticipantType.PLAYER
    skill_level: SkillLevel = SkillLevel.INTERMEDIATE
    preferred_positions: set[str] = Field(default_factory=set)
    play_radius_km: float = Field(default=10.0, gt=0, le=100)
    play_style: str = Field(default="BALANCED", min_length=1, max_length=100)
    match_count: int = Field(default=0, ge=0)
    win_rate: float = Field(default=0.0, ge=0, le=1)


class ClubMatchProfile(MatchParticipantProfile):
    participant_type: Literal[ParticipantType.CLUB] = ParticipantType.CLUB
    active_member_count: int = Field(default=0, ge=0)
    preferred_format: str = Field(min_length=1, max_length=100)
    match_count: int = Field(default=0, ge=0)
    win_rate: float = Field(default=0.0, ge=0, le=1)


class MatchAcceptance(DomainModel):
    acceptance_id: UUID = Field(default_factory=uuid4)
    participant_profile_id: UUID
    decision: AcceptanceDecision
    decided_at: datetime = Field(default_factory=utc_now)


class MatchProposal(DomainModel):
    proposal_id: UUID = Field(default_factory=uuid4)
    designated_booker_id: UUID
    conversation_id: UUID | None = None
    venue_id: UUID | None = None
    venue_court_id: UUID | None = None
    booking_id: UUID | None = None
    status: MatchProposalStatus = MatchProposalStatus.CREATED
    created_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime


class MatchmakingSession(DomainModel):
    session_id: UUID = Field(default_factory=uuid4)
    sport_type: SportType
    play_date: date
    start_time: time
    end_time: time
    timezone: str
    compatibility_score: float = Field(ge=0, le=100)
    distance_km: float = Field(ge=0)
    elo_difference: int = Field(ge=0)
    model_version_id: UUID | None = None
    status: MatchSessionStatus = MatchSessionStatus.PROPOSED
    matched_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime
    accepted_at: datetime | None = None
    participants: list[PlayerProfile | ClubMatchProfile] = Field(
        min_length=2,
        max_length=2,
    )
    acceptances: list[MatchAcceptance] = Field(default_factory=list, max_length=2)
    proposal: MatchProposal | None = None


class MatchmakingQueueEntry(DomainModel):
    queue_entry_id: UUID = Field(default_factory=uuid4)
    profile_id: UUID
    preference: MatchPreference
    matched_session_id: UUID | None = None
    status: QueueStatus = QueueStatus.QUEUED
    queued_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime


class AIModelVersion(DomainModel):
    model_version_id: UUID = Field(default_factory=uuid4)
    model_name: str = Field(min_length=1, max_length=150)
    version: str = Field(min_length=1, max_length=50)
    capability: AiCapability
    trained_at: datetime | None = None
    training_cutoff: datetime | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    artifact_uri: str | None = Field(default=None, max_length=2048)
    status: AiModelStatus


class DemandForecastResult(DomainModel):
    forecast_id: UUID = Field(default_factory=uuid4)
    venue_id: UUID
    sport_type: SportType
    bucket_start: datetime
    bucket_end: datetime
    predicted_demand: float
    lower_bound: float
    upper_bound: float
    model_version_id: UUID
    generated_at: datetime = Field(default_factory=utc_now)


class EmbeddingRecord(DomainModel):
    embedding_id: UUID = Field(default_factory=uuid4)
    source_type: EmbeddingSourceType
    source_id: UUID
    model_version_id: UUID
    dimensions: int = Field(gt=0)
    vector: list[float]
    embedded_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_dimensions(self) -> EmbeddingRecord:
        if len(self.vector) != self.dimensions:
            raise ValueError("dimensions must equal vector length")
        return self


class SemanticDocument(DomainModel):
    document_id: UUID = Field(default_factory=uuid4)
    document_type: SearchDocumentType
    source_id: UUID
    content: str
    content_hash: str = Field(min_length=1, max_length=128)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_updated_at: datetime
    indexed_at: datetime = Field(default_factory=utc_now)
    active: bool = True


class KnowledgeChunk(DomainModel):
    chunk_id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    content: str
    chunk_index: int = Field(ge=0)
    token_count: int = Field(ge=0)


class KnowledgeDocument(DomainModel):
    document_id: UUID = Field(default_factory=uuid4)
    title: str = Field(min_length=1, max_length=300)
    sport_type: SportType | None = None
    source_url: str | None = Field(default=None, max_length=2048)
    version: str = Field(min_length=1, max_length=50)
    content_hash: str = Field(min_length=1, max_length=128)
    active: bool = True
    updated_at: datetime = Field(default_factory=utc_now)


class DemandObservation(DomainModel):
    observation_id: UUID = Field(default_factory=uuid4)
    venue_id: UUID
    bucket_start: datetime
    bucket_end: datetime
    sport_type: SportType
    available_slot_count: int = Field(ge=0)
    booked_slot_count: int = Field(ge=0)
    cancelled_booking_count: int = Field(ge=0)
    occupancy_rate: float = Field(ge=0, le=1)
    booking_count: int = Field(ge=0)


class UserInteraction(DomainModel):
    interaction_id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    target_id: UUID
    target_type: RecommendationTargetType
    interaction_type: InteractionType
    sport_type: SportType | None = None
    weight: float | None = Field(default=None, gt=0, le=100)
    metadata: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime = Field(default_factory=utc_now)


class IdentityDocumentData(DomainModel):
    document_type: IdentityDocumentType = IdentityDocumentType.UNKNOWN
    identity_number: str | None = Field(default=None, pattern=r"^\d{12}$")
    full_name: str | None = Field(default=None, max_length=150)
    date_of_birth: date | None = None
    expiry_date: date | None = None
    gender: str | None = Field(default=None, max_length=20)
    nationality: str | None = Field(default=None, max_length=80)
    place_of_origin: str | None = Field(default=None, max_length=500)
    place_of_residence: str | None = Field(default=None, max_length=500)


class IdentityVerificationScores(DomainModel):
    front_quality: float = Field(ge=0, le=1)
    back_quality: float = Field(ge=0, le=1)
    ocr_confidence: float = Field(ge=0, le=1)
    face_similarity: float = Field(ge=-1, le=1)
    liveness_score: float = Field(ge=0, le=1)


class IdentityVerification(DomainModel):
    verification_id: UUID = Field(default_factory=uuid4)
    subject_id: UUID
    identity_front_key: str = Field(min_length=1, max_length=2048)
    identity_back_key: str = Field(min_length=1, max_length=2048)
    status: IdentityVerificationStatus = IdentityVerificationStatus.PROCESSING
    document: IdentityDocumentData | None = None
    scores: IdentityVerificationScores | None = None
    reason_code: str | None = Field(default=None, max_length=100)
    message: str | None = Field(default=None, max_length=500)
    created_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime
