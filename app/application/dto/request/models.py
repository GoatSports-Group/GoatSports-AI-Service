from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Any
from uuid import UUID

from pydantic import Field, model_validator

from app.domain.base import DomainModel, utc_now
from app.domain.entity.models import PLAY_FORMATS
from app.domain.enums import (
    AcceptanceDecision,
    InteractionType,
    MatchProposalStatus,
    MatchSelectionMode,
    RecommendationTargetType,
    SearchDocumentType,
    SkillLevel,
    SportType,
)


class UserInteractionRequest(DomainModel):
    target_id: UUID
    target_type: RecommendationTargetType
    interaction_type: InteractionType
    sport_type: SportType | None = None
    weight: float | None = Field(default=None, gt=0, le=100)
    metadata: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime = Field(default_factory=utc_now)


class MatchmakingQueueRequest(DomainModel):
    player_name: str = Field(min_length=1, max_length=150)
    player_avatar: str | None = Field(default=None, max_length=2048)
    sport_type: SportType
    skill_level: SkillLevel
    elo_rating: int = Field(default=1200, ge=0, le=5000)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    play_date: date
    start_time: time
    end_time: time
    timezone: str = Field(default="Asia/Ho_Chi_Minh", min_length=1, max_length=64)
    max_elo_difference: int = Field(default=300, ge=0, le=2000)
    max_distance_km: float = Field(default=10.0, gt=0, le=100)
    preferred_positions: set[str] = Field(default_factory=set)
    play_style: str = Field(default="BALANCED", min_length=1, max_length=100)
    match_count: int = Field(default=0, ge=0)
    win_rate: float = Field(default=0.0, ge=0, le=1)
    selection_mode: MatchSelectionMode = MatchSelectionMode.AI
    # Bỏ trống thì hiểu là đánh đơn (client cũ); môn đồng đội bắt buộc phải chọn.
    play_format: str | None = Field(default=None, max_length=40)
    partner_id: UUID | None = None
    club_id: UUID | None = None

    @model_validator(mode="after")
    def validate_schedule(self) -> MatchmakingQueueRequest:
        if self.end_time <= self.start_time:
            raise ValueError("endTime must be after startTime")
        if self.play_date < datetime.now(timezone.utc).date():
            raise ValueError("playDate cannot be in the past")
        self.play_format = self.play_format or f"{self.sport_type.value}_SINGLES"
        format_spec = PLAY_FORMATS.get(self.play_format)
        if format_spec is None or format_spec[0] != self.sport_type:
            raise ValueError("playFormat does not belong to sportType")
        if format_spec[1] == 2 and self.partner_id is None:
            raise ValueError("partnerId is required for doubles")
        if format_spec[1] > 2 and self.club_id is None:
            raise ValueError("clubId is required for team formats")
        return self

    @property
    def side_size(self) -> int:
        return PLAY_FORMATS[self.play_format][1]


class MatchAcceptanceRequest(DomainModel):
    decision: AcceptanceDecision


class MatchProposalUpdateRequest(DomainModel):
    venue_id: UUID | None = None
    venue_court_id: UUID | None = None
    booking_id: UUID | None = None
    status: MatchProposalStatus | None = None


class MatchVenueSelectionRequest(DomainModel):
    venue_id: UUID
    venue_court_id: UUID


class MatchBookingRegistrationRequest(DomainModel):
    booking_id: UUID


class MatchLifecycleSyncRequest(DomainModel):
    booking_id: UUID
    status: str = Field(min_length=1, max_length=50)
    reason: str | None = Field(default=None, max_length=50)


class MatchResultRequest(DomainModel):
    my_score: int = Field(ge=0, le=999)
    opponent_score: int = Field(ge=0, le=999)


class OpponentFeedbackRequest(DomainModel):
    rating: int = Field(ge=1, le=5)
    fair_play_rating: int = Field(ge=1, le=5)
    comment: str | None = Field(default=None, max_length=500)


class SemanticDocumentRequest(DomainModel):
    document_type: SearchDocumentType
    source_id: UUID
    content: str = Field(min_length=1, max_length=100_000)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_updated_at: datetime = Field(default_factory=utc_now)


class KnowledgeDocumentRequest(DomainModel):
    title: str = Field(min_length=1, max_length=300)
    content: str = Field(min_length=1, max_length=500_000)
    sport_type: SportType | None = None
    source_url: str | None = Field(default=None, max_length=2048)
    version: str = Field(default="1.0", min_length=1, max_length=50)


class DemandForecastRequest(DomainModel):
    venue_id: UUID
    sport_type: SportType
    forecast_start: datetime
    forecast_end: datetime
    bucket_minutes: int = Field(default=60, ge=30, le=1440)

    @model_validator(mode="after")
    def validate_forecast_window(self) -> DemandForecastRequest:
        if self.forecast_end <= self.forecast_start:
            raise ValueError("forecastEnd must be after forecastStart")
        if (self.forecast_end - self.forecast_start).total_seconds() > 31 * 86_400:
            raise ValueError("forecast window cannot exceed 31 days")
        return self


class ChatbotQuery(DomainModel):
    message: str = Field(min_length=1, max_length=2000)
    user_location_lat: float | None = None
    user_location_lng: float | None = None
    favorite_sport: SportType | None = None


class IdentityVerificationRequest(DomainModel):
    subject_id: UUID
    identity_front_key: str = Field(min_length=1, max_length=2048)
    identity_back_key: str = Field(min_length=1, max_length=2048)
    identity_front_url: str = Field(min_length=1, max_length=8192)
    identity_back_url: str = Field(min_length=1, max_length=8192)
    liveness_frames: list[str] = Field(min_length=3, max_length=12)

    @model_validator(mode="after")
    def validate_liveness_frames(self) -> IdentityVerificationRequest:
        for frame in self.liveness_frames:
            if not frame.startswith("data:image/") or ";base64," not in frame:
                raise ValueError("livenessFrames must contain base64 image data URLs")
            if len(frame) > 2_800_000:
                raise ValueError("A liveness frame cannot exceed 2 MB")
        return self


class FaceReadinessRequest(DomainModel):
    subject_id: UUID
    frame: str = Field(min_length=32, max_length=1_100_000)

    @model_validator(mode="after")
    def validate_frame(self) -> FaceReadinessRequest:
        if not self.frame.startswith("data:image/") or ";base64," not in self.frame:
            raise ValueError("frame must contain a base64 image data URL")
        return self
