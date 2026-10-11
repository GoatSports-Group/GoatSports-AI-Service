from datetime import date, datetime, time
from uuid import UUID

from pydantic import Field

from app.domain.base import DomainModel
from app.domain.entity.models import (
    AIModelVersion,
    IdentityVerification,
    KnowledgeDocument,
    MatchmakingSession,
    MatchParticipantProfile,
    MatchScoreBreakdown,
)
from app.domain.enums import (
    AiCapability,
    MatchSessionStatus,
    QueueStatus,
    RecommendationTargetType,
    SearchDocumentType,
    SportType,
)


class AiCapabilityStatus(DomainModel):
    capability: AiCapability
    algorithm: str
    storage: str
    status: str
    detail: str


class AiCapabilityOverview(AiCapabilityStatus):
    # Phiên bản model đang phục vụ (đã học/đã nạp); năng lực dùng thuật toán cố định thì không có.
    active_model: AIModelVersion | None = None


class AiDependencyStatus(DomainModel):
    name: str
    status: str  # UP | DOWN
    latency_ms: float | None = None
    detail: str | None = None


class AiAdminStatus(DomainModel):
    checked_at: datetime
    dependencies: list[AiDependencyStatus]
    capabilities: list[AiCapabilityOverview]


class QueueSearchMember(DomainModel):
    user_id: UUID
    name: str


class QueueSearchInfo(DomainModel):
    """Lượt tìm đang chờ như người dẫn đã đặt: đồng đội xem đúng môn, giờ và danh sách đội."""

    sport_type: SportType
    play_format: str | None = None
    play_date: date
    start_time: time
    end_time: time
    queued_at: datetime
    queued_by: UUID | None = None
    leader_name: str | None = None
    members: list[QueueSearchMember] = Field(default_factory=list)


class MatchmakingModelStatus(DomainModel):
    """Hệ thống đang xếp hạng đối thủ bằng mô hình học máy hay bằng trọng số viết tay."""

    mode: str  # LEARNED | RULES
    model_name: str | None = None
    version: str | None = None
    trained_at: datetime | None = None
    auc: float | None = None
    trained_samples: int | None = None
    samples: int  # số kèo đã có kết quả hiện giờ
    positives: int
    min_samples: int
    min_per_class: int
    weights: dict[str, float]  # tỷ trọng (%) theo tiêu chí


class QueueJoinResponse(DomainModel):
    status: QueueStatus
    message: str
    session: MatchmakingSession | None = None
    search: QueueSearchInfo | None = None


class QueueStatusResponse(DomainModel):
    status: QueueStatus | MatchSessionStatus | str
    queue_size: int | None = None
    session: MatchmakingSession | None = None
    search: QueueSearchInfo | None = None


class SharedPostResponse(DomainModel):
    post_id: UUID


class ActionResponse(DomainModel):
    success: bool
    message: str


class MatchCandidate(DomainModel):
    participant: MatchParticipantProfile
    score_breakdown: MatchScoreBreakdown
    distance_km: float = Field(ge=0)
    elo_difference: int = Field(ge=0)
    start_time: str
    end_time: str


class DemandHistorySnapshot(DomainModel):
    """Lịch sử lấp đầy theo giờ từ venue-service; None = giờ chưa có dữ liệu (trước khi cơ sở được tạo)."""

    venue_id: UUID
    venue_name: str
    start: datetime
    court_count: int
    open_time: time | None = None
    close_time: time | None = None
    occupancy: list[float | None]


class MatchBookingSnapshot(DomainModel):
    booking_id: UUID
    matchmaking_session_id: UUID
    player_id: UUID
    venue_court_id: UUID
    status: str


class VenueRecommendation(DomainModel):
    venue_id: UUID
    venue_court_id: UUID | None = None
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
    # Slot con trong som nhat nam tron trong khung gio ranh chung; tran dau dai dung mot slot.
    slot_start: time | None = None
    slot_end: time | None = None


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


class FaceReadinessResponse(DomainModel):
    ready: bool
    reason_code: str
    message: str
    face_count: int
    brightness: float
    centered: bool
    frontal: bool
    glasses_free: bool
    eyes_open: bool
    mask_free: bool
    face_box: tuple[float, float, float, float] | None = None
    yaw_ratio: float
    pitch_ratio: float
    roll_degrees: float
