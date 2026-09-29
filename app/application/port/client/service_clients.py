from datetime import date, time
from typing import Protocol
from uuid import UUID

from app.application.dto.response.models import MatchBookingSnapshot, VenueRecommendation
from app.domain.entity.models import ClubMatchProfile, SideMember
from app.domain.enums import SportType


class VenueServiceClient(Protocol):
    def recommend(
        self,
        latitude: float,
        longitude: float,
        sport_type: SportType,
        radius_km: float,
        play_date: date | None = None,
        start_time: time | None = None,
        end_time: time | None = None,
        max_results: int = 5,
    ) -> tuple[list[VenueRecommendation], UUID | None]: ...

    def get_matchmaking_booking(self, session_id: UUID) -> MatchBookingSnapshot | None: ...


class SocialServiceClient(Protocol):
    def create_match_conversation(
        self,
        session_id: UUID,
        participant_ids: list[UUID],
        creator_id: UUID,
        sport_type: SportType,
    ) -> UUID: ...

    def friend_ids(self, user_id: UUID) -> set[UUID]:
        """Những người đã là bạn (lời mời đã chấp nhận) của user_id."""
        ...


class NotificationServiceClient(Protocol):
    def notify_matchmaking(
        self,
        receiver_id: UUID,
        session_id: UUID,
        title: str,
        content: str,
    ) -> None: ...


class AuthServiceClient(Protocol):
    def apply_match_result(
        self,
        session_id: UUID,
        user_id: UUID,
        sport_type: SportType,
        opponent_elo: int,
        outcome: str,
        player_elo: int | None = None,
    ) -> int: ...

    def teammate(self, user_id: UUID, sport_type: SportType) -> SideMember:
        """Tên và ELO của đồng đội; chưa có hồ sơ môn này thì tạo ở trình độ thấp nhất."""
        ...


class ClubServiceClient(Protocol):
    def matchmaking_profile(
        self, club_id: UUID, play_format: str, latitude: float, longitude: float
    ) -> ClubMatchProfile | None: ...

    def apply_friendly_result(
        self,
        session_id: UUID,
        sport_type: SportType,
        play_format: str | None,
        first: tuple[UUID, int],
        second: tuple[UUID, int],
    ) -> dict[str, int]:
        """Ghi trận giao hữu vào lịch sử hai CLB, cộng thắng/hòa/thua và trả ELO mới theo clubId."""
        ...


class LanguageModelClient(Protocol):
    @property
    def model_name(self) -> str | None: ...
    def generate(self, question: str, context: str) -> str | None: ...