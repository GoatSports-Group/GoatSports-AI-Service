from datetime import date, time
from typing import Protocol
from uuid import UUID

from app.application.dto.response.models import MatchBookingSnapshot, VenueRecommendation
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


class LanguageModelClient(Protocol):
    @property
    def model_name(self) -> str | None: ...
    def generate(self, question: str, context: str) -> str | None: ...