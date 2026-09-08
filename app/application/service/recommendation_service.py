from __future__ import annotations

from uuid import UUID

from app.application.dto.response.models import VenueRecommendation
from app.application.port.client import VenueServiceClient
from app.application.port.persistence import AiDocumentRepository
from app.application.service.model_registry import get_or_create_active_model
from app.application.service.vector_search import cosine_similarity
from app.domain.entity.models import UserInteraction
from app.domain.enums import (
    AiCapability,
    RecommendationTargetType,
    SportType,
)


class RecommendationService:
    def __init__(
        self,
        venue_client: VenueServiceClient,
        repository: AiDocumentRepository,
    ) -> None:
        self.venue_client = venue_client
        self.repository = repository

    def get_recommendations(
        self,
        user_id: UUID | None = None,
        user_lat: float | None = None,
        user_lng: float | None = None,
        sport_type: SportType | None = None,
        max_results: int = 5,
    ) -> list[VenueRecommendation]:
        if user_lat is None or user_lng is None or sport_type is None:
            return []
        get_or_create_active_model(self.repository, AiCapability.VENUE_RECOMMENDATION)
        candidates, _ = self.venue_client.recommend(
            user_lat,
            user_lng,
            sport_type,
            radius_km=30,
            max_results=max(max_results * 4, 20),
        )
        history = (
            self.repository.find_interactions(
                user_id=user_id,
                target_type=RecommendationTargetType.VENUE,
            )
            if user_id
            else []
        )
        user_vector = self._user_preference_vector(history, sport_type)
        for candidate in candidates:
            habit_score = max(0.0, cosine_similarity(user_vector, self._venue_vector(candidate)))
            distance_score = max(0.0, 1.0 - candidate.distance_km / 30.0)
            rating_score = max(0.0, min(1.0, candidate.rating / 5.0))
            repeat_affinity = self._repeat_affinity(history, candidate.venue_id)
            candidate.score = round(
                min(
                    100.0,
                    (habit_score * 0.45 + distance_score * 0.30 + rating_score * 0.15 + repeat_affinity * 0.10)
                    * 100.0,
                ),
                2,
            )
            candidate.match_reason = (
                f"Tương đồng thói quen {habit_score * 100:.0f}%, cách bạn "
                f"{candidate.distance_km:.1f} km và được đánh giá {candidate.rating:.1f}/5."
            )
        return sorted(candidates, key=lambda item: item.score, reverse=True)[:max_results]

    @staticmethod
    def _user_preference_vector(
        history: list[UserInteraction],
        requested_sport: SportType,
    ) -> list[float]:
        sports = list(SportType)
        sport_weights = {sport: 0.0 for sport in sports}
        rating_total = 0.0
        price_total = 0.0
        metadata_count = 0
        for interaction in history:
            sport = interaction.sport_type
            if sport:
                sport_weights[sport] += interaction.weight or 1.0
            rating = interaction.metadata.get("rating")
            price = interaction.metadata.get("price")
            if isinstance(rating, (int, float)) and isinstance(price, (int, float)):
                rating_total += max(0.0, min(5.0, float(rating))) / 5.0
                price_total += max(0.0, min(1.0, float(price) / 500_000.0))
                metadata_count += 1
        sport_weights[requested_sport] += 1.0
        peak = max(sport_weights.values()) or 1.0
        rating_preference = rating_total / metadata_count if metadata_count else 0.8
        price_preference = price_total / metadata_count if metadata_count else 0.5
        return [sport_weights[sport] / peak for sport in sports] + [rating_preference, price_preference]

    @staticmethod
    def _venue_vector(venue: VenueRecommendation) -> list[float]:
        supported = set(venue.sport_types)
        sports = [1.0 if sport.value in supported else 0.0 for sport in SportType]
        average_price = ((venue.min_price or 0) + (venue.max_price or venue.min_price or 0)) / 2
        return sports + [venue.rating / 5.0, min(1.0, average_price / 500_000.0)]

    @staticmethod
    def _repeat_affinity(history: list[UserInteraction], venue_id: UUID) -> float:
        repeated = sum(1 for interaction in history if interaction.target_id == venue_id)
        return min(1.0, repeated / 3.0)

