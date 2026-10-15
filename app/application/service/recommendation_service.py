from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from app.application.dto.response.models import VenueRecommendation
from app.application.port.client import VenueServiceClient
from app.application.port.persistence import AiDocumentRepository
from app.application.service.model_registry import get_or_create_active_model
from app.application.service.post_ranking_service import effective_weight
from app.application.service.vector_search import cosine_similarity
from app.domain.entity.models import UserInteraction
from app.domain.enums import (
    AiCapability,
    InteractionType,
    RecommendationTargetType,
    SportType,
)

# Trung bình Bayes: sân ít đánh giá bị kéo về mức chung, để 5.0 từ 1 lượt không vượt 4.5 từ 40 lượt.
PRIOR_RATING = 3.5
PRIOR_REVIEWS = 5
# Ba lượt đặt gần đây (trọng số 6) là "khách quen" hẳn.
REGULAR_WEIGHT = 18.0


def bayesian_rating(rating: float, reviews: int) -> float:
    reviews = max(0, reviews)
    return (rating * reviews + PRIOR_RATING * PRIOR_REVIEWS) / (reviews + PRIOR_REVIEWS)


class RecommendationService:
    def __init__(
        self,
        venue_client: VenueServiceClient,
        repository: AiDocumentRepository,
        clock=None,
    ) -> None:
        self.venue_client = venue_client
        self.repository = repository
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def get_recommendations(
        self,
        user_id: UUID | None = None,
        user_lat: float | None = None,
        user_lng: float | None = None,
        sport_type: SportType | None = None,
        max_results: int = 5,
        radius_km: float = 30.0,
    ) -> list[VenueRecommendation]:
        if user_lat is None or user_lng is None or sport_type is None:
            return []
        get_or_create_active_model(self.repository, AiCapability.VENUE_RECOMMENDATION)
        candidates, _ = self.venue_client.recommend(
            user_lat,
            user_lng,
            sport_type,
            radius_km=radius_km,
            # venue-service trả tối đa 20 sân mỗi trang.
            max_results=20,
            include_courts=False,
        )
        history = (
            self.repository.find_interactions(
                user_id=user_id,
                target_type=RecommendationTargetType.VENUE,
            )
            if user_id and self.repository.is_personalization_enabled(user_id)
            else []
        )
        now = self.clock()
        weights = [(interaction, effective_weight(interaction, now)) for interaction in history]
        user_vector = self._user_preference_vector(weights, sport_type)
        nearest = min((candidate.distance_km for candidate in candidates), default=None)
        for candidate in candidates:
            habit_score = max(0.0, cosine_similarity(user_vector, self._venue_vector(candidate))) if history else 0.0
            distance_score = max(0.0, 1.0 - candidate.distance_km / radius_km)
            rating_score = bayesian_rating(candidate.rating, candidate.total_reviews) / 5.0
            mine = [(interaction, weight) for interaction, weight in weights if interaction.target_id == candidate.venue_id]
            repeat_affinity = min(1.0, sum(weight for _, weight in mine) / REGULAR_WEIGHT)
            candidate.score = round(
                min(
                    100.0,
                    (habit_score * 0.45 + distance_score * 0.30 + rating_score * 0.15 + repeat_affinity * 0.10)
                    * 100.0,
                ),
                2,
            )
            candidate.match_reason = self._reason(candidate, mine, habit_score, candidate.distance_km == nearest)
        return sorted(candidates, key=lambda item: item.score, reverse=True)[:max_results]

    @staticmethod
    def _reason(
        venue: VenueRecommendation,
        mine: list[tuple[UserInteraction, float]],
        habit_score: float,
        is_nearest: bool,
    ) -> str:
        """Vì sao sân này được gợi ý, bằng lời: không lặp khoảng cách / số sao đã có trên thẻ sân."""
        parts = []
        bookings = sum(1 for interaction, _ in mine if interaction.interaction_type == InteractionType.BOOKING)
        if bookings:
            parts.append(f"Bạn đã đặt sân ở đây {bookings} lần")
        elif mine:
            parts.append("Bạn đã xem sân này gần đây")
        elif habit_score >= 0.8:
            parts.append("Hợp môn và mức giá bạn hay chọn")
        if venue.total_reviews >= PRIOR_REVIEWS and venue.rating >= 4.0:
            parts.append(f"Nhiều người chơi đánh giá tốt ({venue.total_reviews} đánh giá)")
        elif is_nearest:
            parts.append("Gần bạn nhất trong khu vực")
        elif venue.total_reviews == 0:
            parts.append("Sân mới, chưa có đánh giá")
        return " · ".join(parts[:2]) or "Gần bạn và còn hoạt động"

    @staticmethod
    def _user_preference_vector(
        weights: list[tuple[UserInteraction, float]],
        requested_sport: SportType,
    ) -> list[float]:
        sports = list(SportType)
        sport_weights = {sport: 0.0 for sport in sports}
        rating_total = price_total = metadata_weight = 0.0
        for interaction, weight in weights:
            if interaction.sport_type:
                sport_weights[interaction.sport_type] += weight
            rating = interaction.metadata.get("rating")
            price = interaction.metadata.get("price")
            if isinstance(rating, (int, float)) and isinstance(price, (int, float)):
                rating_total += weight * max(0.0, min(5.0, float(rating))) / 5.0
                price_total += weight * max(0.0, min(1.0, float(price) / 500_000.0))
                metadata_weight += weight
        sport_weights[requested_sport] += max(1.0, max(sport_weights.values()))
        peak = max(sport_weights.values()) or 1.0
        rating_preference = rating_total / metadata_weight if metadata_weight else 0.8
        price_preference = price_total / metadata_weight if metadata_weight else 0.5
        return [sport_weights[sport] / peak for sport in sports] + [rating_preference, price_preference]

    @staticmethod
    def _venue_vector(venue: VenueRecommendation) -> list[float]:
        supported = set(venue.sport_types)
        sports = [1.0 if sport.value in supported else 0.0 for sport in SportType]
        average_price = ((venue.min_price or 0) + (venue.max_price or venue.min_price or 0)) / 2
        return sports + [bayesian_rating(venue.rating, venue.total_reviews) / 5.0, min(1.0, average_price / 500_000.0)]
