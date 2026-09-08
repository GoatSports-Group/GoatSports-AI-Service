from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from uuid import UUID

from app.application.dto.response.models import PersonalizedRecommendation
from app.application.port.persistence import AiDocumentRepository
from app.application.service.model_registry import get_or_create_active_model
from app.domain.entity.models import UserInteraction
from app.domain.enums import (
    AiCapability,
    InteractionType,
    RecommendationTargetType,
)

INTERACTION_WEIGHTS = {
    InteractionType.VIEW: 1.0,
    InteractionType.LIKE: 3.0,
    InteractionType.COMMENT: 4.0,
    InteractionType.BOOKING: 6.0,
}


class CollaborativeFilteringService:
    def __init__(self, repository: AiDocumentRepository) -> None:
        self.repository = repository

    def record(self, interaction: UserInteraction) -> UserInteraction:
        return self.repository.save_interaction(interaction)

    def recommend(
        self,
        user_id: UUID,
        target_type: RecommendationTargetType,
        max_results: int = 20,
    ) -> list[PersonalizedRecommendation]:
        get_or_create_active_model(self.repository, AiCapability.FEED_RANKING)
        interactions = self.repository.find_interactions(target_type=target_type)
        matrix = self._build_matrix(interactions)
        current = matrix.get(user_id, {})
        scores: dict[UUID, float] = defaultdict(float)
        similarities: dict[UUID, float] = {}

        for other_id, other_vector in matrix.items():
            if other_id == user_id:
                continue
            similarity = self._sparse_cosine(current, other_vector)
            if similarity <= 0:
                continue
            similarities[other_id] = similarity
            for target_id, value in other_vector.items():
                if target_id not in current:
                    scores[target_id] += similarity * value

        if scores:
            normalizer = sum(similarities.values()) or 1.0
            ranked = [(target_id, value / normalizer) for target_id, value in scores.items()]
            reason = "Được cộng đồng có hành vi tương tự quan tâm."
        else:
            popularity: dict[UUID, float] = defaultdict(float)
            for interaction in interactions:
                if interaction.target_id not in current:
                    popularity[interaction.target_id] += self._effective_weight(interaction)
            ranked = list(popularity.items())
            reason = "Đang được cộng đồng GoatSports quan tâm."

        ranked.sort(key=lambda item: item[1], reverse=True)
        top = ranked[:max_results]
        peak = top[0][1] if top else 1.0
        return [
            PersonalizedRecommendation(
                target_id=target_id,
                target_type=target_type,
                score=round(min(100.0, value / peak * 100.0), 2),
                reason=reason,
            )
            for target_id, value in top
        ]

    def user_interactions(
        self,
        user_id: UUID,
        target_type: RecommendationTargetType | None = None,
    ) -> list[UserInteraction]:
        return self.repository.find_interactions(user_id=user_id, target_type=target_type)

    def _build_matrix(
        self,
        interactions: list[UserInteraction],
    ) -> dict[UUID, dict[UUID, float]]:
        matrix: dict[UUID, dict[UUID, float]] = defaultdict(lambda: defaultdict(float))
        for interaction in interactions:
            matrix[interaction.user_id][interaction.target_id] += self._effective_weight(interaction)
        return {user_id: dict(vector) for user_id, vector in matrix.items()}

    @staticmethod
    def _sparse_cosine(first: dict[UUID, float], second: dict[UUID, float]) -> float:
        if not first or not second:
            return 0.0
        common = first.keys() & second.keys()
        dot = sum(first[item] * second[item] for item in common)
        first_norm = math.sqrt(sum(value * value for value in first.values()))
        second_norm = math.sqrt(sum(value * value for value in second.values()))
        return dot / (first_norm * second_norm) if first_norm and second_norm else 0.0

    @staticmethod
    def _effective_weight(interaction: UserInteraction) -> float:
        base = interaction.weight or INTERACTION_WEIGHTS[interaction.interaction_type]
        occurred_at = interaction.occurred_at
        if occurred_at.tzinfo is None:
            occurred_at = occurred_at.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (datetime.now(timezone.utc) - occurred_at).total_seconds() / 86_400)
        return base * math.exp(-age_days / 90.0)