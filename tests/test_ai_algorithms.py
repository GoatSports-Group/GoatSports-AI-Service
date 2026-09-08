import unittest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from app.application.service.collaborative_filtering import CollaborativeFilteringService
from app.application.service.demand_forecast_service import DemandForecastService
from app.application.service.vector_search import hybrid_text_scores
from app.domain.entity.models import DemandObservation, UserInteraction
from app.domain.enums import InteractionType, RecommendationTargetType, SportType


class FakeInteractionRepository:
    def __init__(self, interactions):
        self.interactions = interactions
        self.model = None

    def find_active_model(self, _):
        return self.model

    def save_model_version(self, model):
        self.model = model
        return model

    def save_interaction(self, interaction):
        self.interactions.append(interaction)
        return interaction

    def find_interactions(self, user_id=None, target_type=None, limit=20_000):
        return [
            interaction
            for interaction in self.interactions
            if (user_id is None or interaction.user_id == user_id)
            and (target_type is None or interaction.target_type == target_type)
        ][:limit]


class AiAlgorithmTests(unittest.TestCase):
    def test_hybrid_search_understands_normalized_vietnamese_synonyms(self):
        scores = hybrid_text_scores(
            "chỗ đá banh gần đây",
            [
                "Sân bóng đá mini ở khu vực lân cận",
                "Giải đấu quần vợt dành cho người mới",
            ],
        )

        self.assertGreater(scores[0][0], scores[1][0])
        self.assertGreater(scores[0][0], 0.1)

    def test_collaborative_filtering_recommends_unseen_post_from_similar_user(self):
        current_user, similar_user = uuid4(), uuid4()
        shared_post, recommended_post = uuid4(), uuid4()
        interactions = [
            self.interaction(current_user, shared_post, InteractionType.LIKE),
            self.interaction(similar_user, shared_post, InteractionType.LIKE),
            self.interaction(similar_user, recommended_post, InteractionType.COMMENT),
        ]
        service = CollaborativeFilteringService(FakeInteractionRepository(interactions))

        recommendations = service.recommend(
            current_user,
            RecommendationTargetType.POST,
            max_results=10,
        )

        self.assertEqual(recommended_post, recommendations[0].target_id)
        self.assertNotIn(shared_post, [item.target_id for item in recommendations])

    def test_demand_forecast_prefers_same_weekday_and_hour(self):
        venue_id = uuid4()
        monday = datetime(2026, 8, 3, 18, tzinfo=timezone.utc)
        observations = [
            DemandObservation(
                venue_id=venue_id,
                bucket_start=monday + timedelta(weeks=index),
                bucket_end=monday + timedelta(weeks=index, hours=1),
                sport_type=SportType.BADMINTON,
                available_slot_count=10,
                booked_slot_count=booked,
                cancelled_booking_count=0,
                occupancy_rate=booked / 10,
                booking_count=booked,
            )
            for index, booked in enumerate([6, 7, 8, 9])
        ]

        predicted, lower, upper = DemandForecastService._predict_bucket(
            monday + timedelta(weeks=5),
            observations,
        )

        self.assertGreater(predicted, 0.7)
        self.assertLessEqual(lower, predicted)
        self.assertGreaterEqual(upper, predicted)

    @staticmethod
    def interaction(user_id, target_id, interaction_type):
        return UserInteraction(
            user_id=user_id,
            target_id=target_id,
            target_type=RecommendationTargetType.POST,
            interaction_type=interaction_type,
        )


if __name__ == "__main__":
    unittest.main()
