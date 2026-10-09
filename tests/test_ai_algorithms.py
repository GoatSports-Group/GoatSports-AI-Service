import unittest
from uuid import uuid4

from app.application.service.collaborative_filtering import CollaborativeFilteringService
from app.application.service.vector_search import hybrid_text_scores
from app.domain.entity.models import UserInteraction
from app.domain.enums import InteractionType, RecommendationTargetType


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
