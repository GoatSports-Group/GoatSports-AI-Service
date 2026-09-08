"""Run against disposable MongoDB and Redis instances, never production data."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from app.application.dto.request.models import (
    ChatbotQuery,
    DemandForecastRequest,
    KnowledgeDocumentRequest,
    SemanticDocumentRequest,
)
from app.application.service.collaborative_filtering import CollaborativeFilteringService
from app.application.service.demand_forecast_service import DemandForecastService
from app.application.service.rag_chatbot_service import RagChatbotService
from app.application.service.recommendation_service import RecommendationService
from app.application.service.semantic_search_service import SemanticSearchService
from app.application.service.service_config import RagConfig, VectorSearchConfig
from app.domain.entity.models import DemandObservation, UserInteraction
from app.domain.enums import InteractionType, RecommendationTargetType, SportType
from app.infrastructure.config.database import database_clients
from app.infrastructure.persistence.repository import MongoAiDocumentRepository


class EmptyVenueClient:
    def recommend(self, *_, **__):
        return [], None


class ExtractiveLanguageModelClient:
    model_name = None

    def generate(self, question: str, context: str) -> None:
        return None


def run() -> None:
    database_clients.connect()
    repository = MongoAiDocumentRepository(database_clients.mongo)
    vector_config = VectorSearchConfig(enabled=False, index_name="embedding_vector_index")

    search = SemanticSearchService(repository, vector_config)
    football_venue_id = uuid4()
    search.index(
        SemanticDocumentRequest(
            document_type="VENUE",
            source_id=football_venue_id,
            content="Sân bóng đá mini lân cận, mặt cỏ nhân tạo và mở cửa buổi tối.",
        )
    )
    search.index(
        SemanticDocumentRequest(
            document_type="POST",
            source_id=uuid4(),
            content="Kỹ thuật giao cầu lông cho người mới.",
        )
    )
    semantic_results = search.search("chỗ đá banh gần đây", max_results=5)
    assert semantic_results and semantic_results[0].source_id == football_venue_id

    rag = RagChatbotService(
        repository,
        RecommendationService(EmptyVenueClient(), repository),
        ExtractiveLanguageModelClient(),
        RagConfig(
            vector_search=vector_config,
            max_context_chunks=5,
            minimum_relevance_score=0.08,
        ),
    )
    rag.ingest(
        KnowledgeDocumentRequest(
            title="Luật cầu lông cơ bản",
            content=(
                "Một trận cầu lông đánh đơn thi đấu theo thể thức ba hiệp thắng hai. "
                "Mỗi hiệp thông thường kết thúc ở 21 điểm và phải hơn đối thủ hai điểm."
            ),
            sport_type="BADMINTON",
            source_url="https://goatsports.local/knowledge/badminton-basic",
        )
    )
    answer = rag.answer_query(
        ChatbotQuery(message="Một hiệp cầu lông bao nhiêu điểm?", favorite_sport="BADMINTON")
    )
    assert answer.grounded and answer.sources and "21" in answer.answer

    current_user, similar_user = uuid4(), uuid4()
    shared_post, suggested_post = uuid4(), uuid4()
    filtering = CollaborativeFilteringService(repository)
    for user_id, post_id, action in (
        (current_user, shared_post, InteractionType.LIKE),
        (similar_user, shared_post, InteractionType.LIKE),
        (similar_user, suggested_post, InteractionType.COMMENT),
    ):
        filtering.record(
            UserInteraction(
                user_id=user_id,
                target_id=post_id,
                target_type=RecommendationTargetType.POST,
                interaction_type=action,
            )
        )
    feed = filtering.recommend(current_user, RecommendationTargetType.POST)
    assert feed and feed[0].target_id == suggested_post

    forecast_service = DemandForecastService(repository)
    venue_id = uuid4()
    first_bucket = datetime(2026, 8, 3, 18, tzinfo=timezone.utc)
    for week, occupancy in enumerate((0.6, 0.7, 0.8, 0.9)):
        start = first_bucket + timedelta(weeks=week)
        forecast_service.record_observation(
            DemandObservation(
                venue_id=venue_id,
                bucket_start=start,
                bucket_end=start + timedelta(hours=1),
                sport_type=SportType.BADMINTON,
                available_slot_count=10,
                booked_slot_count=round(occupancy * 10),
                cancelled_booking_count=0,
                occupancy_rate=occupancy,
                booking_count=round(occupancy * 10),
            )
        )
    target = first_bucket + timedelta(weeks=5)
    forecasts = forecast_service.forecast(
        DemandForecastRequest(
            venue_id=venue_id,
            sport_type=SportType.BADMINTON,
            forecast_start=target,
            forecast_end=target + timedelta(hours=1),
        )
    )
    assert forecasts and forecasts[0].predicted_demand > 0.7

    database_clients.close()
    print("AI_ALGORITHM_INTEGRATION_SMOKE_OK")


if __name__ == "__main__":
    run()
