"""Run against disposable MongoDB and Redis instances, never production data."""

from uuid import uuid4

from app.application.dto.request.models import (
    ChatbotQuery,
    KnowledgeDocumentRequest,
    SemanticDocumentRequest,
)
from app.application.service.collaborative_filtering import CollaborativeFilteringService
from app.application.service.rag_chatbot_service import RagChatbotService
from app.application.service.recommendation_service import RecommendationService
from app.application.service.semantic_search_service import SemanticSearchService
from app.application.service.service_config import RagConfig, VectorSearchConfig
from app.application.service.vector_search import HashingEmbedder
from app.domain.entity.models import UserInteraction
from app.domain.enums import InteractionType, RecommendationTargetType
from app.infrastructure.config.database import database_clients
from app.infrastructure.persistence.repository import MongoAiDocumentRepository


class EmptyVenueClient:
    def recommend(self, *_, **__):
        return [], None


class ExtractiveLanguageModelClient:
    model_name = None

    def complete(self, system: str, prompt: str, max_tokens: int | None = None) -> None:
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
        HashingEmbedder(),
        RagConfig(max_context_chunks=5),
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

    database_clients.close()
    print("AI_ALGORITHM_INTEGRATION_SMOKE_OK")


if __name__ == "__main__":
    run()
