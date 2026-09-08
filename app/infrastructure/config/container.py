from app.application.service.collaborative_filtering import CollaborativeFilteringService
from app.application.service.demand_forecast_service import DemandForecastService
from app.application.service.matchmaking_service import MatchmakingService
from app.application.service.rag_chatbot_service import RagChatbotService
from app.application.service.recommendation_service import RecommendationService
from app.application.service.semantic_search_service import SemanticSearchService
from app.application.service.service_config import MatchmakingConfig, RagConfig, VectorSearchConfig
from app.infrastructure.config.database import database_clients
from app.infrastructure.config.settings import settings
from app.infrastructure.external.client import (
    HttpNotificationServiceClient,
    HttpSocialServiceClient,
    HttpVenueServiceClient,
    OpenAiCompatibleLanguageModelClient,
)
from app.infrastructure.persistence.repository import (
    MongoAiDocumentRepository,
    MongoMatchmakingSessionRepository,
    RedisMatchmakingQueueRepository,
)

document_repository = MongoAiDocumentRepository(database_clients.mongo)
session_repository = MongoMatchmakingSessionRepository(database_clients.mongo)
queue_repository = RedisMatchmakingQueueRepository(database_clients.redis)

venue_client = HttpVenueServiceClient(settings)
social_client = HttpSocialServiceClient(settings)
notification_client = HttpNotificationServiceClient(settings)
language_model_client = OpenAiCompatibleLanguageModelClient(settings)

vector_search_config = VectorSearchConfig(
    enabled=settings.MONGODB_VECTOR_SEARCH_ENABLED,
    index_name=settings.MONGODB_VECTOR_INDEX,
)

collaborative_filtering_service = CollaborativeFilteringService(document_repository)
demand_forecast_service = DemandForecastService(document_repository)
recommendation_service = RecommendationService(venue_client, document_repository)
semantic_search_service = SemanticSearchService(document_repository, vector_search_config)
rag_chatbot_service = RagChatbotService(
    document_repository,
    recommendation_service,
    language_model_client,
    RagConfig(
        vector_search=vector_search_config,
        max_context_chunks=settings.RAG_MAX_CONTEXT_CHUNKS,
        minimum_relevance_score=settings.RAG_MIN_RELEVANCE_SCORE,
    ),
)
matchmaking_service = MatchmakingService(
    queue_repository,
    session_repository,
    document_repository,
    venue_client,
    social_client,
    notification_client,
    MatchmakingConfig(
        queue_ttl_seconds=settings.MATCHMAKING_QUEUE_TTL_SECONDS,
        session_ttl_seconds=settings.MATCHMAKING_SESSION_TTL_SECONDS,
        minimum_score=settings.MATCHMAKING_MIN_SCORE,
        scan_limit=settings.MATCHMAKING_SCAN_LIMIT,
    ),
)
