from app.application.service.collaborative_filtering import CollaborativeFilteringService
from app.application.service.demand_forecast_service import DemandForecastService
from app.application.service.face_readiness_service import FaceReadinessService
from app.application.service.identity_verification_service import IdentityVerificationService
from app.application.service.match_insight_service import MatchInsightService
from app.application.service.matchmaking_model_service import MatchmakingModelService
from app.application.service.matchmaking_service import MatchmakingService
from app.application.service.rag_chatbot_service import RagChatbotService
from app.application.service.recommendation_service import RecommendationService
from app.application.service.semantic_search_service import SemanticSearchService
from app.application.service.service_config import MatchmakingConfig, RagConfig, VectorSearchConfig
from app.infrastructure.config.database import database_clients
from app.infrastructure.config.settings import settings
from app.infrastructure.external.client import (
    HttpAuthServiceClient,
    HttpClubServiceClient,
    HttpNotificationServiceClient,
    HttpSocialServiceClient,
    HttpVenueServiceClient,
    OpenAiCompatibleLanguageModelClient,
)
from app.infrastructure.external.client.binary_object_client import HttpBinaryObjectClient
from app.infrastructure.forecast.chronos_onnx import load_forecaster
from app.infrastructure.persistence.repository import (
    MongoAiDocumentRepository,
    MongoMatchmakingSessionRepository,
    RedisMatchmakingQueueRepository,
)
from app.infrastructure.persistence.repository.redis_identity_verification_repository import (
    RedisIdentityVerificationRepository,
)
from app.infrastructure.vision import (
    OnnxPassiveLivenessAdapter,
    OpenCvFaceReadinessAdapter,
    OpenCvFaceVerificationAdapter,
    OpenCvImageQualityAdapter,
    ProcessIsolatedPaddleOcrCitizenIdAdapter,
)

document_repository = MongoAiDocumentRepository(
    database_clients.mongo,
    vector_search_retry_seconds=settings.MONGODB_VECTOR_SEARCH_RETRY_SECONDS,
)
session_repository = MongoMatchmakingSessionRepository(database_clients.mongo)
queue_repository = RedisMatchmakingQueueRepository(database_clients.redis)

venue_client = HttpVenueServiceClient(settings)
social_client = HttpSocialServiceClient(settings)
notification_client = HttpNotificationServiceClient(settings)
auth_client = HttpAuthServiceClient(settings)
club_client = HttpClubServiceClient(settings)
language_model_client = OpenAiCompatibleLanguageModelClient(settings)

vector_search_config = VectorSearchConfig(
    enabled=settings.MONGODB_VECTOR_SEARCH_ENABLED,
    index_name=settings.MONGODB_VECTOR_INDEX,
)

collaborative_filtering_service = CollaborativeFilteringService(document_repository)
demand_forecast_service = DemandForecastService(
    venue_client, document_repository, load_forecaster(settings.DEMAND_FORECAST_MODEL_PATH)
)
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
        result_window_seconds=settings.MATCHMAKING_RESULT_WINDOW_SECONDS,
    ),
    auth_client,
    club_client,
)

identity_document_adapter = ProcessIsolatedPaddleOcrCitizenIdAdapter(settings)
face_readiness_adapter = OpenCvFaceReadinessAdapter(settings)
face_verification_adapter = OpenCvFaceVerificationAdapter(settings)
liveness_adapter = OnnxPassiveLivenessAdapter(settings)
face_readiness_service = FaceReadinessService(face_readiness_adapter)

identity_verification_service = IdentityVerificationService(
    HttpBinaryObjectClient(settings),
    OpenCvImageQualityAdapter(settings),
    identity_document_adapter,
    face_verification_adapter,
    liveness_adapter,
    RedisIdentityVerificationRepository(database_clients.redis),
    settings.IDENTITY_VERIFICATION_TTL_SECONDS,
)


def warm_up_identity_verification() -> None:
    if settings.IDENTITY_VISION_WARMUP_ENABLED:
        identity_document_adapter.warm_up()
        face_readiness_adapter.warm_up()
        face_verification_adapter.warm_up()
        liveness_adapter.warm_up()


def shutdown_identity_verification() -> None:
    identity_document_adapter.close()

matchmaking_model_service = MatchmakingModelService(session_repository, document_repository)
match_insight_service = MatchInsightService(
    session_repository, language_model_client, enabled=settings.MATCH_INSIGHT_ENABLED
)
