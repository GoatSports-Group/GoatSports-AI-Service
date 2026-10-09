from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid5

from app.application.port.persistence import AiDocumentRepository
from app.domain.entity.models import AIModelVersion
from app.domain.enums import AiCapability, AiModelStatus

DEFAULT_MODELS = {
    AiCapability.MATCHMAKING: "sport-elo-profile-matchmaking-v2",
    AiCapability.VENUE_RECOMMENDATION: "cosine-geo-venue-v1",
    AiCapability.SEMANTIC_SEARCH: "hybrid-tfidf-ngram-embedding-v1",
    AiCapability.FEED_RANKING: "user-user-collaborative-filtering-v1",
    AiCapability.RAG: "hybrid-retrieval-rag-v1",
}


def get_or_create_active_model(
    repository: AiDocumentRepository,
    capability: AiCapability,
) -> AIModelVersion:
    current = repository.find_active_model(capability)
    if current:
        return current
    name = DEFAULT_MODELS[capability]
    now = datetime.now(timezone.utc)
    model = AIModelVersion(
        model_version_id=uuid5(NAMESPACE_URL, f"goatsports:{capability.value}:{name}"),
        model_name=name,
        version="1.0.0",
        capability=capability,
        trained_at=now,
        training_cutoff=now,
        metrics={},
        artifact_uri="builtin://ai-service",
        status=AiModelStatus.ACTIVE,
    )
    return repository.save_model_version(model)
