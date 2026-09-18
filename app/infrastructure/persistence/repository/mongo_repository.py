from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date, datetime, time
from enum import Enum
from time import monotonic
from typing import Any
from uuid import UUID

from pymongo import ASCENDING, DESCENDING
from pymongo.database import Database
from pymongo.errors import OperationFailure, PyMongoError

from app.domain.entity.models import (
    AIModelVersion,
    DemandForecastResult,
    DemandObservation,
    EmbeddingRecord,
    KnowledgeChunk,
    KnowledgeDocument,
    MatchmakingSession,
    SemanticDocument,
    UserInteraction,
)
from app.domain.enums import (
    AiCapability,
    AiModelStatus,
    EmbeddingSourceType,
    MatchSessionStatus,
    RecommendationTargetType,
    SearchDocumentType,
)

logger = logging.getLogger(__name__)


def _to_bson(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (date, time)) and not isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _to_bson(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_bson(item) for item in value]
    return value


def _document(model: Any) -> dict[str, Any]:
    return _to_bson(model.model_dump(by_alias=True, mode="python"))


class MongoMatchmakingSessionRepository:
    def __init__(self, database: Database) -> None:
        self.collection = database.matchmaking_sessions

    def save(self, session: MatchmakingSession) -> MatchmakingSession:
        payload = _document(session)
        self.collection.replace_one(
            {"sessionId": session.session_id}, payload, upsert=True
        )
        return session

    def find_by_id(self, session_id: UUID) -> MatchmakingSession | None:
        payload = self.collection.find_one({"sessionId": session_id}, {"_id": 0})
        return MatchmakingSession.model_validate(payload) if payload else None

    def find_latest_by_participant(self, participant_id: UUID) -> MatchmakingSession | None:
        payload = self.collection.find_one(
            {"participants.participantId": participant_id},
            {"_id": 0},
            sort=[("matchedAt", -1)],
        )
        return MatchmakingSession.model_validate(payload) if payload else None

    def find_active_by_participant(self, participant_id: UUID) -> MatchmakingSession | None:
        payload = self.collection.find_one(
            {
                "participants.participantId": participant_id,
                "status": {
                    "$in": [
                          MatchSessionStatus.PROPOSED.value,
                          MatchSessionStatus.ACCEPTED_BY_ONE.value,
                          MatchSessionStatus.ACCEPTED.value,
                          MatchSessionStatus.VENUE_SELECTED.value,
                          MatchSessionStatus.BOOKING_PENDING.value,
                          MatchSessionStatus.CONFIRMED.value,
                          MatchSessionStatus.CHECKED_IN.value,
                          MatchSessionStatus.RESULT_PENDING.value,
                          MatchSessionStatus.DISPUTED.value,
                    ]
                },
            },
            {"_id": 0},
            sort=[("matchedAt", -1)],
        )
        return MatchmakingSession.model_validate(payload) if payload else None

    def find_by_participant(
        self,
        participant_id: UUID,
        limit: int,
    ) -> list[MatchmakingSession]:
        payloads = self.collection.find(
            {"participants.participantId": participant_id},
            {"_id": 0},
            sort=[("matchedAt", DESCENDING)],
        ).limit(max(1, min(limit, 20)))
        return [MatchmakingSession.model_validate(payload) for payload in payloads]


class MongoAiDocumentRepository:
    """Persistence entry point for the non-matchmaking Mongo documents in the diagram."""

    def __init__(
        self,
        database: Database,
        vector_search_retry_seconds: float = 60.0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.database = database
        self._vector_search_supported: bool | None = None
        self._vector_search_retry_seconds = max(1.0, vector_search_retry_seconds)
        self._vector_search_retry_at = 0.0
        self._clock = clock

    def save_model_version(self, model: AIModelVersion) -> AIModelVersion:
        self.database.ai_model_versions.replace_one(
            {"modelVersionId": model.model_version_id}, _document(model), upsert=True
        )
        return model

    def find_active_model(self, capability: AiCapability) -> AIModelVersion | None:
        payload = self.database.ai_model_versions.find_one(
            {"capability": capability.value, "status": AiModelStatus.ACTIVE.value},
            {"_id": 0},
            sort=[("trainedAt", -1)],
        )
        return AIModelVersion.model_validate(payload) if payload else None

    def save_forecast(self, forecast: DemandForecastResult) -> DemandForecastResult:
        self.database.demand_forecast_results.replace_one(
            {"forecastId": forecast.forecast_id}, _document(forecast), upsert=True
        )
        return forecast

    def save_embedding(self, embedding: EmbeddingRecord) -> EmbeddingRecord:
        self.database.embedding_records.replace_one(
            {"sourceType": embedding.source_type.value, "sourceId": embedding.source_id},
            _document(embedding),
            upsert=True,
        )
        return embedding

    def find_embeddings(
        self,
        source_type: EmbeddingSourceType,
        source_ids: list[UUID],
    ) -> dict[UUID, EmbeddingRecord]:
        if not source_ids:
            return {}
        payloads = self.database.embedding_records.find(
            {"sourceType": source_type.value, "sourceId": {"$in": source_ids}},
            {"_id": 0},
        )
        records = [EmbeddingRecord.model_validate(payload) for payload in payloads]
        return {record.source_id: record for record in records}

    def vector_search_embeddings(
        self,
        source_type: EmbeddingSourceType,
        query_vector: list[float],
        index_name: str,
        limit: int,
    ) -> list[tuple[EmbeddingRecord, float]]:
        if self._vector_search_supported is False:
            return []
        if self._clock() < self._vector_search_retry_at:
            return []

        pipeline = [
            {
                "$vectorSearch": {
                    "index": index_name,
                    "path": "vector",
                    "queryVector": query_vector,
                    "numCandidates": max(100, limit * 10),
                    "limit": limit,
                    "filter": {"sourceType": source_type.value},
                }
            },
            {"$project": {"_id": 0, "document": "$$ROOT", "score": {"$meta": "vectorSearchScore"}}},
        ]
        try:
            payloads = self.database.embedding_records.aggregate(pipeline)
            results = [
                (EmbeddingRecord.model_validate(payload["document"]), float(payload["score"]))
                for payload in payloads
            ]
            self._vector_search_supported = True
            self._vector_search_retry_at = 0.0
            return results
        except OperationFailure as error:
            if error.code == 40324 or (
                "$vectorSearch" in str(error)
                and "Unrecognized pipeline stage" in str(error)
            ):
                self._vector_search_supported = False
                logger.warning(
                    "MongoDB does not support $vectorSearch; using local ranking for this process"
                )
                return []
            logger.warning(
                "MongoDB vector search failed; falling back to local ranking",
                exc_info=True,
            )
            self._vector_search_retry_at = self._clock() + self._vector_search_retry_seconds
            return []
        except PyMongoError:
            logger.warning(
                "MongoDB vector search failed; falling back to local ranking",
                exc_info=True,
            )
            self._vector_search_retry_at = self._clock() + self._vector_search_retry_seconds
            return []

    def save_semantic_document(self, document: SemanticDocument) -> SemanticDocument:
        self.database.semantic_documents.replace_one(
            {
                "documentType": document.document_type.value,
                "sourceId": document.source_id,
            },
            _document(document),
            upsert=True,
        )
        return document

    def find_semantic_document(
        self,
        document_type: SearchDocumentType,
        source_id: UUID,
    ) -> SemanticDocument | None:
        payload = self.database.semantic_documents.find_one(
            {"documentType": document_type.value, "sourceId": source_id},
            {"_id": 0},
        )
        return SemanticDocument.model_validate(payload) if payload else None

    def find_semantic_documents(
        self,
        document_type: SearchDocumentType | None = None,
        limit: int = 1000,
    ) -> list[SemanticDocument]:
        query: dict[str, Any] = {"active": True}
        if document_type:
            query["documentType"] = document_type.value
        payloads = self.database.semantic_documents.find(query, {"_id": 0}).limit(limit)
        return [SemanticDocument.model_validate(payload) for payload in payloads]

    def find_semantic_documents_by_ids(
        self,
        document_ids: list[UUID],
    ) -> list[SemanticDocument]:
        payloads = self.database.semantic_documents.find(
            {"documentId": {"$in": document_ids}, "active": True},
            {"_id": 0},
        )
        return [SemanticDocument.model_validate(payload) for payload in payloads]

    def save_knowledge_document(self, document: KnowledgeDocument) -> KnowledgeDocument:
        self.database.knowledge_documents.replace_one(
            {"documentId": document.document_id}, _document(document), upsert=True
        )
        return document

    def find_knowledge_documents(self) -> dict[UUID, KnowledgeDocument]:
        payloads = self.database.knowledge_documents.find({"active": True}, {"_id": 0})
        documents = [KnowledgeDocument.model_validate(payload) for payload in payloads]
        return {document.document_id: document for document in documents}

    def save_knowledge_chunk(self, chunk: KnowledgeChunk) -> KnowledgeChunk:
        self.database.knowledge_chunks.replace_one(
            {"chunkId": chunk.chunk_id}, _document(chunk), upsert=True
        )
        return chunk

    def replace_knowledge_chunks(
        self,
        document_id: UUID,
        chunks: list[KnowledgeChunk],
    ) -> None:
        self.database.knowledge_chunks.delete_many({"documentId": document_id})
        if chunks:
            self.database.knowledge_chunks.insert_many([_document(chunk) for chunk in chunks])

    def find_knowledge_chunks(self, limit: int = 2000) -> list[KnowledgeChunk]:
        payloads = self.database.knowledge_chunks.find({}, {"_id": 0}).limit(limit)
        return [KnowledgeChunk.model_validate(payload) for payload in payloads]

    def find_knowledge_chunks_by_ids(self, chunk_ids: list[UUID]) -> list[KnowledgeChunk]:
        payloads = self.database.knowledge_chunks.find(
            {"chunkId": {"$in": chunk_ids}},
            {"_id": 0},
        )
        return [KnowledgeChunk.model_validate(payload) for payload in payloads]

    def save_demand_observation(self, observation: DemandObservation) -> DemandObservation:
        self.database.demand_observations.replace_one(
            {"observationId": observation.observation_id},
            _document(observation),
            upsert=True,
        )
        return observation

    def find_demand_observations(
        self,
        venue_id: UUID,
        sport_type: str,
        limit: int = 10_000,
    ) -> list[DemandObservation]:
        payloads = self.database.demand_observations.find(
            {"venueId": venue_id, "sportType": sport_type},
            {"_id": 0},
            sort=[("bucketStart", ASCENDING)],
        ).limit(limit)
        return [DemandObservation.model_validate(payload) for payload in payloads]

    def find_forecasts(
        self,
        venue_id: UUID,
        sport_type: str,
        limit: int = 500,
    ) -> list[DemandForecastResult]:
        payloads = self.database.demand_forecast_results.find(
            {"venueId": venue_id, "sportType": sport_type},
            {"_id": 0},
            sort=[("bucketStart", ASCENDING)],
        ).limit(limit)
        return [DemandForecastResult.model_validate(payload) for payload in payloads]

    def save_interaction(self, interaction: UserInteraction) -> UserInteraction:
        self.database.user_interactions.replace_one(
            {"interactionId": interaction.interaction_id},
            _document(interaction),
            upsert=True,
        )
        return interaction

    def find_interactions(
        self,
        user_id: UUID | None = None,
        target_type: RecommendationTargetType | None = None,
        limit: int = 20_000,
    ) -> list[UserInteraction]:
        query: dict[str, Any] = {}
        if user_id:
            query["userId"] = user_id
        if target_type:
            query["targetType"] = target_type.value
        payloads = self.database.user_interactions.find(
            query,
            {"_id": 0},
            sort=[("occurredAt", DESCENDING)],
        ).limit(limit)
        return [UserInteraction.model_validate(payload) for payload in payloads]


# Compatibility names for callers migrating from the original flat package.
MatchmakingSessionRepository = MongoMatchmakingSessionRepository
AiDocumentRepository = MongoAiDocumentRepository
