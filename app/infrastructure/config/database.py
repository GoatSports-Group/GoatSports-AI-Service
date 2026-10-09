from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.database import Database
from redis import Redis

from app.infrastructure.config.settings import settings


class DatabaseClients:
    def __init__(self) -> None:
        self.mongo_client = MongoClient(
            settings.MONGODB_URI,
            uuidRepresentation="standard",
            tz_aware=True,
            serverSelectionTimeoutMS=5000,
        )
        self.mongo: Database = self.mongo_client[settings.MONGODB_DATABASE]
        self.redis = Redis.from_url(settings.REDIS_URL, decode_responses=True)

    def connect(self) -> None:
        self.mongo_client.admin.command("ping")
        self.redis.ping()
        self._ensure_indexes()

    def close(self) -> None:
        self.mongo_client.close()
        self.redis.close()

    def _ensure_indexes(self) -> None:
        self.mongo.matchmaking_sessions.create_index(
            [("sessionId", ASCENDING)], unique=True, name="uk_matchmaking_session_id"
        )
        self.mongo.matchmaking_sessions.create_index(
            [("participants.participantId", ASCENDING), ("matchedAt", DESCENDING)],
            name="idx_matchmaking_participant_matched_at",
        )
        # `$or` của truy vấn theo người chơi cần mỗi nhánh có index riêng.
        self.mongo.matchmaking_sessions.create_index(
            [("participants.members.userId", ASCENDING), ("matchedAt", DESCENDING)],
            name="idx_matchmaking_member_matched_at",
        )
        self.mongo.matchmaking_sessions.create_index(
            [("participants.managerIds", ASCENDING), ("matchedAt", DESCENDING)],
            name="idx_matchmaking_manager_matched_at",
        )
        self.mongo.matchmaking_sessions.create_index(
            [("status", ASCENDING), ("expiresAt", ASCENDING)],
            name="idx_matchmaking_status_expires_at",
        )
        self.mongo.ai_model_versions.create_index(
            [("modelVersionId", ASCENDING)], unique=True, name="uk_ai_model_version_id"
        )
        self.mongo.ai_model_versions.create_index(
            [("capability", ASCENDING), ("status", ASCENDING), ("trainedAt", DESCENDING)],
            name="idx_model_capability_status",
        )
        self.mongo.venue_demand_forecasts.create_index(
            [("venueId", ASCENDING)], unique=True, name="uk_venue_demand_forecast_venue"
        )
        self.mongo.embedding_records.create_index(
            [("embeddingId", ASCENDING)], unique=True, name="uk_embedding_id"
        )
        self.mongo.embedding_records.create_index(
            [("sourceType", ASCENDING), ("sourceId", ASCENDING)],
            name="idx_embedding_source",
        )
        self.mongo.semantic_documents.create_index(
            [("documentId", ASCENDING)], unique=True, name="uk_semantic_document_id"
        )
        self.mongo.semantic_documents.create_index(
            [("documentType", ASCENDING), ("sourceId", ASCENDING)],
            unique=True,
            name="uk_semantic_document_source",
        )
        self.mongo.knowledge_documents.create_index(
            [("documentId", ASCENDING)], unique=True, name="uk_knowledge_document_id"
        )
        self.mongo.knowledge_chunks.create_index(
            [("chunkId", ASCENDING)], unique=True, name="uk_knowledge_chunk_id"
        )
        self.mongo.knowledge_chunks.create_index(
            [("documentId", ASCENDING), ("chunkIndex", ASCENDING)],
            unique=True,
            name="uk_knowledge_chunk_position",
        )
        self.mongo.user_interactions.create_index(
            [("interactionId", ASCENDING)], unique=True, name="uk_user_interaction_id"
        )
        self.mongo.user_interactions.create_index(
            [("userId", ASCENDING), ("targetType", ASCENDING), ("occurredAt", DESCENDING)],
            name="idx_interaction_user_target_time",
        )
        self.mongo.user_interactions.create_index(
            [("targetType", ASCENDING), ("targetId", ASCENDING), ("occurredAt", DESCENDING)],
            name="idx_interaction_target_time",
        )


database_clients = DatabaseClients()
