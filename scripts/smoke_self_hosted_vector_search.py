from __future__ import annotations

import time
from uuid import uuid4

from pymongo import MongoClient

from app.domain.entity.models import EmbeddingRecord
from app.domain.enums import EmbeddingSourceType
from app.infrastructure.config.settings import settings
from app.infrastructure.persistence.repository.mongo_repository import (
    MongoAiDocumentRepository,
)


def main() -> None:
    if not settings.MONGODB_VECTOR_SEARCH_ENABLED:
        raise RuntimeError("MONGODB_VECTOR_SEARCH_ENABLED must be true")

    client = MongoClient(
        settings.MONGODB_URI,
        uuidRepresentation="standard",
        serverSelectionTimeoutMS=5000,
    )
    database = client[settings.MONGODB_DATABASE]
    repository = MongoAiDocumentRepository(database)
    source_id = uuid4()
    record = EmbeddingRecord(
        source_type=EmbeddingSourceType.KNOWLEDGE_CHUNK,
        source_id=source_id,
        model_version_id=uuid4(),
        dimensions=256,
        vector=[1.0] + [0.0] * 255,
    )

    try:
        client.admin.command("ping")
        repository.save_embedding(record)
        for _ in range(30):
            hits = repository.vector_search_embeddings(
                EmbeddingSourceType.KNOWLEDGE_CHUNK,
                record.vector,
                settings.MONGODB_VECTOR_INDEX,
                10,
            )
            if any(candidate.source_id == source_id for candidate, _ in hits):
                print("AI_REPOSITORY_VECTOR_SEARCH_OK")
                return
            time.sleep(1)
        raise RuntimeError("Vector record was not searchable within 30 seconds")
    finally:
        database.embedding_records.delete_one(
            {
                "sourceType": EmbeddingSourceType.KNOWLEDGE_CHUNK.value,
                "sourceId": source_id,
            }
        )
        client.close()


if __name__ == "__main__":
    main()