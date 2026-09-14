import unittest

from pymongo.errors import AutoReconnect, OperationFailure

from app.domain.enums import EmbeddingSourceType
from app.infrastructure.persistence.repository.mongo_repository import MongoAiDocumentRepository


class UnsupportedVectorSearchCollection:
    def __init__(self):
        self.aggregate_calls = 0

    def aggregate(self, _pipeline):
        self.aggregate_calls += 1
        raise OperationFailure(
            "Unrecognized pipeline stage name: '$vectorSearch'",
            code=40324,
        )


class FakeDatabase:
    def __init__(self):
        self.embedding_records = UnsupportedVectorSearchCollection()


class RecoveringVectorSearchCollection:
    def __init__(self):
        self.aggregate_calls = 0

    def aggregate(self, _pipeline):
        self.aggregate_calls += 1
        if self.aggregate_calls == 1:
            raise AutoReconnect("mongot is temporarily unavailable")
        return []


class RecoveringDatabase:
    def __init__(self):
        self.embedding_records = RecoveringVectorSearchCollection()


class MongoAiDocumentRepositoryTests(unittest.TestCase):
    def test_disables_vector_search_after_local_mongodb_reports_it_unsupported(self):
        database = FakeDatabase()
        repository = MongoAiDocumentRepository(database)

        with self.assertLogs(
            "app.infrastructure.persistence.repository.mongo_repository",
            level="WARNING",
        ) as logs:
            first = repository.vector_search_embeddings(
                EmbeddingSourceType.KNOWLEDGE_CHUNK,
                [0.1, 0.2],
                "embedding_vector_index",
                5,
            )
        second = repository.vector_search_embeddings(
            EmbeddingSourceType.KNOWLEDGE_CHUNK,
            [0.1, 0.2],
            "embedding_vector_index",
            5,
        )

        self.assertEqual([], first)
        self.assertEqual([], second)
        self.assertEqual(1, database.embedding_records.aggregate_calls)
        self.assertEqual(1, len(logs.output))
        self.assertNotIn("Traceback", logs.output[0])

    def test_retries_transient_vector_search_failure_after_cooldown(self):
        current_time = [100.0]
        database = RecoveringDatabase()
        repository = MongoAiDocumentRepository(
            database,
            vector_search_retry_seconds=60,
            clock=lambda: current_time[0],
        )

        with self.assertLogs(
            "app.infrastructure.persistence.repository.mongo_repository",
            level="WARNING",
        ):
            first = repository.vector_search_embeddings(
                EmbeddingSourceType.KNOWLEDGE_CHUNK,
                [0.1, 0.2],
                "embedding_vector_index",
                5,
            )
        during_cooldown = repository.vector_search_embeddings(
            EmbeddingSourceType.KNOWLEDGE_CHUNK,
            [0.1, 0.2],
            "embedding_vector_index",
            5,
        )
        current_time[0] = 161.0
        after_cooldown = repository.vector_search_embeddings(
            EmbeddingSourceType.KNOWLEDGE_CHUNK,
            [0.1, 0.2],
            "embedding_vector_index",
            5,
        )

        self.assertEqual([], first)
        self.assertEqual([], during_cooldown)
        self.assertEqual([], after_cooldown)
        self.assertEqual(2, database.embedding_records.aggregate_calls)


if __name__ == "__main__":
    unittest.main()
