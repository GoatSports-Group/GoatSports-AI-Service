import unittest
from datetime import datetime, timezone

from app.application.dto.response.models import AiCapabilityStatus
from app.application.service.ai_status_service import AiStatusService
from app.domain.entity.models import AIModelVersion
from app.domain.enums import AiCapability, AiModelStatus


def _model(capability: AiCapability, version: str, day: int) -> AIModelVersion:
    return AIModelVersion(
        model_name=f"{capability.value.lower()}-model",
        version=version,
        capability=capability,
        trained_at=datetime(2026, 10, day, tzinfo=timezone.utc),
        status=AiModelStatus.ACTIVE,
    )


class FakeDocuments:
    def __init__(self, models: list[AIModelVersion]):
        self.models = models
        self.calls = 0

    def find_active_models(self) -> list[AIModelVersion]:
        self.calls += 1
        return self.models


def _capability(capability: AiCapability) -> AiCapabilityStatus:
    return AiCapabilityStatus(capability=capability, algorithm="a", storage="s", status="READY", detail="d")


def _fail() -> None:
    raise ConnectionError("refused")


class AiStatusServiceTest(unittest.TestCase):
    def test_attaches_latest_active_model_per_capability_from_one_query(self):
        documents = FakeDocuments([
            _model(AiCapability.DEMAND_FORECAST, "v2", 9),
            _model(AiCapability.DEMAND_FORECAST, "v1", 1),
            _model(AiCapability.MATCHMAKING, "w1", 5),
        ])
        service = AiStatusService(documents, [("MongoDB", lambda: None), ("Redis", lambda: True)])

        status = service.status([
            _capability(AiCapability.DEMAND_FORECAST),
            _capability(AiCapability.MATCHMAKING),
            _capability(AiCapability.RAG),
        ])

        self.assertEqual(documents.calls, 1)
        by_capability = {item.capability: item for item in status.capabilities}
        self.assertEqual(by_capability[AiCapability.DEMAND_FORECAST].active_model.version, "v2")
        self.assertEqual(by_capability[AiCapability.MATCHMAKING].active_model.version, "w1")
        self.assertIsNone(by_capability[AiCapability.RAG].active_model)
        self.assertTrue(all(item.status == "UP" and item.latency_ms is not None for item in status.dependencies))
        # Trả về camelCase như mọi API khác của ai-service.
        payload = status.model_dump(by_alias=True)
        self.assertIn("activeModel", payload["capabilities"][0])
        self.assertIn("latencyMs", payload["dependencies"][0])

    def test_unreachable_mongo_is_reported_without_querying_it(self):
        documents = FakeDocuments([_model(AiCapability.MATCHMAKING, "w1", 5)])
        service = AiStatusService(documents, [("MongoDB", _fail), ("Redis", lambda: True)])

        status = service.status([_capability(AiCapability.MATCHMAKING)])

        self.assertEqual(documents.calls, 0)
        mongo = status.dependencies[0]
        self.assertEqual((mongo.status, mongo.detail), ("DOWN", "ConnectionError"))
        self.assertEqual(status.dependencies[1].status, "UP")
        self.assertIsNone(status.capabilities[0].active_model)


if __name__ == "__main__":
    unittest.main()
