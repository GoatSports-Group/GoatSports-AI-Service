import time
from collections.abc import Callable

from app.application.dto.response.models import (
    AiAdminStatus,
    AiCapabilityOverview,
    AiCapabilityStatus,
    AiDependencyStatus,
)
from app.application.port.persistence.repositories import AiDocumentRepository
from app.domain.base import utc_now

DependencyProbe = tuple[str, Callable[[], object]]


class AiStatusService:
    """Trang Giám sát AI của admin: năng lực AI, model đang phục vụ và kho dữ liệu AI phụ thuộc."""

    def __init__(self, documents: AiDocumentRepository, probes: list[DependencyProbe]) -> None:
        self._documents = documents
        self._probes = probes

    def status(self, capabilities: list[AiCapabilityStatus]) -> AiAdminStatus:
        dependencies = [self._probe(name, check) for name, check in self._probes]
        latest: dict = {}
        if all(item.status == "UP" for item in dependencies if item.name == "MongoDB"):
            # Đã sắp mới nhất trước: bản đầu tiên của mỗi năng lực là model đang phục vụ.
            for model in self._documents.find_active_models():
                latest.setdefault(model.capability, model)
        return AiAdminStatus(
            checked_at=utc_now(),
            dependencies=dependencies,
            capabilities=[
                AiCapabilityOverview(**item.model_dump(), active_model=latest.get(item.capability))
                for item in capabilities
            ],
        )

    @staticmethod
    def _probe(name: str, check: Callable[[], object]) -> AiDependencyStatus:
        started = time.perf_counter()
        try:
            check()
        except Exception as exception:  # noqa: BLE001 - trạng thái kho dữ liệu, không để trang lỗi theo
            return AiDependencyStatus(name=name, status="DOWN", detail=type(exception).__name__)
        return AiDependencyStatus(
            name=name, status="UP", latency_ms=round((time.perf_counter() - started) * 1000, 1)
        )
