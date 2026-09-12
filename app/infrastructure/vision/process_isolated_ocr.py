from __future__ import annotations

from concurrent.futures import Executor, ProcessPoolExecutor
from multiprocessing import get_context
from typing import Any

from app.application.port.identity_verification import DocumentRecognitionResult
from app.domain.entity.models import IdentityDocumentData
from app.infrastructure.config.settings import Settings
from app.infrastructure.vision.open_source_identity import PaddleOcrCitizenIdAdapter

_worker_adapter: PaddleOcrCitizenIdAdapter | None = None


def _get_worker_adapter(settings: Settings) -> PaddleOcrCitizenIdAdapter:
    global _worker_adapter
    if _worker_adapter is None:
        _worker_adapter = PaddleOcrCitizenIdAdapter(settings)
    return _worker_adapter


def _warm_up_worker(settings: Settings) -> None:
    _get_worker_adapter(settings).warm_up()


def _recognize_front_worker(
    settings: Settings,
    image: bytes,
) -> tuple[dict[str, Any], float, bool, tuple[str, ...]]:
    result = _get_worker_adapter(settings).recognize_front(image)
    return (
        result.data.model_dump(mode="json"),
        result.confidence,
        result.expected_side,
        result.raw_lines,
    )


class ProcessIsolatedPaddleOcrCitizenIdAdapter:
    """Keep Paddle inference outside the process serving latency-sensitive APIs."""

    def __init__(
        self,
        settings: Settings,
        executor: Executor | None = None,
    ) -> None:
        self.settings = settings
        self._executor = executor or ProcessPoolExecutor(
            max_workers=settings.IDENTITY_OCR_WORKER_PROCESSES,
            mp_context=get_context("spawn"),
        )

    def warm_up(self) -> None:
        self._executor.submit(_warm_up_worker, self.settings).result()

    def recognize_front(self, image: bytes) -> DocumentRecognitionResult:
        document, confidence, expected_side, raw_lines = self._executor.submit(
            _recognize_front_worker,
            self.settings,
            image,
        ).result()
        return DocumentRecognitionResult(
            data=IdentityDocumentData.model_validate(document),
            confidence=confidence,
            expected_side=expected_side,
            raw_lines=raw_lines,
        )

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
