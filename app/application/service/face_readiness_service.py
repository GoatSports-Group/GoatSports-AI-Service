from __future__ import annotations

import base64
import binascii

from app.application.dto.request.models import FaceReadinessRequest
from app.application.port.identity_verification import FaceReadinessPort, FaceReadinessResult


class FaceReadinessService:
    def __init__(self, readiness_port: FaceReadinessPort) -> None:
        self.readiness_port = readiness_port

    def analyze(self, request: FaceReadinessRequest) -> FaceReadinessResult:
        return self.readiness_port.analyze(self._decode_frame(request.frame))

    @staticmethod
    def _decode_frame(value: str) -> bytes:
        try:
            _, encoded = value.split(",", 1)
            content = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exception:
            raise ValueError("Khung hình camera không hợp lệ.") from exception
        if not content or len(content) > 800_000:
            raise ValueError("Khung hình camera không hợp lệ hoặc vượt quá 800 KB.")
        return content
