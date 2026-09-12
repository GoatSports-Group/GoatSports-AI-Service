from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from app.domain.entity.models import IdentityDocumentData, IdentityVerification


@dataclass(frozen=True)
class ImageQualityResult:
    score: float
    acceptable: bool
    reason_code: str | None = None


@dataclass(frozen=True)
class DocumentRecognitionResult:
    data: IdentityDocumentData
    confidence: float
    expected_side: bool
    raw_lines: tuple[str, ...]


@dataclass(frozen=True)
class FaceComparisonResult:
    similarity: float
    matched: bool


@dataclass(frozen=True)
class LivenessResult:
    score: float
    live: bool
    representative_frame: bytes | None


class BinaryObjectClient(Protocol):
    def download(self, url: str) -> bytes: ...


class ImageQualityPort(Protocol):
    def assess(self, image: bytes) -> ImageQualityResult: ...


class IdentityDocumentRecognitionPort(Protocol):
    def recognize_front(self, image: bytes) -> DocumentRecognitionResult: ...


class FaceVerificationPort(Protocol):
    def contains_face(self, image: bytes) -> bool: ...

    def compare(self, identity_image: bytes, live_image: bytes) -> FaceComparisonResult: ...


class LivenessPort(Protocol):
    def verify(self, frames: list[bytes]) -> LivenessResult: ...


class IdentityVerificationRepository(Protocol):
    def save(self, verification: IdentityVerification, ttl_seconds: int) -> IdentityVerification: ...
    def find_by_id(self, verification_id: UUID) -> IdentityVerification | None: ...
