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


@dataclass(frozen=True)
class FaceReadinessResult:
    ready: bool
    reason_code: str
    message: str
    face_count: int
    brightness: float = 0.0
    centered: bool = False
    frontal: bool = False
    glasses_free: bool = False
    eyes_open: bool = False
    mask_free: bool = False
    face_box: tuple[float, float, float, float] | None = None
    yaw_ratio: float = 0.0
    pitch_ratio: float = 0.0
    roll_degrees: float = 0.0


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


class FaceReadinessPort(Protocol):
    def analyze(self, frame: bytes) -> FaceReadinessResult: ...


class IdentityVerificationRepository(Protocol):
    def save(self, verification: IdentityVerification, ttl_seconds: int) -> IdentityVerification: ...
    def find_by_id(self, verification_id: UUID) -> IdentityVerification | None: ...
