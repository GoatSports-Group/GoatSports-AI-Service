import unittest
from datetime import date
from uuid import uuid4

from app.application.dto.request.models import IdentityVerificationRequest
from app.application.port.identity_verification import (
    DocumentRecognitionResult,
    FaceComparisonResult,
    ImageQualityResult,
    LivenessResult,
)
from app.application.service.identity_verification_service import IdentityVerificationService
from app.domain.entity.models import IdentityDocumentData
from app.domain.enums import IdentityDocumentType, IdentityVerificationStatus


class FakeObjectClient:
    def download(self, url: str) -> bytes:
        return url.encode()


class FakeQualityPort:
    def __init__(self, acceptable: bool = True) -> None:
        self.acceptable = acceptable

    def assess(self, _image: bytes) -> ImageQualityResult:
        return ImageQualityResult(0.92, self.acceptable, None if self.acceptable else "IMAGE_BLURRY")


class FakeDocumentPort:
    def recognize_front(self, _image: bytes) -> DocumentRecognitionResult:
        return DocumentRecognitionResult(
            data=IdentityDocumentData(
                document_type=IdentityDocumentType.CITIZEN_ID_CHIP,
                identity_number="079203000001",
                full_name="NGUYEN VAN A",
                date_of_birth=date(2003, 1, 1),
            ),
            confidence=0.94,
            expected_side=True,
            raw_lines=(),
        )

class FakeLivenessPort:
    def verify(self, frames: list[bytes]) -> LivenessResult:
        return LivenessResult(0.96, True, frames[0])


class FakeFacePort:
    def __init__(self, back_contains_face: bool = False) -> None:
        self.back_contains_face = back_contains_face

    def contains_face(self, _image: bytes) -> bool:
        return self.back_contains_face

    def compare(self, _identity_image: bytes, _live_image: bytes) -> FaceComparisonResult:
        return FaceComparisonResult(0.73, True)


class FakeRepository:
    def __init__(self) -> None:
        self.items = {}

    def save(self, verification, _ttl_seconds: int):
        self.items[verification.verification_id] = verification
        return verification

    def find_by_id(self, verification_id):
        return self.items.get(verification_id)


class IdentityVerificationServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = FakeRepository()
        self.service = IdentityVerificationService(
            FakeObjectClient(),
            FakeQualityPort(),
            FakeDocumentPort(),
            FakeFacePort(),
            FakeLivenessPort(),
            self.repository,
            3600,
        )
        self.subject_id = uuid4()
        self.request = IdentityVerificationRequest(
            subject_id=self.subject_id,
            identity_front_key="temp/identity-front.jpg",
            identity_back_key="temp/identity-back.jpg",
            identity_front_url="https://storage.test/front",
            identity_back_url="https://storage.test/back",
            liveness_frames=["data:image/jpeg;base64,YWJj"] * 3,
        )

    def test_verifies_document_liveness_and_face(self) -> None:
        result = self.service.verify(self.request)

        self.assertEqual(IdentityVerificationStatus.VERIFIED, result.status)
        self.assertEqual("079203000001", result.document.identity_number)
        self.assertEqual(0.73, result.scores.face_similarity)
        self.assertEqual(result, self.service.get(result.verification_id, self.subject_id))

    def test_start_returns_processing_before_background_verification(self) -> None:
        result = self.service.start(self.request)

        self.assertEqual(IdentityVerificationStatus.PROCESSING, result.status)
        self.assertEqual(result, self.service.get(result.verification_id, self.subject_id))

        self.service.process(self.request, result)

        completed = self.service.get(result.verification_id, self.subject_id)
        self.assertEqual(IdentityVerificationStatus.VERIFIED, completed.status)

    def test_retries_before_ocr_when_document_quality_is_low(self) -> None:
        service = IdentityVerificationService(
            FakeObjectClient(),
            FakeQualityPort(acceptable=False),
            FakeDocumentPort(),
            FakeFacePort(),
            FakeLivenessPort(),
            self.repository,
            3600,
        )

        result = service.verify(self.request)

        self.assertEqual(IdentityVerificationStatus.RETRY_REQUIRED, result.status)
        self.assertEqual("IMAGE_BLURRY", result.reason_code)

    def test_rejects_front_side_uploaded_in_the_back_slot(self) -> None:
        service = IdentityVerificationService(
            FakeObjectClient(),
            FakeQualityPort(),
            FakeDocumentPort(),
            FakeFacePort(back_contains_face=True),
            FakeLivenessPort(),
            self.repository,
            3600,
        )

        result = service.verify(self.request)

        self.assertEqual(IdentityVerificationStatus.RETRY_REQUIRED, result.status)
        self.assertEqual("DOCUMENT_SIDE_INVALID", result.reason_code)


if __name__ == "__main__":
    unittest.main()
