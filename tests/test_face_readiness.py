import base64
import unittest
from uuid import uuid4

from app.application.dto.request.models import FaceReadinessRequest
from app.application.port.identity_verification import FaceReadinessResult
from app.application.service.face_readiness_service import FaceReadinessService


class FakeFaceReadinessPort:
    def __init__(self) -> None:
        self.frame = b""

    def analyze(self, frame: bytes) -> FaceReadinessResult:
        self.frame = frame
        return FaceReadinessResult(
            ready=True,
            reason_code="READY",
            message="Sẵn sàng quét khuôn mặt.",
            face_count=1,
            brightness=120.0,
            centered=True,
            frontal=True,
            glasses_free=True,
            eyes_open=True,
            mask_free=True,
            face_box=(0.25, 0.15, 0.5, 0.7),
            yaw_ratio=0.0,
            pitch_ratio=0.5,
            roll_degrees=0.0,
        )


class FaceReadinessServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.port = FakeFaceReadinessPort()
        self.service = FaceReadinessService(self.port)

    def test_decodes_camera_frame_before_analysis(self) -> None:
        frame_bytes = b"camera-frame"
        request = FaceReadinessRequest(
            subject_id=uuid4(),
            frame="data:image/jpeg;base64," + base64.b64encode(frame_bytes).decode(),
        )

        result = self.service.analyze(request)

        self.assertTrue(result.ready)
        self.assertEqual(frame_bytes, self.port.frame)

    def test_rejects_invalid_base64(self) -> None:
        request = FaceReadinessRequest(
            subject_id=uuid4(),
            frame="data:image/jpeg;base64," + ("!" * 32),
        )

        with self.assertRaisesRegex(ValueError, "không hợp lệ"):
            self.service.analyze(request)


if __name__ == "__main__":
    unittest.main()
