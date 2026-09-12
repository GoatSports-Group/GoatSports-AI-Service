from __future__ import annotations

import base64
import binascii
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from uuid import UUID

from app.application.dto.request.models import IdentityVerificationRequest
from app.application.exception import CapabilityUnavailableException, ResourceNotFoundException
from app.application.port.identity_verification import (
    BinaryObjectClient,
    FaceVerificationPort,
    IdentityDocumentRecognitionPort,
    IdentityVerificationRepository,
    ImageQualityPort,
    LivenessPort,
)
from app.domain.base import utc_now
from app.domain.entity.models import (
    IdentityDocumentData,
    IdentityVerification,
    IdentityVerificationScores,
)
from app.domain.enums import IdentityDocumentType, IdentityVerificationStatus

logger = logging.getLogger(__name__)


class IdentityVerificationService:
    def __init__(
        self,
        object_client: BinaryObjectClient,
        quality_port: ImageQualityPort,
        document_port: IdentityDocumentRecognitionPort,
        face_port: FaceVerificationPort,
        liveness_port: LivenessPort,
        repository: IdentityVerificationRepository,
        ttl_seconds: int,
    ) -> None:
        self.object_client = object_client
        self.quality_port = quality_port
        self.document_port = document_port
        self.face_port = face_port
        self.liveness_port = liveness_port
        self.repository = repository
        self.ttl_seconds = ttl_seconds

    def verify(self, request: IdentityVerificationRequest) -> IdentityVerification:
        verification = self.start(request)
        return self._verify(request, verification)

    def start(self, request: IdentityVerificationRequest) -> IdentityVerification:
        expires_at = utc_now() + timedelta(seconds=self.ttl_seconds)
        verification = IdentityVerification(
            subject_id=request.subject_id,
            identity_front_key=request.identity_front_key,
            identity_back_key=request.identity_back_key,
            expires_at=expires_at,
        )
        return self.repository.save(verification, self.ttl_seconds)

    def process(
        self,
        request: IdentityVerificationRequest,
        verification: IdentityVerification,
    ) -> None:
        started_at = time.perf_counter()
        try:
            self._verify(request, verification)
        except Exception:
            logger.exception(
                "Identity verification failed verification_id=%s",
                verification.verification_id,
            )
            self._finish(
                verification,
                IdentityVerificationStatus.RETRY_REQUIRED,
                "PROCESSING_FAILED",
                "Không thể xử lý xác minh lúc này. Vui lòng thử lại.",
                0.0,
                0.0,
            )
        finally:
            logger.info(
                "Identity verification completed verification_id=%s total_duration_ms=%.0f",
                verification.verification_id,
                (time.perf_counter() - started_at) * 1000,
            )

    def _verify(
        self,
        request: IdentityVerificationRequest,
        verification: IdentityVerification,
    ) -> IdentityVerification:
        try:
            stage_started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="identity-download") as executor:
                front_future = executor.submit(
                    self.object_client.download,
                    request.identity_front_url,
                )
                back_future = executor.submit(
                    self.object_client.download,
                    request.identity_back_url,
                )
                front = front_future.result()
                back = back_future.result()
            stage_started = self._log_stage(verification, "download", stage_started)

            frames = [self._decode_frame(frame) for frame in request.liveness_frames]

            front_quality = self.quality_port.assess(front)
            back_quality = self.quality_port.assess(back)
            stage_started = self._log_stage(verification, "quality", stage_started)
            if not front_quality.acceptable or not back_quality.acceptable:
                return self._finish(
                    verification,
                    IdentityVerificationStatus.RETRY_REQUIRED,
                    front_quality.reason_code or back_quality.reason_code or "DOCUMENT_QUALITY_LOW",
                    "Ảnh CCCD bị mờ, chói, quá tối hoặc không nhìn đủ giấy tờ.",
                    front_quality.score,
                    back_quality.score,
                )

            back_contains_face = self.face_port.contains_face(back)
            stage_started = self._log_stage(verification, "document-side", stage_started)
            if back_contains_face:
                return self._finish(
                    verification,
                    IdentityVerificationStatus.RETRY_REQUIRED,
                    "DOCUMENT_SIDE_INVALID",
                    "Vui lòng tải đúng mặt trước và mặt sau của CCCD.",
                    front_quality.score,
                    back_quality.score,
                )

            front_result = self.document_port.recognize_front(front)
            stage_started = self._log_stage(verification, "document-read", stage_started)
            if not front_result.expected_side:
                return self._finish(
                    verification,
                    IdentityVerificationStatus.RETRY_REQUIRED,
                    "DOCUMENT_SIDE_INVALID",
                    "Vui lòng tải đúng mặt trước và mặt sau của CCCD.",
                    front_quality.score,
                    back_quality.score,
                    ocr_confidence=front_result.confidence,
                )

            document = front_result.data
            if document.document_type == IdentityDocumentType.UNKNOWN:
                return self._finish(
                    verification,
                    IdentityVerificationStatus.REJECTED,
                    "UNSUPPORTED_DOCUMENT",
                    "Không nhận diện được CCCD hoặc thẻ Căn cước Việt Nam.",
                    front_quality.score,
                    back_quality.score,
                    ocr_confidence=front_result.confidence,
                )
            if not document.identity_number or not document.full_name:
                return self._finish(
                    verification,
                    IdentityVerificationStatus.MANUAL_REVIEW,
                    "OCR_REQUIRED_FIELDS_MISSING",
                    "Không đọc đủ số định danh và họ tên trên CCCD.",
                    front_quality.score,
                    back_quality.score,
                    ocr_confidence=front_result.confidence,
                    document=document,
                )
            if document.expiry_date and document.expiry_date < utc_now().date():
                return self._finish(
                    verification,
                    IdentityVerificationStatus.REJECTED,
                    "DOCUMENT_EXPIRED",
                    "CCCD đã hết hạn.",
                    front_quality.score,
                    back_quality.score,
                    ocr_confidence=front_result.confidence,
                    document=document,
                )

            liveness = self.liveness_port.verify(frames)
            stage_started = self._log_stage(verification, "liveness", stage_started)
            if not liveness.live or not liveness.representative_frame:
                return self._finish(
                    verification,
                    IdentityVerificationStatus.RETRY_REQUIRED,
                    "LIVENESS_FAILED",
                    "Không xác nhận được người thật trước camera. Vui lòng thử lại ở nơi đủ sáng.",
                    front_quality.score,
                    back_quality.score,
                    ocr_confidence=front_result.confidence,
                    liveness_score=liveness.score,
                    document=document,
                )

            face = self.face_port.compare(front, liveness.representative_frame)
            self._log_stage(verification, "face-match", stage_started)
            status = (
                IdentityVerificationStatus.VERIFIED
                if face.matched
                else IdentityVerificationStatus.REJECTED
            )
            reason = None if face.matched else "FACE_MISMATCH"
            message = (
                "Xác minh CCCD và khuôn mặt thành công."
                if face.matched
                else "Khuôn mặt hiện tại không khớp với ảnh trên CCCD."
            )
            return self._finish(
                verification,
                status,
                reason,
                message,
                front_quality.score,
                back_quality.score,
                ocr_confidence=front_result.confidence,
                face_similarity=face.similarity,
                liveness_score=liveness.score,
                document=document,
            )
        except CapabilityUnavailableException:
            raise
        except (ValueError, binascii.Error) as exception:
            return self._finish(
                verification,
                IdentityVerificationStatus.RETRY_REQUIRED,
                "INVALID_IMAGE_DATA",
                str(exception),
                0.0,
                0.0,
            )

    def get(self, verification_id: UUID, subject_id: UUID) -> IdentityVerification:
        verification = self.repository.find_by_id(verification_id)
        if verification is None or verification.subject_id != subject_id:
            raise ResourceNotFoundException("Không tìm thấy phiên xác minh danh tính.")
        return verification

    def _finish(
        self,
        verification: IdentityVerification,
        status: IdentityVerificationStatus,
        reason_code: str | None,
        message: str,
        front_quality: float,
        back_quality: float,
        ocr_confidence: float = 0.0,
        face_similarity: float = 0.0,
        liveness_score: float = 0.0,
        document: IdentityDocumentData | None = None,
    ) -> IdentityVerification:
        completed = verification.model_copy(
            update={
                "status": status,
                "reason_code": reason_code,
                "message": message,
                "document": document,
                "scores": IdentityVerificationScores(
                    front_quality=front_quality,
                    back_quality=back_quality,
                    ocr_confidence=ocr_confidence,
                    face_similarity=face_similarity,
                    liveness_score=liveness_score,
                ),
            }
        )
        return self.repository.save(completed, self.ttl_seconds)

    @staticmethod
    def _decode_frame(frame: str) -> bytes:
        _, payload = frame.split(",", 1)
        decoded = base64.b64decode(payload, validate=True)
        if not decoded or len(decoded) > 2 * 1024 * 1024:
            raise ValueError("Ảnh camera không hợp lệ hoặc vượt quá 2 MB.")
        return decoded

    @staticmethod
    def _log_stage(
        verification: IdentityVerification,
        stage: str,
        started_at: float,
    ) -> float:
        completed_at = time.perf_counter()
        logger.info(
            "Identity verification stage completed verification_id=%s stage=%s duration_ms=%.0f",
            verification.verification_id,
            stage,
            (completed_at - started_at) * 1000,
        )
        return completed_at
