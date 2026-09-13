from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Query, status

from app.application.dto.request.models import FaceReadinessRequest, IdentityVerificationRequest
from app.application.dto.response.models import FaceReadinessResponse, IdentityVerificationResponse
from app.infrastructure.config.container import (
    face_readiness_service,
    identity_verification_service,
)

router = APIRouter(prefix="/identity-verifications", tags=["Identity Verification"])


@router.post(
    "/readiness",
    response_model=FaceReadinessResponse,
    summary="Analyze whether a live camera frame is ready for identity capture",
)
def analyze_face_readiness(request: FaceReadinessRequest) -> FaceReadinessResponse:
    result = face_readiness_service.analyze(request)
    return FaceReadinessResponse(
        ready=result.ready,
        reason_code=result.reason_code,
        message=result.message,
        face_count=result.face_count,
        brightness=result.brightness,
        centered=result.centered,
        frontal=result.frontal,
        glasses_free=result.glasses_free,
        eyes_open=result.eyes_open,
        mask_free=result.mask_free,
        face_box=result.face_box,
        yaw_ratio=result.yaw_ratio,
        pitch_ratio=result.pitch_ratio,
        roll_degrees=result.roll_degrees,
    )


@router.post(
    "",
    response_model=IdentityVerificationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Verify Vietnamese citizen ID, liveness and face ownership",
)
def verify_identity(
    request: IdentityVerificationRequest,
    background_tasks: BackgroundTasks,
) -> IdentityVerificationResponse:
    verification = identity_verification_service.start(request)
    background_tasks.add_task(identity_verification_service.process, request, verification)
    return IdentityVerificationResponse(verification=verification)


@router.get(
    "/{verification_id}",
    response_model=IdentityVerificationResponse,
    summary="Read a short-lived identity verification result",
)
def get_identity_verification(
    verification_id: UUID,
    subject_id: UUID = Query(alias="subjectId"),
) -> IdentityVerificationResponse:
    return IdentityVerificationResponse(
        verification=identity_verification_service.get(verification_id, subject_id)
    )
