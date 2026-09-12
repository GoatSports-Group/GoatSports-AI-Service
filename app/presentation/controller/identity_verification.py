from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Query, status

from app.application.dto.request.models import IdentityVerificationRequest
from app.application.dto.response.models import IdentityVerificationResponse
from app.infrastructure.config.container import identity_verification_service

router = APIRouter(prefix="/identity-verifications", tags=["Identity Verification"])


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
