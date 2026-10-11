from typing import Annotated

from fastapi import APIRouter, Header

from app.application.dto.response.models import AiAdminStatus
from app.application.exception import ForbiddenException
from app.infrastructure.config.container import ai_status_service
from app.infrastructure.config.settings import settings
from app.presentation.controller.capabilities import capability_statuses

router = APIRouter(prefix="/admin", tags=["AI Admin"])

REQUIRED_AUTHORITY = "GET_AI_STATUS"


@router.get(
    "/status",
    response_model=AiAdminStatus,
    response_model_by_alias=True,
    summary="Trạng thái AI cho admin: năng lực, model đang phục vụ, MongoDB/Redis",
)
def admin_status(
    x_gateway_secret: Annotated[str | None, Header(alias="X-Gateway-Secret")] = None,
    x_user_authorities: Annotated[str | None, Header(alias="X-User-Authorities")] = None,
) -> AiAdminStatus:
    # Gateway đặt X-User-Authorities từ JWT đã xác thực (và xóa bản client tự gửi); chỉ tin khi có secret.
    if not settings.GATEWAY_SECRET or x_gateway_secret != settings.GATEWAY_SECRET:
        raise ForbiddenException("Yêu cầu phải đi qua API Gateway.")
    if REQUIRED_AUTHORITY not in (x_user_authorities or "").split(","):
        raise ForbiddenException("Bạn không có quyền xem trạng thái AI.")
    return ai_status_service.status(capability_statuses())
