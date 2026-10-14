from typing import Annotated

from fastapi import APIRouter, Header

from app.application.dto.request.models import ContentModerationRequest
from app.application.dto.response.models import ContentModerationVerdict
from app.application.exception import ForbiddenException
from app.infrastructure.config.container import content_moderation_service
from app.infrastructure.config.settings import settings

router = APIRouter(prefix="/moderation", tags=["AI Content Moderation"])


@router.post(
    "/classify",
    response_model=ContentModerationVerdict,
    response_model_by_alias=True,
    summary="[internal] Gợi ý nhãn vi phạm, mức độ 0–3 và lý do cho một đoạn chữ; 503 khi LLM chưa sẵn sàng",
)
def classify(
    request: ContentModerationRequest,
    x_gateway_secret: Annotated[str | None, Header(alias="X-Gateway-Secret")] = None,
) -> ContentModerationVerdict:
    # Chỉ social-service và venue-service gọi (job nền), không đi qua gateway nên không có danh tính người dùng.
    if not settings.GATEWAY_SECRET or x_gateway_secret != settings.GATEWAY_SECRET:
        raise ForbiddenException("Chỉ dịch vụ nội bộ mới được gọi phân loại nội dung.")
    return content_moderation_service.classify(request)
