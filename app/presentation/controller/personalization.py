from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Header

from app.application.dto.request.models import PersonalizationSettingRequest, PostRankRequest
from app.application.dto.response.models import PersonalizationSettingResponse, PostScore
from app.application.exception import ForbiddenException
from app.infrastructure.config.container import activity_ingest_service, post_ranking_service
from app.infrastructure.config.settings import settings

router = APIRouter(prefix="/personalization", tags=["AI Personalization"])


def _from_gateway(secret: str | None) -> bool:
    return bool(settings.GATEWAY_SECRET) and secret == settings.GATEWAY_SECRET


def _current_user(secret: str | None, user_id: str | None) -> UUID:
    # X-User-Id chỉ đáng tin khi đi qua gateway (gateway xóa bản client tự gửi rồi gắn lại từ JWT).
    if not _from_gateway(secret) or not user_id:
        raise ForbiddenException("Bạn cần đăng nhập để xem cài đặt cá nhân hóa.")
    try:
        return UUID(user_id)
    except ValueError as exception:
        raise ForbiddenException("Danh tính không hợp lệ.") from exception


@router.get(
    "/settings",
    response_model=PersonalizationSettingResponse,
    response_model_by_alias=True,
    summary="Người dùng có cho phép dùng hành vi của mình để gợi ý không",
)
def get_setting(
    x_gateway_secret: Annotated[str | None, Header(alias="X-Gateway-Secret")] = None,
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
) -> PersonalizationSettingResponse:
    user_id = _current_user(x_gateway_secret, x_user_id)
    return PersonalizationSettingResponse(enabled=activity_ingest_service.personalization_enabled(user_id))


@router.put(
    "/settings",
    response_model=PersonalizationSettingResponse,
    response_model_by_alias=True,
    summary="Bật / tắt cá nhân hóa; tắt thì xóa hành vi đã lưu",
)
def update_setting(
    request: PersonalizationSettingRequest,
    x_gateway_secret: Annotated[str | None, Header(alias="X-Gateway-Secret")] = None,
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
) -> PersonalizationSettingResponse:
    user_id = _current_user(x_gateway_secret, x_user_id)
    return PersonalizationSettingResponse(enabled=activity_ingest_service.set_personalization(user_id, request.enabled))


@router.post(
    "/posts/rank",
    response_model=list[PostScore],
    response_model_by_alias=True,
    summary="[internal] Chấm điểm bài viết ứng viên cho bảng tin Khám phá",
)
def rank_posts(
    request: PostRankRequest,
    x_gateway_secret: Annotated[str | None, Header(alias="X-Gateway-Secret")] = None,
) -> list[PostScore]:
    if not _from_gateway(x_gateway_secret):
        raise ForbiddenException("Chỉ dịch vụ nội bộ mới được xếp hạng bảng tin.")
    return post_ranking_service.rank(request.user_id, request.candidates)
