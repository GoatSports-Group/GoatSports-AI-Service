from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Header, Query, Response, status

from app.application.dto.request.models import KnowledgeGapAnswer
from app.application.dto.response.models import (
    FaqView,
    KnowledgeGapAnswerResult,
    KnowledgeGapPage,
    KnowledgeGapView,
)
from app.application.exception import ForbiddenException
from app.domain.entity.models import KnowledgeGap
from app.domain.enums import KnowledgeGapStatus
from app.infrastructure.config.container import knowledge_gap_service
from app.infrastructure.config.settings import settings

router = APIRouter(prefix="/assistant-admin", tags=["AI Assistant Knowledge (admin)"])


def _admin(secret: str | None, user: str | None, roles: str | None) -> str:
    # Danh tính chỉ đáng tin khi đi qua gateway; trang này chỉ dành cho quản trị viên (ADMIN).
    trusted = bool(settings.GATEWAY_SECRET) and secret == settings.GATEWAY_SECRET
    role_names = {role.strip().removeprefix("ROLE_") for role in (roles or "").split(",")}
    if not trusted or "ADMIN" not in role_names:
        raise ForbiddenException("Chỉ quản trị viên mới quản lý kiến thức của trợ lý.")
    return user or "admin"


def _view(gap: KnowledgeGap) -> KnowledgeGapView:
    return KnowledgeGapView(**gap.model_dump(exclude={"embedding", "knowledge_document_id"}),)


Secret = Annotated[str | None, Header(alias="X-Gateway-Secret")]
UserEmail = Annotated[str | None, Header(alias="X-User-Email")]
Roles = Annotated[str | None, Header(alias="X-User-Roles")]


@router.get("/gaps", response_model=KnowledgeGapPage, response_model_by_alias=True,
            summary="Câu hỏi trợ lý chưa trả lời được, hỏi nhiều nhất trước")
def list_gaps(
    x_gateway_secret: Secret = None,
    x_user_email: UserEmail = None,
    x_user_roles: Roles = None,
    status_filter: Annotated[KnowledgeGapStatus | None, Query(alias="status")] = KnowledgeGapStatus.OPEN,
    page: Annotated[int, Query(ge=0)] = 0,
    size: Annotated[int, Query(ge=1, le=50)] = 20,
) -> KnowledgeGapPage:
    _admin(x_gateway_secret, x_user_email, x_user_roles)
    gaps, total = knowledge_gap_service.list(status_filter, page, size)
    return KnowledgeGapPage(items=[_view(gap) for gap in gaps], total=total, page=page, size=size)


@router.post("/gaps/{gap_id}/answer", response_model=KnowledgeGapAnswerResult, response_model_by_alias=True,
             summary="Trả lời câu hỏi: lưu thành tài liệu FAQ, trợ lý dùng được ngay")
def answer_gap(
    gap_id: UUID,
    request: KnowledgeGapAnswer,
    x_gateway_secret: Secret = None,
    x_user_email: UserEmail = None,
    x_user_roles: Roles = None,
) -> KnowledgeGapAnswerResult:
    admin = _admin(x_gateway_secret, x_user_email, x_user_roles)
    gap, answerable = knowledge_gap_service.answer(gap_id, request.question, request.answer, admin)
    return KnowledgeGapAnswerResult(gap=_view(gap), answerable=answerable)


@router.post("/gaps/{gap_id}/dismiss", response_model=KnowledgeGapView, response_model_by_alias=True,
             summary="Bỏ qua: lạc đề, hoặc cần tính năng mới chứ không phải câu trả lời")
def dismiss_gap(
    gap_id: UUID,
    x_gateway_secret: Secret = None,
    x_user_email: UserEmail = None,
    x_user_roles: Roles = None,
) -> KnowledgeGapView:
    admin = _admin(x_gateway_secret, x_user_email, x_user_roles)
    return _view(knowledge_gap_service.dismiss(gap_id, admin))


@router.get("/faqs", response_model=list[FaqView], response_model_by_alias=True,
            summary="Các câu trả lời bổ sung admin đã thêm")
def list_faqs(
    x_gateway_secret: Secret = None,
    x_user_email: UserEmail = None,
    x_user_roles: Roles = None,
) -> list[FaqView]:
    _admin(x_gateway_secret, x_user_email, x_user_roles)
    return [
        FaqView(document_id=document.document_id, question=document.title,
                answer=gap.answer if gap else None, updated_at=document.updated_at,
                resolved_by=gap.resolved_by if gap else None, gap_id=gap.gap_id if gap else None,
                asked_count=gap.count if gap else 0)
        for document, gap in knowledge_gap_service.faqs()
    ]


@router.delete("/faqs/{document_id}", status_code=status.HTTP_204_NO_CONTENT,
               summary="Gỡ câu trả lời bổ sung; câu hỏi quay lại hàng chờ")
def delete_faq(
    document_id: UUID,
    x_gateway_secret: Secret = None,
    x_user_email: UserEmail = None,
    x_user_roles: Roles = None,
) -> Response:
    _admin(x_gateway_secret, x_user_email, x_user_roles)
    knowledge_gap_service.delete_faq(document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
