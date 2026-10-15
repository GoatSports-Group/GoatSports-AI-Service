from typing import Annotated

from fastapi import APIRouter, Header, status

from app.application.dto.request.models import ChatbotFeedback, ChatbotQuery, KnowledgeDocumentRequest
from app.application.dto.response.models import ChatbotResponse, KnowledgeIngestResponse
from app.application.exception import ForbiddenException
from app.application.service.assistant_service import identity_from_headers
from app.infrastructure.config.container import assistant_service, rag_chatbot_service
from app.infrastructure.config.settings import settings

router = APIRouter(prefix="/chatbot", tags=["AI Sports Assistant"])


@router.post(
    "/knowledge",
    response_model=KnowledgeIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="[internal] Nạp tài liệu vào kho tri thức RAG",
)
def ingest_knowledge(
    request: KnowledgeDocumentRequest,
    x_gateway_secret: Annotated[str | None, Header(alias="X-Gateway-Secret")] = None,
) -> KnowledgeIngestResponse:
    # Chỉ dịch vụ nội bộ: kho tri thức chính nạp từ thư mục knowledge/ lúc khởi động; gateway cũng chặn đường này.
    if not settings.GATEWAY_SECRET or x_gateway_secret != settings.GATEWAY_SECRET:
        raise ForbiddenException("Chỉ dịch vụ nội bộ mới được nạp tài liệu vào kho tri thức.")
    return rag_chatbot_service.ingest(request)


@router.post(
    "/query",
    response_model=ChatbotResponse,
    response_model_by_alias=True,
    summary="Hỏi trợ lý GOAT AI: tài liệu (RAG) + công cụ đọc dữ liệu thật + nhớ vài lượt hội thoại",
)
def ask_chatbot(
    query: ChatbotQuery,
    x_gateway_secret: Annotated[str | None, Header(alias="X-Gateway-Secret")] = None,
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
    x_user_email: Annotated[str | None, Header(alias="X-User-Email")] = None,
    x_user_username: Annotated[str | None, Header(alias="X-User-Username")] = None,
    x_user_roles: Annotated[str | None, Header(alias="X-User-Roles")] = None,
    x_user_authorities: Annotated[str | None, Header(alias="X-User-Authorities")] = None,
) -> ChatbotResponse:
    # Danh tính chỉ đáng tin khi request đi qua gateway (gateway xóa X-User-* do client tự gửi). Gọi thẳng vào
    # ai-service mà tự gắn X-User-Id thì bị coi là khách, nên không xem được đơn của người khác.
    trusted = bool(settings.GATEWAY_SECRET) and x_gateway_secret == settings.GATEWAY_SECRET
    identity = (
        identity_from_headers(x_user_id, x_user_email, x_user_username, x_user_roles, x_user_authorities)
        if trusted
        else None
    )
    return assistant_service.answer(query, identity)


@router.post(
    "/feedback",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Người dùng báo câu trả lời chưa hữu ích: câu hỏi vào hàng chờ để admin bổ sung kiến thức",
)
def report_unhelpful_answer(
    feedback: ChatbotFeedback,
    x_gateway_secret: Annotated[str | None, Header(alias="X-Gateway-Secret")] = None,
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
) -> dict[str, bool]:
    trusted = bool(settings.GATEWAY_SECRET) and x_gateway_secret == settings.GATEWAY_SECRET
    identity = identity_from_headers(x_user_id, None, None, None, None) if trusted else None
    accepted = assistant_service.feedback(feedback.question, feedback.answer, identity,
                                          str(feedback.conversation_id) if feedback.conversation_id else None)
    return {"accepted": accepted}
