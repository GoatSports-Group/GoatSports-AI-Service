from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Header, status

from app.application.dto.request.models import ChatbotQuery, KnowledgeDocumentRequest
from app.application.dto.response.models import ChatbotResponse, KnowledgeIngestResponse
from app.application.exception import ForbiddenException
from app.infrastructure.config.container import rag_chatbot_service
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
    summary="Hỏi đáp trợ lý thể thao GoatSports bằng RAG",
)
def ask_chatbot(
    query: ChatbotQuery,
    x_user_id: Annotated[UUID | None, Header(alias="X-User-Id")] = None,
) -> ChatbotResponse:
    return rag_chatbot_service.answer_query(query, x_user_id)
