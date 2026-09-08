from typing import Annotated

from fastapi import APIRouter, Query, status

from app.application.dto.request.models import SemanticDocumentRequest
from app.application.dto.response.models import SemanticSearchResult
from app.domain.entity.models import SemanticDocument
from app.domain.enums import SearchDocumentType
from app.infrastructure.config.container import semantic_search_service

router = APIRouter(prefix="/semantic-search", tags=["AI Semantic Search"])


@router.post(
    "/documents",
    response_model=SemanticDocument,
    status_code=status.HTTP_201_CREATED,
    summary="Lập chỉ mục tài liệu tìm kiếm",
)
def index_document(request: SemanticDocumentRequest) -> SemanticDocument:
    return semantic_search_service.index(request)


@router.get(
    "",
    response_model=list[SemanticSearchResult],
    summary="Tìm sân, bài viết và sự kiện theo ngữ nghĩa",
)
def search(
    query: Annotated[str, Query(min_length=1, max_length=500)],
    document_type: Annotated[SearchDocumentType | None, Query()] = None,
    max_results: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[SemanticSearchResult]:
    return semantic_search_service.search(query, document_type, max_results)
