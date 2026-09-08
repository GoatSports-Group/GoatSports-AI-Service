from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Header, Query, status

from app.application.dto.request.models import UserInteractionRequest
from app.application.dto.response.models import PersonalizedRecommendation
from app.domain.entity.models import UserInteraction
from app.domain.enums import RecommendationTargetType
from app.infrastructure.config.container import collaborative_filtering_service

router = APIRouter(prefix="/personalization", tags=["AI Personalization"])


@router.post(
    "/interactions",
    response_model=UserInteraction,
    status_code=status.HTTP_201_CREATED,
    summary="Ghi nhận hành vi phục vụ cá nhân hóa",
)
def record_interaction(
    request: UserInteractionRequest,
    x_user_id: Annotated[UUID, Header(alias="X-User-Id")],
) -> UserInteraction:
    interaction = UserInteraction(user_id=x_user_id, **request.model_dump())
    return collaborative_filtering_service.record(interaction)


@router.get(
    "/feed",
    response_model=list[PersonalizedRecommendation],
    summary="Xếp hạng bảng tin bằng collaborative filtering",
)
def recommend_feed(
    x_user_id: Annotated[UUID, Header(alias="X-User-Id")],
    max_results: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[PersonalizedRecommendation]:
    return collaborative_filtering_service.recommend(
        x_user_id,
        RecommendationTargetType.POST,
        max_results,
    )
