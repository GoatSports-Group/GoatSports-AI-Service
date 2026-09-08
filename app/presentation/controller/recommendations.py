from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Header, Query

from app.application.dto.response.models import VenueRecommendation
from app.domain.enums import SportType
from app.infrastructure.config.container import recommendation_service

router = APIRouter(prefix="/recommendations", tags=["AI Recommendations"])


@router.get(
    "/venues",
    response_model=list[VenueRecommendation],
    summary="Gợi ý sân thể thao",
)
def get_recommended_venues(
    x_user_id: Annotated[UUID | None, Header(alias="X-User-Id")] = None,
    user_lat: Annotated[float | None, Query(ge=-90, le=90, description="Vĩ độ người dùng")] = None,
    user_lng: Annotated[float | None, Query(ge=-180, le=180, description="Kinh độ người dùng")] = None,
    sport_type: Annotated[SportType | None, Query(description="Môn thể thao")] = None,
    max_results: Annotated[int, Query(ge=1, le=20, description="Số lượng gợi ý")] = 5,
) -> list[VenueRecommendation]:
    return recommendation_service.get_recommendations(
        user_id=x_user_id,
        user_lat=user_lat,
        user_lng=user_lng,
        sport_type=sport_type,
        max_results=max_results,
    )
