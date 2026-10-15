from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Header, Query

from app.application.dto.response.models import VenueRecommendation
from app.domain.enums import SportType
from app.infrastructure.config.container import recommendation_service
from app.infrastructure.config.settings import settings

router = APIRouter(prefix="/recommendations", tags=["AI Recommendations"])


@router.get(
    "/venues",
    response_model=list[VenueRecommendation],
    summary="Gợi ý sân thể thao",
)
def get_recommended_venues(
    x_gateway_secret: Annotated[str | None, Header(alias="X-Gateway-Secret")] = None,
    x_user_id: Annotated[UUID | None, Header(alias="X-User-Id")] = None,
    user_lat: Annotated[float | None, Query(ge=-90, le=90, description="Vĩ độ người dùng")] = None,
    user_lng: Annotated[float | None, Query(ge=-180, le=180, description="Kinh độ người dùng")] = None,
    sport_type: Annotated[SportType | None, Query(description="Môn thể thao")] = None,
    max_results: Annotated[int, Query(ge=1, le=20, description="Số lượng gợi ý")] = 5,
    radius_km: Annotated[float, Query(ge=0.5, le=100, description="Bán kính tìm (km)")] = 30.0,
) -> list[VenueRecommendation]:
    # Thói quen chỉ lấy theo X-User-Id do gateway gắn; gọi thẳng mà tự gửi header thì coi như khách.
    trusted = bool(settings.GATEWAY_SECRET) and x_gateway_secret == settings.GATEWAY_SECRET
    return recommendation_service.get_recommendations(
        user_id=x_user_id if trusted else None,
        user_lat=user_lat,
        user_lng=user_lng,
        sport_type=sport_type,
        max_results=max_results,
        radius_km=radius_km,
    )
