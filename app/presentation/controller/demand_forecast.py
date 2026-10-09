from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Header

from app.application.exception import ForbiddenException
from app.domain.entity.models import DemandForecastResult
from app.infrastructure.config.container import demand_forecast_service
from app.infrastructure.config.settings import settings

router = APIRouter(prefix="/demand-forecast", tags=["AI Demand Forecast"])


@router.get(
    "/venues/{venue_id}",
    response_model=DemandForecastResult,
    response_model_by_alias=True,
    summary="Dự báo tỉ lệ lấp đầy 7 ngày tới của cơ sở (Chronos-Bolt), kèm khoảng 10–90% và điểm kiểm chứng",
)
def venue_forecast(
    venue_id: UUID,
    x_user_id: Annotated[UUID, Header(alias="X-User-Id")],
    x_gateway_secret: Annotated[str | None, Header(alias="X-Gateway-Secret")] = None,
) -> DemandForecastResult:
    # Danh tính chỉ tin được khi request đi qua gateway; venue-service kiểm chủ sở hữu cơ sở.
    if not settings.GATEWAY_SECRET or x_gateway_secret != settings.GATEWAY_SECRET:
        raise ForbiddenException("Yêu cầu phải đi qua API Gateway.")
    return demand_forecast_service.forecast(x_user_id, venue_id)
