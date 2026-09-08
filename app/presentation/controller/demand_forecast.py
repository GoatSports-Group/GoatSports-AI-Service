from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, status

from app.application.dto.request.models import DemandForecastRequest
from app.domain.entity.models import DemandForecastResult, DemandObservation
from app.domain.enums import SportType
from app.infrastructure.config.container import demand_forecast_service

router = APIRouter(prefix="/demand-forecast", tags=["AI Demand Forecast"])


@router.post(
    "/observations",
    response_model=DemandObservation,
    status_code=status.HTTP_201_CREATED,
    summary="Ghi nhận dữ liệu lấp đầy sân",
)
def record_observation(observation: DemandObservation) -> DemandObservation:
    return demand_forecast_service.record_observation(observation)


@router.post(
    "",
    response_model=list[DemandForecastResult],
    summary="Dự báo nhu cầu đặt sân theo chuỗi thời gian",
)
def generate_forecast(request: DemandForecastRequest) -> list[DemandForecastResult]:
    return demand_forecast_service.forecast(request)


@router.get(
    "",
    response_model=list[DemandForecastResult],
    summary="Xem kết quả dự báo nhu cầu",
)
def get_forecasts(
    venue_id: Annotated[UUID, Query()],
    sport_type: Annotated[SportType, Query()],
) -> list[DemandForecastResult]:
    return demand_forecast_service.get_forecasts(venue_id, sport_type)
