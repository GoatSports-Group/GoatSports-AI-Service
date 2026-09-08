from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from app.application.dto.request.models import DemandForecastRequest
from app.application.exception import InsufficientForecastDataException
from app.application.port.persistence import AiDocumentRepository
from app.application.service.model_registry import get_or_create_active_model
from app.domain.entity.models import (
    DemandForecastResult,
    DemandObservation,
)
from app.domain.enums import AiCapability


class DemandForecastService:
    def __init__(self, repository: AiDocumentRepository) -> None:
        self.repository = repository

    def record_observation(self, observation: DemandObservation) -> DemandObservation:
        return self.repository.save_demand_observation(observation)

    def forecast(self, request: DemandForecastRequest) -> list[DemandForecastResult]:
        observations = self.repository.find_demand_observations(
            request.venue_id,
            request.sport_type.value,
        )
        if not observations:
            raise InsufficientForecastDataException(
                "Chưa có dữ liệu lịch sử để dự báo nhu cầu cho sân và môn thể thao này."
            )

        model = get_or_create_active_model(self.repository, AiCapability.DEMAND_FORECAST)
        bucket_delta = timedelta(minutes=request.bucket_minutes)
        bucket_start = request.forecast_start
        results: list[DemandForecastResult] = []
        while bucket_start < request.forecast_end:
            bucket_end = min(bucket_start + bucket_delta, request.forecast_end)
            predicted, lower, upper = self._predict_bucket(bucket_start, observations)
            forecast = DemandForecastResult(
                venue_id=request.venue_id,
                sport_type=request.sport_type,
                bucket_start=bucket_start,
                bucket_end=bucket_end,
                predicted_demand=round(predicted, 4),
                lower_bound=round(lower, 4),
                upper_bound=round(upper, 4),
                model_version_id=model.model_version_id,
            )
            self.repository.save_forecast(forecast)
            results.append(forecast)
            bucket_start = bucket_end
        return results

    def get_forecasts(self, venue_id, sport_type) -> list[DemandForecastResult]:
        return self.repository.find_forecasts(venue_id, sport_type.value)

    @classmethod
    def _predict_bucket(
        cls,
        target: datetime,
        observations: list[DemandObservation],
    ) -> tuple[float, float, float]:
        exact = [
            item
            for item in observations
            if cls._as_utc(item.bucket_start).weekday() == cls._as_utc(target).weekday()
            and cls._as_utc(item.bucket_start).hour == cls._as_utc(target).hour
        ]
        same_hour = [
            item
            for item in observations
            if cls._as_utc(item.bucket_start).hour == cls._as_utc(target).hour
        ]
        samples = exact or same_hour or observations
        newest = max(cls._as_utc(item.bucket_start) for item in observations)
        weights = [
            0.92
            ** max(0.0, (newest - cls._as_utc(item.bucket_start)).total_seconds() / (7 * 86_400))
            for item in samples
        ]
        total_weight = sum(weights) or 1.0
        mean = sum(item.occupancy_rate * weight for item, weight in zip(samples, weights)) / total_weight
        variance = (
            sum(weight * (item.occupancy_rate - mean) ** 2 for item, weight in zip(samples, weights))
            / total_weight
        )
        deviation = math.sqrt(max(0.0, variance))
        return mean, max(0.0, mean - 1.96 * deviation), min(1.0, mean + 1.96 * deviation)

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)

