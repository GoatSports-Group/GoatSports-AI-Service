"""Dự báo nhu cầu đặt sân (AI-08) bằng Chronos-Bolt: tỉ lệ lấp đầy từng giờ của 7 ngày tới, kèm khoảng 10–90%.

Không có nhánh thống kê dự phòng: dự báo luôn do mô hình sinh ra. Độ tin cậy được nói thật bằng hai cách:
khoảng quantile (dữ liệu ít → khoảng rộng) và chấm điểm trên tuần vừa qua so với cách đoán "giống tuần trước".
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import NAMESPACE_URL, UUID, uuid5

import numpy as np

from app.application.exception import (
    CapabilityUnavailableException,
    InsufficientForecastDataException,
)
from app.application.port.client import VenueServiceClient
from app.application.port.forecast import DemandForecaster
from app.application.port.persistence import AiDocumentRepository
from app.domain.entity.models import (
    AIModelVersion,
    DemandBacktest,
    DemandForecastHour,
    DemandForecastResult,
)
from app.domain.enums import AiCapability, AiModelStatus

HORIZON_HOURS = 168
HISTORY_HOURS = 2048
BACKTEST_HOURS = 168
MIN_HISTORY_HOURS = 24
CACHE_SECONDS = 15 * 60


class DemandForecastService:
    def __init__(
        self,
        venue_client: VenueServiceClient,
        repository: AiDocumentRepository,
        forecaster: DemandForecaster | None,
        clock=lambda: datetime.now(timezone.utc),
    ) -> None:
        self.venue_client = venue_client
        self.repository = repository
        self.forecaster = forecaster
        self.clock = clock

    def forecast(self, owner_id: UUID, venue_id: UUID) -> DemandForecastResult:
        if self.forecaster is None:
            raise CapabilityUnavailableException("Mô hình dự báo nhu cầu (Chronos-Bolt) chưa được cài trong ai-service.")
        now = self.clock()
        cached = self.repository.find_venue_forecast(venue_id)
        if (
            cached is not None
            and cached.owner_id == owner_id
            and cached.model_revision == self.forecaster.revision
            and (now - _utc(cached.generated_at)).total_seconds() < CACHE_SECONDS
        ):
            return cached

        history = self.venue_client.owner_demand_history(owner_id, venue_id, HISTORY_HOURS)
        values = np.array([np.nan if value is None else value for value in history.occupancy], dtype=np.float32)
        observed = int((~np.isnan(values)).sum())
        if observed < MIN_HISTORY_HOURS:
            raise InsufficientForecastDataException(
                "Cơ sở cần ít nhất 1 ngày dữ liệu đặt sân để dự báo nhu cầu."
            )

        lower, median, upper = self._predict(values, HORIZON_HOURS)
        forecast_start = history.start + timedelta(hours=len(values))
        step = self.forecaster.prediction_length
        hours = [
            DemandForecastHour(
                start=forecast_start + timedelta(hours=index),
                lower=round(float(lower[index]), 4),
                median=round(float(median[index]), 4),
                upper=round(float(upper[index]), 4),
                extended=index >= step,
            )
            for index in range(HORIZON_HOURS)
        ]
        model = self._model_version()
        result = DemandForecastResult(
            venue_id=venue_id,
            owner_id=owner_id,
            model_version_id=model.model_version_id,
            model_name=self.forecaster.model_name,
            model_revision=self.forecaster.revision,
            generated_at=now,
            forecast_start=forecast_start,
            court_count=history.court_count,
            open_time=history.open_time,
            close_time=history.close_time,
            history_hours=observed,
            hours=hours,
            backtest=self.backtest(values),
        )
        return self.repository.save_venue_forecast(result)

    def backtest(self, values: np.ndarray) -> DemandBacktest | None:
        """Dự báo tuần vừa qua chỉ từ dữ liệu trước nó; so trung vị với thực tế và với "giống tuần trước"."""
        if values.size < 2 * BACKTEST_HOURS:
            return None
        train, actual = values[:-BACKTEST_HOURS], values[-BACKTEST_HOURS:]
        previous_week = values[-2 * BACKTEST_HOURS:-BACKTEST_HOURS]
        if int((~np.isnan(train)).sum()) < MIN_HISTORY_HOURS:
            return None
        known = ~np.isnan(actual) & ~np.isnan(previous_week)
        if int(known.sum()) < MIN_HISTORY_HOURS:
            return None
        lower, median, upper = self._predict(train, BACKTEST_HOURS)
        inside = (actual >= lower - 1e-6) & (actual <= upper + 1e-6)
        return DemandBacktest(
            hours=int(known.sum()),
            model_mae=round(float(np.abs(median - actual)[known].mean()), 4),
            baseline_mae=round(float(np.abs(previous_week - actual)[known].mean()), 4),
            coverage=round(float(inside[known].mean()), 4),
        )

    def _predict(self, history: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        quantiles = list(self.forecaster.quantiles)
        # Các đầu ra quantile của mô hình có thể "chéo nhau" ở vài giờ (p10 > p90): sắp xếp lại theo từng giờ
        # (rearrangement) cho đúng thứ tự, rồi cắt về khoảng tỉ lệ lấp đầy [0, 1].
        prediction = np.clip(np.sort(self.forecaster.predict(history, horizon), axis=0), 0.0, 1.0)
        return (prediction[quantiles.index(0.1)], prediction[quantiles.index(0.5)],
                prediction[quantiles.index(0.9)])

    def _model_version(self) -> AIModelVersion:
        name, revision = self.forecaster.model_name, self.forecaster.revision
        current = self.repository.find_active_model(AiCapability.DEMAND_FORECAST)
        if current and current.model_name == name and current.version == revision[:12]:
            return current
        model = AIModelVersion(
            model_version_id=uuid5(NAMESPACE_URL, f"goatsports:{AiCapability.DEMAND_FORECAST.value}:{name}:{revision}"),
            model_name=name,
            version=revision[:12],
            capability=AiCapability.DEMAND_FORECAST,
            # Mô hình huấn luyện sẵn (zero-shot), không học lại trên dữ liệu GoatSports: trained_at là lúc đưa vào
            # dùng (find_active_model chọn bản mới nhất theo trường này), training_cutoff để trống.
            trained_at=self.clock(),
            training_cutoff=None,
            metrics={"context_hours": float(HISTORY_HOURS), "horizon_hours": float(HORIZON_HOURS)},
            artifact_uri=f"https://huggingface.co/amazon/{name}/tree/{revision}",
            status=AiModelStatus.ACTIVE,
        )
        return self.repository.save_model_version(model)


def _utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
