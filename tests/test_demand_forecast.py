import unittest
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from app.application.dto.response.models import DemandHistorySnapshot
from app.application.exception import (
    CapabilityUnavailableException,
    InsufficientForecastDataException,
)
from app.application.service.demand_forecast_service import HISTORY_HOURS, DemandForecastService
from app.infrastructure.forecast.chronos_onnx import ChronosBoltConfig, ChronosBoltForecaster

QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MODEL = Path(__file__).resolve().parents[1] / "models" / "chronos-bolt-tiny.onnx"


def weekly_pattern(hours: int) -> np.ndarray:
    index = np.arange(hours)
    return (0.1 + 0.7 * np.exp(-((index % 24 - 19) ** 2) / 6)).astype(np.float32)


class RepeatLastWeekForecaster:
    """Đoán "giống tuần trước" với khoảng ±0.1: đủ để kiểm tra phần service quanh mô hình."""

    quantiles = QUANTILES
    prediction_length = 64
    model_name = "chronos-bolt-tiny"
    revision = "a0e552de83495b5c28c14c71c374f3e33280b340"

    def __init__(self):
        self.calls = 0

    def predict(self, history, horizon):
        self.calls += 1
        last_week = np.nan_to_num(history[-168:], nan=0.0)
        median = np.resize(last_week, horizon)
        return np.stack([median + (q - 0.5) for q in QUANTILES])


class FakeVenueClient:
    def __init__(self, occupancy):
        self.occupancy = occupancy
        self.calls = 0

    def owner_demand_history(self, owner_id, venue_id, hours):
        self.calls += 1
        assert hours == HISTORY_HOURS
        return DemandHistorySnapshot(
            venue_id=venue_id, venue_name="GOAT", start=datetime(2026, 7, 1, 0, 0), court_count=4,
            open_time=time(6, 0), close_time=time(22, 0), occupancy=self.occupancy,
        )


class FakeRepository:
    def __init__(self):
        self.forecasts = {}
        self.models = []

    def find_venue_forecast(self, venue_id):
        return self.forecasts.get(venue_id)

    def save_venue_forecast(self, forecast):
        self.forecasts[forecast.venue_id] = forecast
        return forecast

    def find_active_model(self, _):
        return self.models[-1] if self.models else None

    def save_model_version(self, model):
        self.models.append(model)
        return model


class DemandForecastServiceTest(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 9, 5, 0, tzinfo=timezone.utc)
        self.repository = FakeRepository()
        self.forecaster = RepeatLastWeekForecaster()
        history = [None] * 100 + weekly_pattern(HISTORY_HOURS - 100).tolist()
        self.client = FakeVenueClient(history)
        self.service = DemandForecastService(self.client, self.repository, self.forecaster, clock=lambda: self.now)

    def test_forecast_covers_seven_days_with_ordered_clipped_quantiles(self):
        result = self.service.forecast(uuid4(), uuid4())

        self.assertEqual(168, len(result.hours))
        self.assertEqual(datetime(2026, 7, 1) + timedelta(hours=HISTORY_HOURS), result.forecast_start)
        self.assertEqual(HISTORY_HOURS - 100, result.history_hours)
        for hour in result.hours:
            self.assertTrue(0 <= hour.lower <= hour.median <= hour.upper <= 1)
        # 64 giờ đầu là một lần chạy mô hình; phần sau là chạy nối.
        self.assertFalse(result.hours[63].extended)
        self.assertTrue(result.hours[64].extended)
        self.assertEqual("chronos-bolt-tiny", result.model_name)
        self.assertEqual(1, len(self.repository.models))

    def test_backtest_compares_last_week_against_repeat_last_week(self):
        result = self.service.forecast(uuid4(), uuid4())

        # Chuỗi lặp đúng mỗi 168 giờ: đoán "giống tuần trước" trúng, nên sai số mô hình (cũng lặp tuần) = baseline = 0.
        self.assertIsNotNone(result.backtest)
        self.assertEqual(168, result.backtest.hours)
        self.assertAlmostEqual(0.0, result.backtest.model_mae, places=4)
        self.assertAlmostEqual(result.backtest.baseline_mae, result.backtest.model_mae, places=4)
        self.assertEqual(1.0, result.backtest.coverage)

    def test_recent_forecast_is_reused_for_the_same_owner_only(self):
        owner, venue = uuid4(), uuid4()
        first = self.service.forecast(owner, venue)
        self.now += timedelta(minutes=10)
        self.assertEqual(first.forecast_id, self.service.forecast(owner, venue).forecast_id)
        self.assertEqual(1, self.client.calls)

        # Người khác (venue-service sẽ kiểm quyền) không được nhận bản đệm của chủ sân.
        self.service.forecast(uuid4(), venue)
        self.assertEqual(2, self.client.calls)

        self.now += timedelta(minutes=20)
        self.assertNotEqual(first.forecast_id, self.service.forecast(owner, venue).forecast_id)

    def test_less_than_one_day_of_history_is_rejected(self):
        self.client.occupancy = [None] * (HISTORY_HOURS - 10) + [0.5] * 10
        with self.assertRaises(InsufficientForecastDataException):
            self.service.forecast(uuid4(), uuid4())

    def test_missing_model_reports_capability_unavailable(self):
        service = DemandForecastService(self.client, self.repository, None, clock=lambda: self.now)
        with self.assertRaises(CapabilityUnavailableException):
            service.forecast(uuid4(), uuid4())


class ZeroSession:
    """Mạng trả 0 ở thang chuẩn hóa → dự báo = trung bình lịch sử: kiểm tra chuẩn hóa / chia đoạn / chạy nối."""

    def __init__(self):
        self.inputs = []

    def run(self, _, feeds):
        self.inputs.append(feeds)
        return [np.zeros((feeds["patched_context"].shape[0], len(QUANTILES), 64), np.float32)]


class ChronosOnnxPreprocessingTest(unittest.TestCase):
    def setUp(self):
        config = ChronosBoltConfig(context_length=2048, patch_size=16, prediction_length=64,
                                   quantiles=QUANTILES, use_arcsinh=False, eps=1e-5)
        self.session = ZeroSession()
        self.forecaster = ChronosBoltForecaster("unused.onnx", config=config, session=self.session)

    def test_long_horizon_rolls_out_each_quantile_and_returns_history_mean(self):
        history = np.array([np.nan] * 20 + [0.2, 0.4] * 30, dtype=np.float32)

        prediction = self.forecaster.predict(history, 168)

        self.assertEqual((9, 168), prediction.shape)
        np.testing.assert_allclose(prediction, 0.3, atol=1e-5)
        # Lần 1: một chuỗi; lần 2, 3: mỗi mức quantile một chuỗi (9).
        self.assertEqual([1, 9, 9], [feeds["patched_context"].shape[0] for feeds in self.session.inputs])

    def test_missing_hours_are_masked_patch_by_patch(self):
        history = np.full(40, np.nan, dtype=np.float32)
        history[-20:] = 0.5

        self.forecaster.predict(history, 64)

        feeds = self.session.inputs[0]
        self.assertEqual((1, 128, 32), feeds["patched_context"].shape)
        # 20 giờ cuối có dữ liệu: chỉ 2 đoạn 16 giờ cuối được chú ý; giá trị chưa có dữ liệu bị đặt 0.
        np.testing.assert_array_equal(feeds["attention_mask"][0, -3:], [0.0, 1.0, 1.0])
        self.assertEqual(0.0, float(np.abs(feeds["patched_context"][0, :126]).sum()))


@unittest.skipUnless(MODEL.is_file(), "chưa có models/chronos-bolt-tiny.onnx (chạy scripts/export_chronos_bolt.py)")
class ChronosOnnxModelTest(unittest.TestCase):
    def test_real_model_follows_the_daily_peak(self):
        forecaster = ChronosBoltForecaster(MODEL)
        history = weekly_pattern(24 * 7 * 6)

        prediction = forecaster.predict(history, 168)
        median = prediction[QUANTILES.index(0.5)]

        self.assertTrue(np.isfinite(prediction).all())
        # Lịch sử kết thúc ở 0 giờ: giờ cao điểm 19h của ngày mai phải cao hơn hẳn 4h sáng.
        self.assertGreater(median[19], median[4] + 0.3)

    def test_service_orders_crossing_quantiles(self):
        service = DemandForecastService(None, None, ChronosBoltForecaster(MODEL))
        lower, median, upper = service._predict(weekly_pattern(24 * 7 * 6), 168)
        self.assertTrue(((lower <= median) & (median <= upper)).all())


if __name__ == "__main__":
    unittest.main()
