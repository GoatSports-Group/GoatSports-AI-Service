from __future__ import annotations

from typing import Protocol

import numpy as np


class DemandForecaster(Protocol):
    """Mô hình dự báo chuỗi thời gian: lịch sử (NaN = chưa có dữ liệu) → (số quantile, horizon)."""

    @property
    def quantiles(self) -> tuple[float, ...]: ...

    @property
    def prediction_length(self) -> int:
        """Số giờ một lần chạy mô hình; xa hơn là dự báo chạy nối."""
        ...

    @property
    def model_name(self) -> str: ...

    @property
    def revision(self) -> str: ...

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray: ...
