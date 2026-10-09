"""Chronos-Bolt (Amazon, Apache-2.0) chạy bằng onnxruntime, không cần PyTorch.

File ONNX chỉ chứa phần mạng nơ-ron (embedding → encoder T5 → decoder → đầu ra quantile). Phần còn lại của
``ChronosBoltModelForForecasting.forward`` được viết lại ở đây bằng numpy, đúng như mã gốc chronos-forecasting 2.3.2:

- chuẩn hóa từng chuỗi (trừ trung bình, chia độ lệch chuẩn, bỏ qua NaN), chia đoạn 16 giờ, mặt nạ "có dữ liệu";
- đổi kết quả về thang gốc;
- dự báo dài hơn 64 giờ: nối từng mức quantile vào lịch sử, dự báo tiếp, rồi gom 9 x 9 kết quả về 9 mức quantile.

``scripts/export_chronos_bolt.py`` so module này với pipeline PyTorch gốc lúc build image (sai lệch phải < 1e-4).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChronosBoltConfig:
    context_length: int
    patch_size: int
    prediction_length: int
    quantiles: tuple[float, ...]
    use_arcsinh: bool
    eps: float
    model_name: str = "chronos-bolt-tiny"
    revision: str = ""

    @classmethod
    def load(cls, path: Path) -> ChronosBoltConfig:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            context_length=int(raw["context_length"]),
            patch_size=int(raw["patch_size"]),
            prediction_length=int(raw["prediction_length"]),
            quantiles=tuple(float(q) for q in raw["quantiles"]),
            use_arcsinh=bool(raw["use_arcsinh"]),
            eps=float(raw["eps"]),
            model_name=str(raw.get("model_name", "chronos-bolt-tiny")),
            revision=str(raw.get("revision", "")),
        )


def load_forecaster(model_path: str) -> ChronosBoltForecaster | None:
    """Thiếu file mô hình (máy dev chưa chạy script xuất) → None: endpoint báo "chưa cài", service khác vẫn chạy."""
    path = Path(model_path)
    if not path.is_file() or not path.with_suffix(".json").is_file():
        logger.warning("Demand forecast model not found at %s; forecasting is disabled", path)
        return None
    return ChronosBoltForecaster(path)


class ChronosBoltForecaster:
    """``predict(history, horizon)`` → mảng (số quantile, horizon). ``history`` dùng NaN cho giờ chưa có dữ liệu."""

    def __init__(self, model_path: str | Path, config: ChronosBoltConfig | None = None, session=None) -> None:
        model_path = Path(model_path)
        self.config = config or ChronosBoltConfig.load(model_path.with_suffix(".json"))
        if session is None:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        self.session = session

    @property
    def quantiles(self) -> tuple[float, ...]:
        return self.config.quantiles

    @property
    def prediction_length(self) -> int:
        return self.config.prediction_length

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def revision(self) -> str:
        return self.config.revision

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        cfg = self.config
        context = np.asarray(history, dtype=np.float32)[None, -cfg.context_length:]
        prediction = self._run(context)  # (1, nq, H)
        predictions = [prediction]
        remaining = horizon - prediction.shape[-1]
        if remaining > 0:
            context = np.repeat(context[:, None, :], len(cfg.quantiles), axis=1)  # (1, nq, len)
        while remaining > 0:
            context = np.concatenate([context, prediction], axis=-1)[..., -cfg.context_length:]
            batch, paths, length = context.shape
            samples = self._run(context.reshape(batch * paths, length)).reshape(batch, paths * len(cfg.quantiles), -1)
            prediction = np.quantile(samples, cfg.quantiles, axis=1).transpose(1, 0, 2).astype(np.float32)
            predictions.append(prediction)
            remaining -= prediction.shape[-1]
        return np.concatenate(predictions, axis=-1)[0, :, :horizon]

    def _run(self, contexts: np.ndarray) -> np.ndarray:
        """contexts (B, len ≤ L) → dự báo (B, nq, H) theo thang gốc."""
        cfg = self.config
        batch = contexts.shape[0]
        padded = np.full((batch, cfg.context_length), np.nan, dtype=np.float32)
        padded[:, cfg.context_length - contexts.shape[1]:] = contexts
        patches = cfg.context_length // cfg.patch_size

        mask = ~np.isnan(padded)
        counts = mask.sum(axis=1, keepdims=True)
        safe = np.where(mask, padded, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            loc = np.where(counts > 0, safe.sum(axis=1, keepdims=True) / counts, 0.0)
            variance = np.where(mask, (padded - loc) ** 2, 0.0).sum(axis=1, keepdims=True) / counts
            scale = np.where(counts > 0, np.sqrt(variance), 1.0)
        scale = np.where(scale == 0, cfg.eps, scale)
        scaled = (padded - loc) / scale
        if cfg.use_arcsinh:
            scaled = np.arcsinh(scaled)

        patch_mask = mask.reshape(batch, patches, cfg.patch_size).astype(np.float32)
        patch_values = np.where(patch_mask > 0, scaled.reshape(batch, patches, cfg.patch_size), 0.0)
        patched = np.concatenate([patch_values, patch_mask], axis=-1).astype(np.float32)
        attention = (patch_mask.sum(axis=-1) > 0).astype(np.float32)

        normalized = self.session.run(None, {"patched_context": patched, "attention_mask": attention})[0]
        values = np.sinh(normalized) if cfg.use_arcsinh else normalized
        return (values * scale[:, :, None] + loc[:, :, None]).astype(np.float32)
