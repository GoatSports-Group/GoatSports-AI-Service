"""Mô hình học máy xếp hạng đối thủ: logistic regression dự đoán "kèo này có thành công không".

Đầu vào là 7 điểm thành phần đã tính cho mỗi cặp (ELO, trình độ, khung giờ, khoảng cách, phong cách, vị trí,
kinh nghiệm). Nhãn lấy từ kết quả thật của các phiên đã kết thúc. Khi chưa đủ dữ liệu, hệ thống vẫn xếp hạng
theo trọng số viết tay (``SCORE_WEIGHTS``); mô hình chỉ thay trọng số đó khi đã qua kiểm chứng chéo.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from app.domain.entity.models import AIModelVersion, MatchmakingSession, MatchScoreBreakdown
from app.domain.enums import AiCapability, AiModelStatus, MatchSessionStatus

LEARNED_MODEL_NAME = "matchmaking-logistic-regression"
FEATURES = (
    "elo_score",
    "skill_score",
    "schedule_score",
    "distance_score",
    "play_style_score",
    "position_score",
    "experience_score",
)
# Dưới các ngưỡng này mô hình học từ nhiễu nhiều hơn từ dữ liệu, nên vẫn dùng trọng số viết tay.
MIN_SAMPLES = 40
MIN_PER_CLASS = 10
MIN_AUC = 0.55
FOLDS = 5

_FAILED = {MatchSessionStatus.REJECTED, MatchSessionStatus.EXPIRED, MatchSessionStatus.CANCELLED}


def features(breakdown: MatchScoreBreakdown) -> np.ndarray:
    return np.array([getattr(breakdown, name) / 100.0 for name in FEATURES], dtype=float)


def label(session: MatchmakingSession) -> int | None:
    """1: kèo diễn ra và không ai chấm trận tệ; 0: bị từ chối, hủy, hết hạn hoặc bị chấm ≤ 2 sao; None: chưa xong."""
    if session.score_breakdown is None:
        return None
    if session.status == MatchSessionStatus.COMPLETED:
        worst = min(
            (min(item.rating, item.fair_play_rating) for item in session.feedback),
            default=5,
        )
        return 0 if worst <= 2 else 1
    return 0 if session.status in _FAILED else None


def fit(x: np.ndarray, y: np.ndarray, l2: float = 0.05, epochs: int = 3000, lr: float = 0.5) -> tuple[np.ndarray, float]:
    """Logistic regression có chuẩn hóa L2, huấn luyện bằng gradient descent (dữ liệu nhỏ, không cần thư viện)."""
    weights = np.zeros(x.shape[1])
    bias = 0.0
    n = len(y)
    for _ in range(epochs):
        p = _sigmoid(x @ weights + bias)
        error = p - y
        weights -= lr * (x.T @ error / n + l2 * weights)
        bias -= lr * float(error.mean())
    return weights, bias


def auc(y: np.ndarray, scores: np.ndarray) -> float:
    """AUC = xác suất một kèo thành công được chấm cao hơn một kèo thất bại (thống kê Mann–Whitney)."""
    positives = scores[y == 1]
    negatives = scores[y == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.5
    greater = (positives[:, None] > negatives[None, :]).sum()
    ties = (positives[:, None] == negatives[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(positives) * len(negatives)))


@dataclass(frozen=True)
class TrainingOutcome:
    samples: int
    positives: int
    auc: float | None
    model: AIModelVersion | None
    reason: str


def train(sessions: list[MatchmakingSession], now: datetime | None = None) -> TrainingOutcome:
    rows = [(features(s.score_breakdown), label(s), s.session_id) for s in sessions]
    rows = [(f, y, sid) for f, y, sid in rows if y is not None]
    samples = len(rows)
    positives = sum(y for _, y, _ in rows)
    negatives = samples - positives
    if samples < MIN_SAMPLES or positives < MIN_PER_CLASS or negatives < MIN_PER_CLASS:
        return TrainingOutcome(
            samples, positives, None, None,
            f"Cần ít nhất {MIN_SAMPLES} kèo có kết quả, mỗi loại (thành công / thất bại) ≥ {MIN_PER_CLASS}.",
        )
    x = np.vstack([f for f, _, _ in rows])
    y = np.array([y for _, y, _ in rows], dtype=float)

    # Kiểm chứng chéo: mỗi kèo được chấm bởi mô hình không nhìn thấy nó lúc học. Chia theo mã phiên cho ổn định.
    folds = np.array([sid.int % FOLDS for _, _, sid in rows])
    out_of_fold = np.zeros(samples)
    for fold in range(FOLDS):
        held = folds == fold
        if held.all() or not held.any():
            continue
        w, b = fit(x[~held], y[~held])
        out_of_fold[held] = _sigmoid(x[held] @ w + b)
    cross_auc = auc(y, out_of_fold)
    if cross_auc < MIN_AUC:
        return TrainingOutcome(
            samples, positives, cross_auc, None,
            f"AUC kiểm chứng chéo {cross_auc:.2f} < {MIN_AUC}: mô hình chưa tốt hơn đoán mò đủ để dùng.",
        )

    weights, bias = fit(x, y)
    trained_at = now or datetime.now(timezone.utc)
    metrics = {"auc": round(cross_auc, 4), "samples": float(samples), "positives": float(positives), "bias": bias}
    metrics.update({f"w_{name}": float(value) for name, value in zip(FEATURES, weights)})
    model = AIModelVersion(
        model_name=LEARNED_MODEL_NAME,
        version=trained_at.strftime("%Y%m%d%H%M"),
        capability=AiCapability.MATCHMAKING,
        trained_at=trained_at,
        training_cutoff=trained_at,
        metrics=metrics,
        artifact_uri="mongodb://ai_model_versions (trọng số trong metrics)",
        status=AiModelStatus.ACTIVE,
    )
    return TrainingOutcome(samples, positives, cross_auc, model, "Đã huấn luyện và bật mô hình mới.")


def is_learned(model: AIModelVersion | None) -> bool:
    return bool(model and model.model_name == LEARNED_MODEL_NAME and "bias" in model.metrics)


def success_probability(model: AIModelVersion, breakdown: MatchScoreBreakdown) -> float:
    """Xác suất kèo thành công (0–1) theo mô hình đã học."""
    weights = np.array([model.metrics.get(f"w_{name}", 0.0) for name in FEATURES])
    return float(_sigmoid(features(breakdown) @ weights + model.metrics.get("bias", 0.0)))


def importance(model: AIModelVersion) -> dict[str, float]:
    """Tỷ trọng ảnh hưởng (%) của từng tiêu chí theo trọng số đã học; trọng số âm (làm kèo dễ hỏng) tính là 0."""
    positive = {name: max(0.0, model.metrics.get(f"w_{name}", 0.0)) for name in FEATURES}
    total = sum(positive.values())
    if total <= 0:
        return {name: 0.0 for name in FEATURES}
    return {name: round(100.0 * value / total, 1) for name, value in positive.items()}


def _sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))
