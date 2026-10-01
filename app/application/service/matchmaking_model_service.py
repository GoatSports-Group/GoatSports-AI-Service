import logging

from app.application.dto.response.models import MatchmakingModelStatus
from app.application.port.persistence import AiDocumentRepository, MatchmakingSessionRepository
from app.application.service.matchmaking_model import (
    FEATURES,
    MIN_PER_CLASS,
    MIN_SAMPLES,
    TrainingOutcome,
    importance,
    is_learned,
    label,
    train,
)
from app.application.service.matchmaking_scoring import SCORE_WEIGHTS
from app.domain.enums import AiCapability

logger = logging.getLogger(__name__)

_RULE_KEYS = {
    "elo_score": "elo",
    "skill_score": "skill",
    "schedule_score": "schedule",
    "distance_score": "distance",
    "play_style_score": "play_style",
    "position_score": "position",
    "experience_score": "experience",
}


class MatchmakingModelService:
    """Huấn luyện lại mô hình xếp hạng từ kết quả kèo thật, và cho biết hệ thống đang dùng mô hình hay luật."""

    def __init__(
        self,
        session_repository: MatchmakingSessionRepository,
        document_repository: AiDocumentRepository,
        sample_limit: int = 5000,
    ) -> None:
        self.session_repository = session_repository
        self.document_repository = document_repository
        self.sample_limit = sample_limit

    def train(self) -> TrainingOutcome:
        outcome = train(self.session_repository.find_recent(self.sample_limit))
        if outcome.model:
            self.document_repository.save_model_version(outcome.model)
        logger.info(
            "Matchmaking model training: samples=%s positives=%s auc=%s -> %s",
            outcome.samples, outcome.positives, outcome.auc, outcome.reason,
        )
        return outcome

    def status(self) -> MatchmakingModelStatus:
        # ponytail: đếm lại trên tối đa `sample_limit` phiên mỗi lần gọi; thêm cache nếu trang được mở dày.
        labels = [label(session) for session in self.session_repository.find_recent(self.sample_limit)]
        samples = sum(1 for value in labels if value is not None)
        positives = sum(1 for value in labels if value == 1)
        active = self.document_repository.find_active_model(AiCapability.MATCHMAKING)
        if is_learned(active):
            return MatchmakingModelStatus(
                mode="LEARNED",
                model_name=active.model_name,
                version=active.version,
                trained_at=active.trained_at,
                auc=active.metrics.get("auc"),
                trained_samples=int(active.metrics.get("samples", 0)),
                samples=samples,
                positives=positives,
                min_samples=MIN_SAMPLES,
                min_per_class=MIN_PER_CLASS,
                weights=importance(active),
            )
        return MatchmakingModelStatus(
            mode="RULES",
            samples=samples,
            positives=positives,
            min_samples=MIN_SAMPLES,
            min_per_class=MIN_PER_CLASS,
            weights={name: round(100 * SCORE_WEIGHTS[_RULE_KEYS[name]], 1) for name in FEATURES},
        )
