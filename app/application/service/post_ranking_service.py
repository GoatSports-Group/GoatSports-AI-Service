"""Xếp hạng bài viết cho tab "Khám phá": social-service chọn ứng viên (bài người xem được thấy, mới nhất), ai-service
chấm điểm, social-service sắp lại. Ai-service không tự chọn bài nên không bao giờ lộ bài người xem không được thấy.

Điểm (0..1) gồm các tín hiệu giải thích được:
- tác giả quen: người xem từng thích / bình luận / lưu bài của tác giả này;
- môn quen: tỉ trọng môn của bài trong các tương tác của người xem;
- người giống bạn: những người có tương tác giống người xem đã tương tác với bài này (collaborative filtering);
- độ nóng: lượt thích và bình luận của bài;
- độ mới: bài mới được ưu tiên, giảm dần theo giờ.
Người chưa có tương tác (hoặc đã tắt cá nhân hóa) chỉ còn độ nóng + độ mới.
"""

import math
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from uuid import UUID

from app.application.dto.request.models import PostCandidate
from app.application.dto.response.models import PostScore
from app.application.port.persistence.repositories import AiDocumentRepository
from app.application.service.model_registry import get_or_create_active_model
from app.domain.entity.models import UserInteraction
from app.domain.enums import AiCapability, InteractionType, RecommendationTargetType

INTERACTION_WEIGHTS = {
    InteractionType.VIEW: 1.0,
    InteractionType.LIKE: 3.0,
    InteractionType.SAVE: 4.0,
    InteractionType.COMMENT: 4.0,
    InteractionType.SHARE: 5.0,
    InteractionType.BOOKING: 6.0,
}
HALF_LIFE_DAYS = 90.0
WINDOW_DAYS = 180
FRESHNESS_HOURS = 48.0

PERSONAL_WEIGHTS = {"author": 0.30, "sport": 0.15, "neighbors": 0.15, "popularity": 0.10, "freshness": 0.30}
COLD_WEIGHTS = {"popularity": 0.35, "freshness": 0.65}


def effective_weight(interaction: UserInteraction, now: datetime) -> float:
    occurred_at = interaction.occurred_at
    if occurred_at.tzinfo is None:
        occurred_at = occurred_at.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (now - occurred_at).total_seconds() / 86_400)
    base = interaction.weight or INTERACTION_WEIGHTS[interaction.interaction_type]
    return base * math.exp(-age_days * math.log(2) / HALF_LIFE_DAYS)


class PostRankingService:
    def __init__(self, repository: AiDocumentRepository, snapshot_seconds: float = 300.0, clock=None) -> None:
        self.repository = repository
        self.snapshot_seconds = snapshot_seconds
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()
        self._snapshot: dict[UUID, dict[UUID, float]] = {}
        self._snapshot_at = -math.inf

    def rank(self, user_id: UUID, candidates: list[PostCandidate]) -> list[PostScore]:
        get_or_create_active_model(self.repository, AiCapability.FEED_RANKING)
        now = self.clock()
        personal = self.repository.is_personalization_enabled(user_id)
        history = (
            self.repository.find_interactions(user_id=user_id, target_type=RecommendationTargetType.POST,
                                              since=now - timedelta(days=WINDOW_DAYS))
            if personal else []
        )
        authors: dict[str, float] = defaultdict(float)
        sports: dict[str, float] = defaultdict(float)
        for interaction in history:
            weight = effective_weight(interaction, now)
            if interaction.metadata.get("authorId"):
                authors[str(interaction.metadata["authorId"])] += weight
            if interaction.sport_type:
                sports[interaction.sport_type.value] += weight
        sport_total = sum(sports.values())
        neighbors = self._neighbor_scores(user_id, {item.post_id for item in candidates}) if history else {}

        scores = []
        for candidate in candidates:
            signals = {
                "popularity": min(1.0, math.log1p(candidate.like_count + 2 * candidate.comment_count) / math.log1p(60)),
                "freshness": _freshness(candidate.published_at, now),
            }
            if history:
                signals["author"] = 1 - math.exp(-authors.get(str(candidate.author_id), 0.0) / 6.0)
                signals["sport"] = (sports.get(candidate.sport.value, 0.0) / sport_total
                                    if candidate.sport and sport_total else 0.0)
                signals["neighbors"] = neighbors.get(candidate.post_id, 0.0)
            weights = PERSONAL_WEIGHTS if history else COLD_WEIGHTS
            scores.append(PostScore(
                post_id=candidate.post_id,
                score=round(sum(weights[name] * value for name, value in signals.items()), 6),
            ))
        return scores

    def _neighbor_scores(self, user_id: UUID, candidate_ids: set[UUID]) -> dict[UUID, float]:
        matrix = self._matrix()
        mine = matrix.get(user_id, {})
        if not mine:
            return {}
        raw: dict[UUID, float] = defaultdict(float)
        for other_id, vector in matrix.items():
            if other_id == user_id:
                continue
            similarity = _cosine(mine, vector)
            if similarity <= 0:
                continue
            for post_id in candidate_ids & vector.keys():
                raw[post_id] += similarity * vector[post_id]
        peak = max(raw.values(), default=0.0)
        return {post_id: value / peak for post_id, value in raw.items()} if peak else {}

    def _matrix(self) -> dict[UUID, dict[UUID, float]]:
        # ponytail: ma trận người × bài dựng lại 5 phút một lần trong bộ nhớ từ hành vi 180 ngày; cỡ vài trăm nghìn
        # hành vi trở lên thì chuyển sang job tính sẵn độ giống giữa người dùng và lưu Mongo.
        with self._lock:
            if time.monotonic() - self._snapshot_at < self.snapshot_seconds:
                return self._snapshot
            now = self.clock()
            matrix: dict[UUID, dict[UUID, float]] = defaultdict(lambda: defaultdict(float))
            for interaction in self.repository.find_interactions(target_type=RecommendationTargetType.POST,
                                                                 since=now - timedelta(days=WINDOW_DAYS)):
                matrix[interaction.user_id][interaction.target_id] += effective_weight(interaction, now)
            self._snapshot = {user: dict(vector) for user, vector in matrix.items()}
            self._snapshot_at = time.monotonic()
            return self._snapshot


def _freshness(published_at: datetime | None, now: datetime) -> float:
    if published_at is None:
        return 0.0
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    age_hours = max(0.0, (now - published_at).total_seconds() / 3600)
    return math.exp(-age_hours / FRESHNESS_HOURS)


def _cosine(first: dict[UUID, float], second: dict[UUID, float]) -> float:
    common = first.keys() & second.keys()
    if not common:
        return 0.0
    dot = sum(first[item] * second[item] for item in common)
    norm = math.sqrt(sum(v * v for v in first.values())) * math.sqrt(sum(v * v for v in second.values()))
    return dot / norm if norm else 0.0
