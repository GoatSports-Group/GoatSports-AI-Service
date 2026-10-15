"""Đánh giá offline bộ xếp hạng tab Khám phá: giấu tương tác cuối của mỗi người, xem bài đó có lọt top K không.

    python -m scripts.eval_feed_ranking --synthetic        # dữ liệu giả có sở thích rõ, kiểm thuật toán
    python -m scripts.eval_feed_ranking --mongo            # hành vi thật trong user_interactions (chỉ đọc)

So với mốc "mới + nóng" (cách xếp cho người chưa có tương tác). Cá nhân hóa phải hơn mốc này mới đáng dùng.
Chỉ đo tín hiệu hành vi; ứng viên là mọi bài có trong dữ liệu, đăng cùng lúc, nên độ mới không phân biệt được bài nào.
"""

import argparse
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from app.application.dto.request.models import PostCandidate
from app.application.service.post_ranking_service import PostRankingService
from app.domain.entity.models import UserInteraction
from app.domain.enums import InteractionType, RecommendationTargetType, SportType

NOW = datetime.now(timezone.utc)


class Repository:
    """Repository chỉ đọc trong bộ nhớ; người được đánh giá bị giấu tương tác cuối."""

    def __init__(self, interactions: list[UserInteraction]):
        self.interactions = interactions
        self.disabled: set[UUID] = set()
        self.model = None

    def find_active_model(self, _):
        return self.model

    def save_model_version(self, model):
        self.model = model
        return model

    def is_personalization_enabled(self, user_id):
        return user_id not in self.disabled

    def find_interactions(self, user_id=None, target_type=None, limit=20_000, since=None):
        return [
            item for item in self.interactions
            if (user_id is None or item.user_id == user_id)
            and (target_type is None or item.target_type == target_type)
            and (since is None or item.occurred_at >= since)
        ][:limit]


def synthetic(users: int = 120, authors: int = 30, posts_per_author: int = 6, seed: int = 7) -> list[UserInteraction]:
    """Mỗi người thích một môn và vài tác giả quen; 80% tương tác theo sở thích, 20% ngẫu nhiên."""
    rng = random.Random(seed)
    sports = list(SportType)
    author_ids = [uuid4() for _ in range(authors)]
    author_sport = {author: rng.choice(sports) for author in author_ids}
    posts = [(uuid4(), author) for author in author_ids for _ in range(posts_per_author)]
    interactions = []
    for _ in range(users):
        user = uuid4()
        sport = rng.choice(sports)
        liked_authors = rng.sample([a for a in author_ids if author_sport[a] == sport] or author_ids, k=2)
        for step in range(rng.randint(4, 9)):
            if rng.random() < 0.8:
                pool = [post for post in posts if post[1] in liked_authors]
            else:
                pool = posts
            post, author = rng.choice(pool)
            interactions.append(UserInteraction(
                user_id=user, target_id=post, target_type=RecommendationTargetType.POST,
                interaction_type=rng.choice([InteractionType.LIKE, InteractionType.COMMENT, InteractionType.SAVE]),
                sport_type=author_sport[author], metadata={"authorId": str(author)},
                occurred_at=NOW - timedelta(days=30) + timedelta(hours=step * 6),
            ))
    return interactions


def from_mongo() -> list[UserInteraction]:
    from app.infrastructure.config.database import database_clients
    from app.infrastructure.persistence.repository import MongoAiDocumentRepository

    database_clients.connect()
    try:
        return MongoAiDocumentRepository(database_clients.mongo).find_interactions(
            target_type=RecommendationTargetType.POST)
    finally:
        database_clients.close()


def evaluate(interactions: list[UserInteraction], k: int) -> None:
    by_user: dict[UUID, list[UserInteraction]] = defaultdict(list)
    for item in interactions:
        by_user[item.user_id].append(item)
    posts: dict[UUID, tuple[UUID, SportType | None]] = {}
    for item in interactions:
        author = item.metadata.get("authorId")
        if author:
            posts[item.target_id] = (UUID(str(author)), item.sport_type)
    popularity = Counter(item.target_id for item in interactions)

    hits = {"personal": 0, "baseline": 0}
    evaluated = 0
    for user, history in by_user.items():
        history.sort(key=lambda item: item.occurred_at)
        held_out = history[-1]
        seen = {item.target_id for item in history[:-1]}
        if len(history) < 3 or held_out.target_id in seen or held_out.target_id not in posts:
            continue
        evaluated += 1
        visible = [item for item in interactions if not (item.user_id == user and item is held_out)]
        candidates = [
            PostCandidate(post_id=post, author_id=author, sport=sport,
                          like_count=popularity[post] - (1 if post == held_out.target_id else 0),
                          published_at=NOW - timedelta(days=1))
            for post, (author, sport) in posts.items() if post not in seen
        ]
        for mode in hits:
            repository = Repository(visible)
            if mode == "baseline":
                repository.disabled.add(user)
            scores = PostRankingService(repository, snapshot_seconds=0, clock=lambda: NOW).rank(user, candidates)
            top = [item.post_id for item in sorted(scores, key=lambda item: item.score, reverse=True)[:k]]
            hits[mode] += held_out.target_id in top

    if not evaluated:
        print("Chưa đủ dữ liệu: cần người có ít nhất 3 tương tác với bài, tương tác cuối là bài mới với họ.")
        return
    print(f"Người được đánh giá: {evaluated} | Ứng viên mỗi người: ~{len(posts)} bài")
    for mode, label in (("personal", "Cá nhân hóa"), ("baseline", "Mới + nóng")):
        print(f"  {label:<12} Hit@{k}: {hits[mode] / evaluated:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true")
    source.add_argument("--mongo", action="store_true")
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # console Windows mặc định cp1252
    evaluate(synthetic() if args.synthetic else from_mongo(), args.k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
