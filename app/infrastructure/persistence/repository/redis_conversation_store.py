from __future__ import annotations

import json

from redis import Redis

_TTL_SECONDS = 30 * 60
_MAX_TURNS = 6  # 6 lượt hỏi-đáp gần nhất là đủ để hiểu câu nối tiếp mà prompt không phình


class RedisConversationStore:
    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    def load(self, key: str) -> list[dict[str, str]]:
        raw = self.redis.lrange(f"ai:assistant:history:{key}", 0, -1)
        messages = []
        for item in raw:
            try:
                messages.append(json.loads(item))
            except ValueError:
                continue
        return messages

    def append(self, key: str, question: str, answer: str) -> None:
        name = f"ai:assistant:history:{key}"
        pipeline = self.redis.pipeline()
        pipeline.rpush(
            name,
            json.dumps({"role": "user", "content": question}, ensure_ascii=False),
            json.dumps({"role": "assistant", "content": answer}, ensure_ascii=False),
        )
        pipeline.ltrim(name, -2 * _MAX_TURNS, -1)
        pipeline.expire(name, _TTL_SECONDS)
        pipeline.execute()

    def allow(self, key: str, limit_per_minute: int) -> bool:
        name = f"ai:assistant:rate:{key}"
        count = int(self.redis.incr(name))
        if count == 1:
            self.redis.expire(name, 60)
        return count <= limit_per_minute
