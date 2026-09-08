from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import datetime, timezone
from uuid import UUID

from pydantic import TypeAdapter
from redis import Redis
from redis.lock import Lock

from app.domain.entity.models import (
    ClubMatchProfile,
    MatchmakingQueueEntry,
    MatchParticipantProfile,
    PlayerProfile,
)
from app.domain.enums import QueueStatus

participant_profile_adapter = TypeAdapter(PlayerProfile | ClubMatchProfile)


class RedisMatchmakingQueueRepository:
    QUEUE_KEY = "ai:matchmaking:queue"

    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    @staticmethod
    def _entry_key(entry_id: UUID) -> str:
        return f"ai:matchmaking:entry:{entry_id}"

    @staticmethod
    def _profile_key(profile_id: UUID) -> str:
        return f"ai:matchmaking:profile:{profile_id}"

    @staticmethod
    def _participant_key(participant_id: UUID) -> str:
        return f"ai:matchmaking:participant:{participant_id}"

    def lock(self) -> AbstractContextManager[Lock]:
        return self.redis.lock(
            "ai:matchmaking:lock",
            timeout=10,
            blocking_timeout=5,
        )

    def save(
        self,
        entry: MatchmakingQueueEntry,
        profile: MatchParticipantProfile,
    ) -> MatchmakingQueueEntry:
        ttl = self._ttl(entry.expires_at)
        entry_key = self._entry_key(entry.queue_entry_id)
        profile_key = self._profile_key(profile.participant_profile_id)
        participant_key = self._participant_key(profile.participant_id)
        with self.redis.pipeline(transaction=True) as pipe:
            pipe.set(entry_key, entry.model_dump_json(by_alias=True), ex=ttl)
            pipe.set(profile_key, profile.model_dump_json(by_alias=True), ex=ttl)
            pipe.set(participant_key, str(entry.queue_entry_id), ex=ttl)
            if entry.status == QueueStatus.QUEUED:
                pipe.zadd(self.QUEUE_KEY, {str(entry.queue_entry_id): entry.queued_at.timestamp()})
            else:
                pipe.zrem(self.QUEUE_KEY, str(entry.queue_entry_id))
            pipe.execute()
        return entry

    def find_by_participant(self, participant_id: UUID) -> MatchmakingQueueEntry | None:
        entry_id = self.redis.get(self._participant_key(participant_id))
        if not entry_id:
            return None
        return self.find_entry(UUID(entry_id))

    def find_entry(self, entry_id: UUID) -> MatchmakingQueueEntry | None:
        payload = self.redis.get(self._entry_key(entry_id))
        return MatchmakingQueueEntry.model_validate_json(payload) if payload else None

    def find_profile(self, profile_id: UUID) -> MatchParticipantProfile | None:
        payload = self.redis.get(self._profile_key(profile_id))
        return participant_profile_adapter.validate_json(payload) if payload else None

    def queued_entries(self, limit: int) -> list[MatchmakingQueueEntry]:
        self._purge_missing_entries()
        entry_ids = self.redis.zrange(self.QUEUE_KEY, 0, max(limit - 1, 0))
        entries: list[MatchmakingQueueEntry] = []
        for raw_id in entry_ids:
            entry = self.find_entry(UUID(raw_id))
            if entry and entry.status == QueueStatus.QUEUED:
                entries.append(entry)
        return entries

    def queue_size(self) -> int:
        self._purge_missing_entries()
        return int(self.redis.zcard(self.QUEUE_KEY))

    def remove_from_active_queue(self, entry_id: UUID) -> None:
        self.redis.zrem(self.QUEUE_KEY, str(entry_id))

    def _purge_missing_entries(self) -> None:
        for raw_id in self.redis.zrange(self.QUEUE_KEY, 0, -1):
            if not self.redis.exists(self._entry_key(UUID(raw_id))):
                self.redis.zrem(self.QUEUE_KEY, raw_id)

    @staticmethod
    def _ttl(expires_at: datetime) -> int:
        now = datetime.now(timezone.utc)
        normalized = expires_at if expires_at.tzinfo else expires_at.replace(tzinfo=timezone.utc)
        return max(1, int((normalized - now).total_seconds()))


# Compatibility name for callers migrating from the original flat package.
MatchmakingQueueRepository = RedisMatchmakingQueueRepository