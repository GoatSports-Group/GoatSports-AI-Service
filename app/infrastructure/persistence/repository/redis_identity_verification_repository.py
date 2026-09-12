from uuid import UUID

from redis import Redis

from app.domain.entity.models import IdentityVerification


class RedisIdentityVerificationRepository:
    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    @staticmethod
    def _key(verification_id: UUID) -> str:
        return f"ai:identity-verification:{verification_id}"

    def save(
        self,
        verification: IdentityVerification,
        ttl_seconds: int,
    ) -> IdentityVerification:
        self.redis.set(
            self._key(verification.verification_id),
            verification.model_dump_json(by_alias=True),
            ex=ttl_seconds,
        )
        return verification

    def find_by_id(self, verification_id: UUID) -> IdentityVerification | None:
        payload = self.redis.get(self._key(verification_id))
        return IdentityVerification.model_validate_json(payload) if payload else None
