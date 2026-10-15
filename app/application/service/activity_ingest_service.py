"""Biến sự kiện hành vi (Kafka `user-activity-topic`) thành `user_interactions` cho gợi ý cá nhân hóa.

Hợp đồng sự kiện (JSON, khóa Kafka = userId để sự kiện của một người giữ đúng thứ tự):
    {"eventId", "eventType", "producer", "occurredAt", "userId", "payload": {...}}

- activity.venue.viewed.v1    payload: venueId, sportTypes[], minPrice, rating
- activity.booking.state.v1   payload: bookingId, venueId, sportType, status, totalPrice, rating
- activity.post.reaction.v1   payload: postId, authorId, sport, reaction (LIKE|SAVE), active
- activity.post.comment.v1    payload: postId, authorId, sport, commentId, active
- activity.post.shared.v1     payload: postId, authorId, sport

Mỗi hành vi có mã cố định suy ra từ nguồn (vd. một đơn đặt sân = một hành vi), nên nhận trùng (Kafka giao lại,
nạp bù chạy lại) chỉ ghi đè chứ không đếm hai lần. Bỏ thích, xóa bình luận, đơn bị hủy thì xóa hành vi tương ứng.
"""

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, NAMESPACE_URL, uuid5

from app.application.port.persistence.repositories import AiDocumentRepository
from app.domain.entity.models import UserInteraction
from app.domain.enums import InteractionType, RecommendationTargetType, SportType

logger = logging.getLogger(__name__)

VENUE_VIEWED = "activity.venue.viewed.v1"
BOOKING_STATE = "activity.booking.state.v1"
POST_REACTION = "activity.post.reaction.v1"
POST_COMMENT = "activity.post.comment.v1"
POST_SHARED = "activity.post.shared.v1"

# Đơn còn hiệu lực mới là tín hiệu "thích sân này"; hủy / hết hạn / không đến thì gỡ.
_BOOKED_STATUSES = {"CONFIRMED", "CHECKED_IN", "COMPLETED"}
_REACTIONS = {"LIKE": InteractionType.LIKE, "SAVE": InteractionType.SAVE}


class ActivityIngestService:
    def __init__(self, repository: AiDocumentRepository) -> None:
        self.repository = repository

    def handle(self, event: dict[str, Any]) -> None:
        """Bỏ qua (có log) sự kiện hỏng thay vì ném lỗi: một bản tin sai không được chặn cả hàng đợi."""
        try:
            user_id = UUID(str(event["userId"]))
            event_type = event["eventType"]
            payload = event.get("payload") or {}
            occurred_at = _timestamp(event.get("occurredAt"))
        except (KeyError, TypeError, ValueError):
            logger.warning("Skipping malformed activity event %s", event.get("eventId"))
            return
        if not self.repository.is_personalization_enabled(user_id):
            return
        try:
            self._apply(user_id, event_type, payload, occurred_at)
        except (KeyError, TypeError, ValueError):
            logger.warning("Skipping activity event %s with bad payload", event.get("eventId"))

    def personalization_enabled(self, user_id: UUID) -> bool:
        return self.repository.is_personalization_enabled(user_id)

    def set_personalization(self, user_id: UUID, enabled: bool) -> bool:
        """Tắt thì xóa luôn hành vi đã lưu và từ đó bỏ qua sự kiện mới của người này."""
        self.repository.set_personalization_enabled(user_id, enabled)
        if not enabled:
            self.repository.delete_user_interactions(user_id)
        return enabled

    def _apply(self, user_id: UUID, event_type: str, payload: dict[str, Any], occurred_at: datetime) -> None:
        if event_type == VENUE_VIEWED:
            venue_id = UUID(payload["venueId"])
            sports = [_sport(value) for value in payload.get("sportTypes") or []]
            # Một người xem một sân nhiều lần trong ngày chỉ tính một lượt.
            key = f"view:{user_id}:{venue_id}:{occurred_at.date().isoformat()}"
            self._save(key, user_id, venue_id, RecommendationTargetType.VENUE, InteractionType.VIEW,
                       next((sport for sport in sports if sport), None), occurred_at,
                       _venue_metadata(payload))
        elif event_type == BOOKING_STATE:
            key = f"booking:{UUID(payload['bookingId'])}"
            if payload.get("status") not in _BOOKED_STATUSES:
                self.repository.delete_interaction(_interaction_id(key))
                return
            self._save(key, user_id, UUID(payload["venueId"]), RecommendationTargetType.VENUE,
                       InteractionType.BOOKING, _sport(payload.get("sportType")), occurred_at,
                       _venue_metadata(payload))
        elif event_type == POST_REACTION:
            interaction_type = _REACTIONS[payload["reaction"]]
            post_id = UUID(payload["postId"])
            key = f"reaction:{post_id}:{user_id}:{interaction_type.value}"
            self._post_interaction(key, user_id, post_id, interaction_type, payload, occurred_at)
        elif event_type == POST_COMMENT:
            key = f"comment:{UUID(payload['commentId'])}"
            self._post_interaction(key, user_id, UUID(payload["postId"]), InteractionType.COMMENT, payload,
                                   occurred_at)
        elif event_type == POST_SHARED:
            post_id = UUID(payload["postId"])
            self._post_interaction(f"share:{post_id}:{user_id}", user_id, post_id, InteractionType.SHARE,
                                   {**payload, "active": True}, occurred_at)
        else:
            logger.debug("Ignoring activity event type %s", event_type)

    def _post_interaction(self, key: str, user_id: UUID, post_id: UUID, interaction_type: InteractionType,
                          payload: dict[str, Any], occurred_at: datetime) -> None:
        if not payload.get("active", True):
            self.repository.delete_interaction(_interaction_id(key))
            return
        author_id = payload.get("authorId")
        # Tương tác với bài của chính mình không nói gì về sở thích.
        if author_id and UUID(author_id) == user_id:
            return
        self._save(key, user_id, post_id, RecommendationTargetType.POST, interaction_type,
                   _sport(payload.get("sport")), occurred_at, {"authorId": author_id} if author_id else {})

    def _save(self, key: str, user_id: UUID, target_id: UUID, target_type: RecommendationTargetType,
              interaction_type: InteractionType, sport: SportType | None, occurred_at: datetime,
              metadata: dict[str, Any]) -> None:
        self.repository.save_interaction(UserInteraction(
            interaction_id=_interaction_id(key),
            user_id=user_id,
            target_id=target_id,
            target_type=target_type,
            interaction_type=interaction_type,
            sport_type=sport,
            metadata=metadata,
            occurred_at=occurred_at,
        ))


def _interaction_id(key: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"goatsports:interaction:{key}")


def _sport(value: Any) -> SportType | None:
    try:
        return SportType(value) if value else None
    except ValueError:
        return None


def _venue_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = {}
    for source, target in (("minPrice", "price"), ("totalPrice", "price"), ("rating", "rating")):
        value = payload.get(source)
        if isinstance(value, (int, float)) and target not in metadata:
            metadata[target] = float(value)
    return metadata


def _timestamp(value: Any) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
