from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.application.dto.request.models import (
    MatchBookingRegistrationRequest,
    MatchmakingQueueRequest,
    MatchProposalUpdateRequest,
    MatchResultRequest,
    MatchShareRequest,
    MatchVenueSelectionRequest,
    OpponentFeedbackRequest,
)
from app.application.dto.response.models import MatchCandidate, QueueSearchInfo, QueueSearchMember
from app.application.exception import (
    ConflictException,
    DownstreamServiceException,
    ForbiddenException,
    ResourceNotFoundException,
)
from app.application.port.client import (
    AuthServiceClient,
    ClubServiceClient,
    NotificationServiceClient,
    SocialServiceClient,
    VenueServiceClient,
)
from app.application.port.persistence import (
    AiDocumentRepository,
    MatchmakingQueueRepository,
    MatchmakingSessionRepository,
)
from app.application.service.elo_rating import calculate_new_rating
from app.application.service.match_share import SHAREABLE, build_match_highlight
from app.application.service.matchmaking_model import is_learned, success_probability
from app.application.service.matchmaking_scoring import compatibility_breakdown, overlapping_window
from app.application.service.model_registry import get_or_create_active_model
from app.application.service.service_config import MatchmakingConfig
from app.domain.entity.models import (
    PLAY_FORMATS,
    ClubMatchProfile,
    MatchAcceptance,
    MatchmakingQueueEntry,
    MatchmakingSession,
    MatchParticipantProfile,
    MatchPreference,
    MatchProposal,
    MatchResult,
    MatchResultClaim,
    MatchVenueOption,
    OpponentFeedback,
    PlayerProfile,
    SideMember,
)
from app.domain.enums import (
    AcceptanceDecision,
    AiCapability,
    MatchProposalStatus,
    MatchSelectionMode,
    MatchSessionStatus,
    QueueStatus,
    VenueSearchStatus,
)


class MatchmakingError(Exception):
    pass


class MatchmakingNotFoundError(ResourceNotFoundException, MatchmakingError):
    pass


class MatchmakingForbiddenError(ForbiddenException, MatchmakingError):
    pass


class MatchmakingConflictError(ConflictException, MatchmakingError):
    pass


logger = logging.getLogger(__name__)

MATCHMAKING_FLOW_STATUSES = {
    MatchSessionStatus.PROPOSED,
    MatchSessionStatus.ACCEPTED_BY_ONE,
    MatchSessionStatus.ACCEPTED,
    MatchSessionStatus.VENUE_SELECTED,
    MatchSessionStatus.BOOKING_PENDING,
}

# Đã đồng ý nhưng chưa đặt xong sân: qua hết khung giờ chơi thì kèo không thể diễn ra nữa.
_UNBOOKED_STATUSES = {
    MatchSessionStatus.ACCEPTED,
    MatchSessionStatus.VENUE_SELECTED,
    MatchSessionStatus.BOOKING_PENDING,
}

_SPORT_LABELS = {
    "BADMINTON": "cầu lông",
    "PICKLEBALL": "pickleball",
    "TENNIS": "tennis",
    "FOOTBALL": "bóng đá",
    "BASKETBALL": "bóng rổ",
    "VOLLEYBALL": "bóng chuyền",
}

SCHEDULE_RESERVING_STATUSES = MATCHMAKING_FLOW_STATUSES | {
    MatchSessionStatus.CONFIRMED,
    MatchSessionStatus.CHECKED_IN,
    MatchSessionStatus.RESULT_PENDING,
    MatchSessionStatus.DISPUTED,
}

_RESULT_SETTLEMENT_STATUSES = {
    MatchSessionStatus.CONFIRMED,
    MatchSessionStatus.CHECKED_IN,
    MatchSessionStatus.RESULT_PENDING,
    MatchSessionStatus.DISPUTED,
}

_BOOKING_PROGRESS_RANK = {
    MatchSessionStatus.BOOKING_PENDING: 0,
    MatchSessionStatus.CONFIRMED: 1,
    MatchSessionStatus.CHECKED_IN: 2,
}
_BOOKING_PROGRESS_TERMINAL_AHEAD = {
    MatchSessionStatus.RESULT_PENDING,
    MatchSessionStatus.DISPUTED,
    MatchSessionStatus.COMPLETED,
}


def _is_stale_booking_event(current: MatchSessionStatus, target: MatchSessionStatus) -> bool:
    """Guard against out-of-order booking webhooks/syncs downgrading a session
    that has already progressed further (relevant now that lifecycle updates
    can arrive both via a trusted push from venue-service and via lazy pull)."""
    if current in _BOOKING_PROGRESS_TERMINAL_AHEAD:
        return True
    current_rank = _BOOKING_PROGRESS_RANK.get(current)
    return current_rank is not None and _BOOKING_PROGRESS_RANK[target] < current_rank


def _cannot_cancel_booking(current: MatchSessionStatus) -> bool:
    return current == MatchSessionStatus.CHECKED_IN or current in _BOOKING_PROGRESS_TERMINAL_AHEAD

MATCH_BUFFER = timedelta(minutes=60)


class MatchmakingService:
    def __init__(
        self,
        queue_repository: MatchmakingQueueRepository,
        session_repository: MatchmakingSessionRepository,
        document_repository: AiDocumentRepository,
        venue_client: VenueServiceClient,
        social_client: SocialServiceClient,
        notification_client: NotificationServiceClient,
        config: MatchmakingConfig,
        auth_client: AuthServiceClient | None = None,
        club_client: ClubServiceClient | None = None,
    ) -> None:
        self.queue_repository = queue_repository
        self.session_repository = session_repository
        self.document_repository = document_repository
        self.venue_client = venue_client
        self.social_client = social_client
        self.notification_client = notification_client
        self.config = config
        self.auth_client = auth_client
        self.club_client = club_client

    def join_queue(
        self,
        participant_id: UUID,
        request: MatchmakingQueueRequest,
    ) -> MatchmakingSession | None:
        active_session = self._active_session(participant_id, MATCHMAKING_FLOW_STATUSES)
        if active_session:
            raise MatchmakingConflictError(
                f"Bạn đang có một kèo {_SPORT_LABELS.get(active_session.sport_type.value, active_session.sport_type.value)} "
                f"ngày {active_session.play_date.strftime('%d/%m/%Y')} chưa hoàn tất. Mở kèo đó ở Ghép kèo gần đây "
                "để tiếp tục hoặc hủy, rồi tìm kèo mới."
            )
        profile = self._side_profile(participant_id, request)
        for key_id in profile.queue_keys():
            who = self._who(profile, key_id, participant_id)
            if key_id != participant_id and self._active_session(key_id, MATCHMAKING_FLOW_STATUSES):
                raise MatchmakingConflictError(f"{who} đang có một kèo khác chưa hoàn tất.")
            conflict = self._schedule_conflict(key_id, request)
            if conflict:
                prefix = "Lịch tìm đối thủ" if key_id == participant_id else f"Lịch của {who.lower()}"
                raise MatchmakingConflictError(
                    f"{prefix} trùng hoặc cách dưới 60 phút với trận "
                    f"{conflict.sport_type.value} lúc "
                    f"{conflict.start_time.strftime('%H:%M')}–{conflict.end_time.strftime('%H:%M')} "
                    f"ngày {conflict.play_date.strftime('%d/%m/%Y')}."
                )

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=self.config.queue_ttl_seconds)
        preference = MatchPreference(
            sport_type=request.sport_type,
            play_format=request.play_format,
            play_date=request.play_date,
            start_time=request.start_time,
            end_time=request.end_time,
            timezone=request.timezone,
            max_elo_difference=request.max_elo_difference,
            max_distance_km=request.max_distance_km,
            selection_mode=request.selection_mode,
        )

        with self.queue_repository.lock():
            previous = self.queue_repository.find_by_participant(participant_id)
            if previous and previous.status != QueueStatus.QUEUED:
                previous = None
            if previous:
                led = self.queue_repository.find_profile(previous.profile_id)
                if (
                    isinstance(led, PlayerProfile)
                    and led.queued_by is not None
                    and led.queued_by != participant_id
                ):
                    if set(led.queue_keys()) == set(profile.queue_keys()) and led.play_format == profile.play_format:
                        # Người dẫn đã đưa đúng đội này vào hàng chờ: coi như đang cùng tìm.
                        # Trước đây lượt của người dẫn bị ghi đè lặng lẽ (đổi người dẫn, đổi giờ).
                        return None
                    raise MatchmakingConflictError(
                        f"{self._member_name(led, led.queued_by)} đã đưa bạn vào một đội đang tìm đối thủ. "
                        "Rời đội đó trước khi tìm kèo khác."
                    )
            for key_id in profile.queue_keys():
                other = self.queue_repository.find_by_participant(key_id)
                if (
                    other
                    and other.status == QueueStatus.QUEUED
                    and (previous is None or other.queue_entry_id != previous.queue_entry_id)
                ):
                    raise MatchmakingConflictError(
                        self._busy_message(profile, key_id, participant_id, other)
                    )
            if previous:
                old_profile = self.queue_repository.find_profile(previous.profile_id)
                if old_profile and set(old_profile.queue_keys()) != set(profile.queue_keys()):
                    # Đổi bạn cặp hoặc đổi CLB: đóng lượt cũ để người cũ không còn thấy lượt mới.
                    previous.status = QueueStatus.CANCELLED
                    self.queue_repository.save(previous, old_profile)
                    previous = None
            if previous:
                entry = MatchmakingQueueEntry(
                    queue_entry_id=previous.queue_entry_id,
                    profile_id=profile.participant_profile_id,
                    preference=preference,
                    queued_at=previous.queued_at,
                    expires_at=expires_at,
                )
            else:
                entry = MatchmakingQueueEntry(
                    profile_id=profile.participant_profile_id,
                    preference=preference,
                    queued_at=now,
                    expires_at=expires_at,
                )
            self.queue_repository.save(entry, profile)
            matched = (
                self._match_best_candidate(entry, profile)
                if request.selection_mode == MatchSelectionMode.AI
                else None
            )
        if matched is None and previous is None:
            self._notify_team_added(entry, profile, participant_id)
        return matched

    def _side_profile(self, user_id: UUID, request: MatchmakingQueueRequest) -> MatchParticipantProfile:
        """Dựng hồ sơ một bên theo hình thức: đánh đơn, cặp đôi đăng ký sẵn, hoặc CLB."""
        if request.club_id is not None:
            return self._club_side(user_id, request)
        members = [
            SideMember(
                user_id=user_id,
                name=request.player_name,
                elo_rating=request.elo_rating,
                match_count=request.match_count,
            )
        ]
        name, elo_rating = request.player_name, request.elo_rating
        if request.partner_ids:
            members.extend(self._teammates(user_id, request))
            name = (
                f"{request.player_name} & {members[1].name}" if len(members) == 2 else f"Đội {request.player_name}"
            )[:150]
            # ELO của bên là trung bình các thành viên; khi chốt kết quả mỗi người vẫn được tính riêng.
            elo_rating = round(sum(member.elo_rating for member in members) / len(members))
        return PlayerProfile(
            participant_id=user_id,
            name=name,
            avatar_url=request.player_avatar,
            sport_type=request.sport_type,
            skill_level=request.skill_level,
            elo_rating=elo_rating,
            preferred_positions=request.preferred_positions,
            play_radius_km=request.max_distance_km,
            play_style=request.play_style,
            match_count=request.match_count,
            win_rate=request.win_rate,
            latitude=request.latitude,
            longitude=request.longitude,
            play_format=request.play_format,
            queued_by=user_id,
            members=members,
        )

    def _teammates(self, user_id: UUID, request: MatchmakingQueueRequest) -> list[SideMember]:
        """Mỗi đồng đội phải là bạn bè, hoặc cùng là thành viên đang hoạt động của CLB đã chọn (đúng môn)."""
        if user_id in request.partner_ids:
            raise MatchmakingConflictError("Đồng đội phải là người khác.")
        clubmates: set[UUID] = set()
        if request.teammate_club_id is not None:
            if not self.club_client:
                raise DownstreamServiceException("Club service is not configured")
            club = self.club_client.matchmaking_profile(
                request.teammate_club_id, request.play_format or "", request.latitude, request.longitude
            )
            if club is None or user_id not in club.member_ids:
                raise MatchmakingForbiddenError("Bạn không phải thành viên của CLB đã chọn.")
            if club.sport_type != request.sport_type:
                raise MatchmakingConflictError("CLB không thi đấu môn này.")
            clubmates = set(club.member_ids)
        pending = [teammate for teammate in request.partner_ids if teammate not in clubmates]
        if pending:
            friends = self.social_client.friend_ids(user_id)
            if any(teammate not in friends for teammate in pending):
                raise MatchmakingForbiddenError(
                    "Đồng đội phải là bạn bè của bạn hoặc thành viên cùng CLB đã chọn."
                )
        if not self.auth_client:
            raise DownstreamServiceException("Auth service is not configured")
        # ponytail: 2 lời gọi auth cho mỗi đồng đội (tối đa 10); gộp thành một API theo lô nếu sân 11 bị chậm.
        return [self.auth_client.teammate(teammate, request.sport_type) for teammate in request.partner_ids]

    def _club_side(self, user_id: UUID, request: MatchmakingQueueRequest) -> ClubMatchProfile:
        assert request.club_id is not None and request.play_format is not None
        if not self.club_client:
            raise DownstreamServiceException("Club service is not configured")
        # Vị trí tìm sân là nơi người quản lý chọn, không phải địa chỉ đăng ký của CLB.
        club = self.club_client.matchmaking_profile(
            request.club_id, request.play_format, request.latitude, request.longitude
        )
        if club is None:
            raise MatchmakingNotFoundError("Không tìm thấy câu lạc bộ.")
        if user_id not in club.manager_ids:
            raise MatchmakingForbiddenError("Chỉ chủ CLB hoặc quản lý mới được tìm đối thủ cho CLB.")
        if club.sport_type != request.sport_type:
            raise MatchmakingConflictError("CLB không thi đấu môn này.")
        needed = PLAY_FORMATS[request.play_format][1]
        if club.active_member_count < needed:
            raise MatchmakingConflictError(
                f"CLB cần ít nhất {needed} thành viên đang hoạt động cho hình thức này."
            )
        club.queued_by = user_id
        return club

    @staticmethod
    def _who(profile: MatchParticipantProfile, key_id: UUID, user_id: UUID) -> str:
        if key_id == user_id:
            return "Bạn"
        if isinstance(profile, ClubMatchProfile):
            return "CLB"
        return "Bạn cặp" if len(profile.members) == 2 else "Đồng đội"

    def get_status(self, participant_id: UUID) -> tuple[str, int | None, MatchmakingSession | None]:
        session = self._active_session(participant_id)
        if session:
            session = self._ensure_conversation(session)
            session = self._ensure_venue_options(session)
            session = self._sync_booking(session)
            return session.status.value, None, session
        entry = self.queue_repository.find_by_participant(participant_id)
        if entry:
            if entry.status == QueueStatus.QUEUED and entry.preference.selection_mode == MatchSelectionMode.AI:
                profile = self.queue_repository.find_profile(entry.profile_id)
                if profile:
                    with self.queue_repository.lock():
                        refreshed = self.queue_repository.find_by_participant(participant_id)
                        if refreshed and refreshed.status == QueueStatus.QUEUED:
                            matched = self._match_best_candidate(refreshed, profile)
                            if matched:
                                return matched.status.value, None, matched
            return entry.status.value, self.queue_repository.queue_size(), None
        latest = self.session_repository.find_latest_by_participant(participant_id)
        if latest:
            # Phiên đã kết thúc không đi qua `_active_session` nên cũng không đi qua
            # `_expire_if_needed`; vẫn phải thử đồng bộ lại ELO ở đây, vì đây chính
            # là màn hình người chơi đang nhìn sau khi chốt kết quả.
            latest = self._retry_elo_sync(latest)
            return latest.status.value, None, latest
        return "NOT_IN_QUEUE", None, None

    def get_candidates(self, participant_id: UUID, limit: int = 5) -> list[MatchCandidate]:
        entry = self.queue_repository.find_by_participant(participant_id)
        if not entry or entry.status != QueueStatus.QUEUED:
            return []
        profile = self.queue_repository.find_profile(entry.profile_id)
        if not profile:
            return []
        ranked = self._rank_candidates(entry, profile)
        return [
            MatchCandidate(
                participant=candidate_profile,
                scoreBreakdown=breakdown,
                distanceKm=distance,
                eloDifference=elo_difference,
                startTime=window[0].isoformat(timespec="minutes"),
                endTime=window[1].isoformat(timespec="minutes"),
            )
            for _, candidate_profile, breakdown, distance, elo_difference, window in ranked[:max(1, min(limit, 10))]
        ]

    def select_candidate(
        self,
        participant_id: UUID,
        candidate_participant_id: UUID,
    ) -> MatchmakingSession:
        if participant_id == candidate_participant_id:
            raise MatchmakingConflictError("Bạn không thể tự chọn chính mình.")
        with self.queue_repository.lock():
            entry = self.queue_repository.find_by_participant(participant_id)
            candidate_entry = self.queue_repository.find_by_participant(candidate_participant_id)
            if not entry or not candidate_entry or entry.status != QueueStatus.QUEUED or candidate_entry.status != QueueStatus.QUEUED:
                raise MatchmakingConflictError("Đối thủ không còn trong hàng chờ.")
            profile = self.queue_repository.find_profile(entry.profile_id)
            candidate_profile = self.queue_repository.find_profile(candidate_entry.profile_id)
            if not profile or not candidate_profile:
                raise MatchmakingNotFoundError("Không tìm thấy hồ sơ ghép trận.")
            result = compatibility_breakdown(profile, entry.preference, candidate_profile, candidate_entry.preference)
            if result is None or result[0].total_score < self.config.minimum_score:
                raise MatchmakingConflictError("Đối thủ không còn đáp ứng tiêu chí ghép trận.")
            breakdown, distance, elo_difference = result
            return self._create_session(
                entry,
                profile,
                candidate_entry,
                candidate_profile,
                breakdown,
                distance,
                elo_difference,
            )

    def leave_queue(self, participant_id: UUID) -> bool:
        with self.queue_repository.lock():
            entry = self.queue_repository.find_by_participant(participant_id)
            if not entry or entry.status != QueueStatus.QUEUED:
                return False
            profile = self.queue_repository.find_profile(entry.profile_id)
            if profile is None:
                return False
            entry.status = QueueStatus.CANCELLED
            entry.expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
            self.queue_repository.save(entry, profile)
        # Đội thiếu một người thì không ghép được nữa: cả đội dừng tìm, những người còn lại được báo.
        if isinstance(profile, PlayerProfile) and len(profile.members) > 1:
            leader = self._member_name(profile, profile.queued_by)
            if participant_id == profile.queued_by:
                title, content = "Đội đã dừng tìm đối thủ", f"{leader} đã hủy lượt tìm đối thủ của đội."
            else:
                title = "Đội đã dừng tìm đối thủ"
                content = (
                    f"{self._member_name(profile, participant_id)} đã rời đội nên lượt tìm đối thủ của đội đã dừng. "
                    f"{leader} có thể chọn người khác rồi tìm lại."
                )
            self._notify_users(
                [member.user_id for member in profile.members if member.user_id != participant_id],
                entry.queue_entry_id, title, content,
            )
        return True

    def current_search(self, participant_id: UUID) -> QueueSearchInfo | None:
        """Lượt tìm đang chờ có mình (tự tìm, hoặc được người khác thêm vào đội)."""
        entry = self.queue_repository.find_by_participant(participant_id)
        if not entry or entry.status != QueueStatus.QUEUED:
            return None
        profile = self.queue_repository.find_profile(entry.profile_id)
        if profile is None:
            return None
        preference = entry.preference
        return QueueSearchInfo(
            sport_type=preference.sport_type,
            play_format=preference.play_format,
            play_date=preference.play_date,
            start_time=preference.start_time,
            end_time=preference.end_time,
            queued_at=entry.queued_at,
            queued_by=profile.queued_by,
            leader_name=self._member_name(profile, profile.queued_by),
            members=[
                QueueSearchMember(user_id=member.user_id, name=member.name)
                for member in (profile.members if not isinstance(profile, ClubMatchProfile) else [])
            ],
        )

    @staticmethod
    def _member_name(profile: MatchParticipantProfile, user_id: UUID | None) -> str:
        for member in profile.members:
            if member.user_id == user_id:
                return member.name
        return profile.name

    def _busy_message(
        self,
        profile: MatchParticipantProfile,
        key_id: UUID,
        participant_id: UUID,
        other: MatchmakingQueueEntry,
    ) -> str:
        if isinstance(profile, ClubMatchProfile) or key_id == participant_id:
            who = self._who(profile, key_id, participant_id)
        else:
            who = self._member_name(profile, key_id)
        other_profile = self.queue_repository.find_profile(other.profile_id)
        if (
            isinstance(other_profile, PlayerProfile)
            and len(other_profile.members) > 1
            and other_profile.queued_by not in (None, key_id)
        ):
            return f"{who} đang ở trong đội của {self._member_name(other_profile, other_profile.queued_by)} đang tìm đối thủ."
        return f"{who} đang ở một lượt tìm đối thủ khác."

    def _notify_team_added(
        self, entry: MatchmakingQueueEntry, profile: MatchParticipantProfile, leader_id: UUID
    ) -> None:
        if not isinstance(profile, PlayerProfile) or len(profile.members) < 2:
            return
        preference = entry.preference
        sport = _SPORT_LABELS.get(preference.sport_type.value, preference.sport_type.value)
        self._notify_users(
            [member.user_id for member in profile.members if member.user_id != leader_id],
            entry.queue_entry_id,
            "Bạn được thêm vào một đội đang tìm đối thủ",
            f"{self._member_name(profile, leader_id)} đã thêm bạn vào đội {sport} "
            f"ngày {preference.play_date.strftime('%d/%m/%Y')} "
            f"{preference.start_time.strftime('%H:%M')}–{preference.end_time.strftime('%H:%M')}. "
            "Mở AI ghép trận để xem hoặc rời đội.",
        )

    def _notify_users(self, user_ids: list[UUID], reference_id: UUID, title: str, content: str) -> None:
        """Best-effort: thông báo lỗi không làm hỏng thao tác hàng chờ."""
        for user_id in user_ids:
            try:
                self.notification_client.notify_matchmaking(user_id, reference_id, title, content)
            except DownstreamServiceException:
                continue

    def share_session(self, session_id: UUID, user_id: UUID, request: MatchShareRequest) -> UUID:
        """Người chơi trong trận khoe trận lên Cộng đồng, gắn thẻ đối thủ (và đồng đội). Trả id bài viết."""
        session = self._require_session(session_id)
        side = self._require_participant(session, user_id)
        if session.status not in SHAREABLE:
            raise MatchmakingConflictError("Chỉ khoe được kèo đã chốt hoặc trận đã đấu xong.")
        return self.social_client.publish_highlight(
            build_match_highlight(session, side, user_id, request.title, request.content)
        )

    def get_session(self, session_id: UUID, participant_id: UUID) -> MatchmakingSession:
        session = self._require_session(session_id)
        self._require_participant(session, participant_id)
        session = self._expire_if_needed(session)
        session = self._ensure_conversation(session)
        session = self._ensure_venue_options(session)
        return self._sync_booking(session)

    def get_history(
        self,
        participant_id: UUID,
        limit: int = 5,
        offset: int = 0,
    ) -> list[MatchmakingSession]:
        sessions = self.session_repository.find_by_participant(
            participant_id,
            max(1, min(limit, 20)),
            max(0, offset),
        )
        healed = []
        for session in sessions:
            session = self._expire_if_needed(session)
            session = self._ensure_conversation(session)
            session = self._ensure_venue_options(session)
            session = self._sync_booking(session)
            healed.append(session)
        return healed

    def decide(
        self,
        session_id: UUID,
        participant_id: UUID,
        decision: AcceptanceDecision,
    ) -> MatchmakingSession:
        with self.queue_repository.lock():
            session = self._expire_if_needed(self._require_session(session_id))
            profile = self._require_participant(session, participant_id)
            if session.status not in {
                MatchSessionStatus.PROPOSED,
                MatchSessionStatus.ACCEPTED_BY_ONE,
            }:
                raise MatchmakingConflictError("Phiên ghép kèo không còn chờ phản hồi.")
            if any(item.participant_profile_id == profile.participant_profile_id for item in session.acceptances):
                raise MatchmakingConflictError("Bạn đã phản hồi lời mời ghép kèo này.")

            session.acceptances.append(
                MatchAcceptance(
                    participant_profile_id=profile.participant_profile_id,
                    decision=decision,
                )
            )
            if decision == AcceptanceDecision.REJECTED:
                session.status = MatchSessionStatus.REJECTED
                if session.proposal:
                    session.proposal.status = MatchProposalStatus.CANCELLED
            elif len(session.acceptances) == 1:
                session.status = MatchSessionStatus.ACCEPTED_BY_ONE
            else:
                session.status = MatchSessionStatus.ACCEPTED
                session.accepted_at = datetime.now(timezone.utc)
                self._attach_conversation(session, participant_id)
            return self.session_repository.save(session)

    def update_proposal(
        self,
        session_id: UUID,
        participant_id: UUID,
        request: MatchProposalUpdateRequest,
    ) -> MatchmakingSession:
        session = self._expire_if_needed(self._require_session(session_id))
        self._require_participant(session, participant_id)
        if not session.proposal:
            raise MatchmakingConflictError("Phiên ghép kèo chưa có đề xuất.")
        self._require_designated_booker(session, participant_id)
        if request.venue_id is not None:
            session.proposal.venue_id = request.venue_id
        if request.venue_court_id is not None:
            session.proposal.venue_court_id = request.venue_court_id
        if request.booking_id is not None:
            session.proposal.booking_id = request.booking_id
        if request.status is not None:
            # Compatibility endpoint: lifecycle state is always derived from venue-service,
            # never trusted from the browser.
            return self._sync_booking(session)
        return self.session_repository.save(session)

    def _match_best_candidate(
        self,
        current_entry: MatchmakingQueueEntry,
        current_profile: MatchParticipantProfile,
    ) -> MatchmakingSession | None:
        ranked = self._rank_candidates(current_entry, current_profile)
        if not ranked:
            return None
        candidate_entry, candidate_profile, breakdown, distance, elo_difference, _ = ranked[0]
        return self._create_session(
            current_entry,
            current_profile,
            candidate_entry,
            candidate_profile,
            breakdown,
            distance,
            elo_difference,
        )

    def _rank_candidates(
        self,
        current_entry: MatchmakingQueueEntry,
        current_profile: MatchParticipantProfile,
    ) -> list[tuple]:
        ranked: list[tuple] = []
        active_model = self.document_repository.find_active_model(AiCapability.MATCHMAKING)
        learned = active_model if is_learned(active_model) else None
        for candidate_entry in self.queue_repository.queued_entries(self.config.scan_limit):
            if candidate_entry.queue_entry_id == current_entry.queue_entry_id:
                continue
            candidate_profile = self.queue_repository.find_profile(candidate_entry.profile_id)
            if candidate_profile is None or set(candidate_profile.queue_keys()) & set(current_profile.queue_keys()):
                continue
            result = compatibility_breakdown(
                current_profile,
                current_entry.preference,
                candidate_profile,
                candidate_entry.preference,
            )
            if result is None or result[0].total_score < self.config.minimum_score:
                continue
            breakdown, distance, elo_difference = result
            if learned:
                breakdown.success_probability = round(100.0 * success_probability(learned, breakdown), 1)
            window = overlapping_window(current_entry.preference, candidate_entry.preference)
            assert window is not None
            ranked.append((candidate_entry, candidate_profile, breakdown, distance, elo_difference, window))
        # Mô hình học máy (nếu đã đủ dữ liệu) quyết định thứ tự; ngưỡng tối thiểu vẫn theo điểm luật ở trên.
        ranked.sort(
            key=lambda item: item[2].success_probability if learned else item[2].total_score,
            reverse=True,
        )
        return ranked

    def _create_session(
        self,
        current_entry: MatchmakingQueueEntry,
        current_profile: MatchParticipantProfile,
        candidate_entry: MatchmakingQueueEntry,
        candidate_profile: MatchParticipantProfile,
        breakdown,
        distance: float,
        elo_difference: int,
    ) -> MatchmakingSession:
        window = overlapping_window(current_entry.preference, candidate_entry.preference)
        assert window is not None
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=self.config.session_ttl_seconds)
        model = get_or_create_active_model(
            self.document_repository,
            AiCapability.MATCHMAKING,
        )

        session = MatchmakingSession(
            sport_type=current_entry.preference.sport_type,
            play_date=current_entry.preference.play_date,
            start_time=window[0],
            end_time=window[1],
            timezone=current_entry.preference.timezone,
            compatibility_score=breakdown.total_score,
            distance_km=distance,
            elo_difference=elo_difference,
            score_breakdown=breakdown,
            model_version_id=model.model_version_id,
            matched_at=now,
            expires_at=expires_at,
            participants=[current_profile, candidate_profile],
            proposal=MatchProposal(
                designated_booker_id=current_profile.queued_by or current_profile.participant_id,
                expires_at=expires_at,
            ),
        )
        self.session_repository.save(session)
        for entry, profile in [
            (current_entry, current_profile),
            (candidate_entry, candidate_profile),
        ]:
            entry.status = QueueStatus.MATCHED
            entry.matched_session_id = session.session_id
            entry.expires_at = expires_at
            self.queue_repository.save(entry, profile)
        return session

    def enrich_session_venue(self, session_id: UUID) -> None:
        session = self.session_repository.find_by_id(session_id)
        if (
            not session
            or not session.proposal
            or session.status != MatchSessionStatus.ACCEPTED
            or session.proposal.venue_options
            or session.proposal.venue_search_status != VenueSearchStatus.PENDING
        ):
            return
        latitude = sum(item.latitude for item in session.participants) / len(session.participants)
        longitude = sum(item.longitude for item in session.participants) / len(session.participants)
        radii = [
            item.play_radius_km
            for item in session.participants
            if isinstance(item, PlayerProfile)
        ]
        radius_km = min(radii) if radii else 10.0
        try:
            recommendations, _ = self.venue_client.recommend(
                latitude,
                longitude,
                session.sport_type,
                radius_km,
                session.play_date,
                session.start_time,
                session.end_time,
                max_results=3,
            )
            options = [
                MatchVenueOption(
                    venueId=item.venue_id,
                    venueCourtId=item.venue_court_id,
                    venueName=item.venue_name,
                    address=item.address,
                    distanceKm=item.distance_km,
                    rating=item.rating,
                    minPrice=item.min_price,
                    maxPrice=item.max_price,
                    score=item.score,
                    matchReason=item.match_reason,
                    suggestedCourts=item.suggested_courts,
                    slotStart=item.slot_start,
                    slotEnd=item.slot_end,
                )
                for item in recommendations[:3]
                if item.venue_court_id is not None
            ]
            if not options:
                # Chua co san doi tac con trong: goi y san ngoai he thong gan nhom, nguoi dat goi san de chot.
                options = self.venue_client.nearby_external_venues(
                    latitude, longitude, session.sport_type, radius_km, max_results=3
                )
        except (DownstreamServiceException, ValueError, KeyError):
            self._set_venue_search_status(session_id, VenueSearchStatus.SERVICE_UNAVAILABLE)
            self._notify_venue_search_updated(session_id)
            return
        if not options:
            self._set_venue_search_status(session_id, VenueSearchStatus.NO_AVAILABILITY)
            self._notify_venue_search_updated(session_id)
            return
        with self.queue_repository.lock():
            current = self.session_repository.find_by_id(session_id)
            if (
                not current
                or not current.proposal
                or current.status != MatchSessionStatus.ACCEPTED
                or current.proposal.venue_options
            ):
                return
            current.proposal.venue_options = options
            current.proposal.venue_search_status = VenueSearchStatus.READY
            self.session_repository.save(current)
        self._notify_venue_search_updated(session_id)

    def refresh_venue_options(
        self,
        session_id: UUID,
        participant_id: UUID,
    ) -> MatchmakingSession:
        with self.queue_repository.lock():
            session = self._require_session(session_id)
            self._require_participant(session, participant_id)
            if session.status != MatchSessionStatus.ACCEPTED or not session.proposal:
                raise MatchmakingConflictError("Chỉ có thể tìm lại sân sau khi cả hai bên xác nhận kèo.")
            session.proposal.venue_options = []
            session.proposal.venue_search_status = VenueSearchStatus.PENDING
            self.session_repository.save(session)
        self.enrich_session_venue(session_id)
        return self.session_repository.find_by_id(session_id) or session

    def cancel_session(
        self,
        session_id: UUID,
        participant_id: UUID,
    ) -> MatchmakingSession:
        with self.queue_repository.lock():
            session = self._require_session(session_id)
            self._require_participant(session, participant_id)
            if session.status not in {
                MatchSessionStatus.PROPOSED,
                MatchSessionStatus.ACCEPTED_BY_ONE,
                MatchSessionStatus.ACCEPTED,
                MatchSessionStatus.VENUE_SELECTED,
            } and not (
                session.status == MatchSessionStatus.CONFIRMED
                and session.proposal
                and session.proposal.offline_booking
            ):
                raise MatchmakingConflictError("Kèo không thể hủy ở trạng thái hiện tại.")
            session.status = MatchSessionStatus.CANCELLED
            if session.proposal:
                session.proposal.status = MatchProposalStatus.CANCELLED
                session.proposal.cancel_reason = "PLAYER_CANCELLED"
            return self.session_repository.save(session)

    def select_venue(
        self,
        session_id: UUID,
        participant_id: UUID,
        request: MatchVenueSelectionRequest,
    ) -> MatchmakingSession:
        with self.queue_repository.lock():
            session = self._expire_if_needed(self._require_session(session_id))
            self._require_designated_booker(session, participant_id)
            if session.status != MatchSessionStatus.ACCEPTED or not session.proposal:
                raise MatchmakingConflictError("Chỉ được chọn sân sau khi cả hai người chơi đã chấp nhận.")
            option = next(
                (
                    item
                    for item in session.proposal.venue_options
                    if item.venue_id == request.venue_id
                    and item.venue_court_id == request.venue_court_id
                ),
                None,
            )
            if not option:
                raise MatchmakingConflictError("Sân đã chọn không nằm trong danh sách còn lịch trống.")
            now = datetime.now(timezone.utc)
            session.proposal.venue_id = option.venue_id
            if option.external:
                # San ngoai he thong: nguoi dat da goi san chot gio, khong co dat coc hay ve check-in.
                session.proposal.offline_booking = True
                session.proposal.status = MatchProposalStatus.BOOKED
                session.proposal.selected_at = now
                session.proposal.booked_at = now
                session.confirmed_at = now
                session.status = MatchSessionStatus.CONFIRMED
                return self.session_repository.save(session)
            session.proposal.venue_court_id = option.venue_court_id
            if option.slot_start and option.slot_end:
                # Tu day gio cua keo la gio thi dau that (mot slot), khong con la ca khung ranh chung:
                # dat san, check-in, nhac nhap ty so va kiem tra trung lich deu dung gio nay.
                session.start_time = option.slot_start
                session.end_time = option.slot_end
            session.proposal.status = MatchProposalStatus.VENUE_SELECTED
            session.proposal.selected_at = now
            session.status = MatchSessionStatus.VENUE_SELECTED
            return self.session_repository.save(session)

    def register_booking(
        self,
        session_id: UUID,
        participant_id: UUID,
        request: MatchBookingRegistrationRequest,
    ) -> MatchmakingSession:
        with self.queue_repository.lock():
            session = self._require_session(session_id)
            self._require_designated_booker(session, participant_id)
            if session.status not in {MatchSessionStatus.VENUE_SELECTED, MatchSessionStatus.BOOKING_PENDING}:
                raise MatchmakingConflictError("Phiên ghép trận chưa sẵn sàng để gắn booking.")
            assert session.proposal is not None
            if session.proposal.booking_id and session.proposal.booking_id != request.booking_id:
                raise MatchmakingConflictError("Phiên ghép trận đã được gắn với một booking khác.")
            snapshot = self.venue_client.get_matchmaking_booking(session.session_id)
            if (
                snapshot is None
                or snapshot.booking_id != request.booking_id
                or snapshot.player_id != participant_id
                or snapshot.venue_court_id != session.proposal.venue_court_id
            ):
                raise MatchmakingConflictError("Booking không khớp với người đặt cọc, sân hoặc phiên ghép trận.")
            session.proposal.booking_id = request.booking_id
            session.proposal.status = MatchProposalStatus.BOOKING_PENDING
            session.proposal.booking_created_at = session.proposal.booking_created_at or datetime.now(timezone.utc)
            session.status = MatchSessionStatus.BOOKING_PENDING
            return self.session_repository.save(session)

    def update_lifecycle(
        self,
        session_id: UUID,
        booking_id: UUID,
        booking_status: str,
        reason: str | None = None,
    ) -> MatchmakingSession:
        """Apply an idempotent booking event received from the trusted venue-service.

        Safe against out-of-order delivery: an event that would move the session
        backward relative to its current progress (e.g. a delayed CONFIRMED event
        arriving after the match was already CHECKED_IN) is ignored rather than
        applied, since this can now be invoked both from a real-time push and from
        the lazy pull in `_sync_booking`.
        """
        with self.queue_repository.lock():
            session = self._require_session(session_id)
            if not session.proposal:
                raise MatchmakingConflictError("Phiên ghép trận chưa có đề xuất booking.")
            if session.proposal.booking_id and session.proposal.booking_id != booking_id:
                raise MatchmakingConflictError("Booking không thuộc phiên ghép trận này.")
            normalized = booking_status.upper()
            now = datetime.now(timezone.utc)
            if normalized in {"PENDING_PAYMENT", "PENDING", "CONFIRMED", "PAID", "CHECKED_IN", "COMPLETED"}:
                target = {
                    "PENDING_PAYMENT": MatchSessionStatus.BOOKING_PENDING,
                    "PENDING": MatchSessionStatus.BOOKING_PENDING,
                    "CONFIRMED": MatchSessionStatus.CONFIRMED,
                    "PAID": MatchSessionStatus.CONFIRMED,
                    "CHECKED_IN": MatchSessionStatus.CHECKED_IN,
                    "COMPLETED": MatchSessionStatus.CHECKED_IN,
                }[normalized]
                if _is_stale_booking_event(session.status, target):
                    session.proposal.booking_id = booking_id
                    return self.session_repository.save(session)
                session.proposal.booking_id = booking_id
                if target == MatchSessionStatus.BOOKING_PENDING:
                    session.proposal.status = MatchProposalStatus.BOOKING_PENDING
                    session.proposal.booking_created_at = session.proposal.booking_created_at or now
                elif target == MatchSessionStatus.CONFIRMED:
                    session.proposal.status = MatchProposalStatus.BOOKED
                    session.proposal.booked_at = session.proposal.booked_at or now
                    session.confirmed_at = session.confirmed_at or now
                else:
                    session.proposal.status = MatchProposalStatus.CHECKED_IN
                    session.proposal.checked_in_at = session.proposal.checked_in_at or now
                    session.checked_in_at = session.checked_in_at or now
                session.status = target
            elif normalized in {"CANCELLED", "EXPIRED", "REFUNDED"}:
                if _cannot_cancel_booking(session.status):
                    session.proposal.booking_id = booking_id
                    return self.session_repository.save(session)
                session.proposal.booking_id = booking_id
                session.proposal.status = MatchProposalStatus.CANCELLED
                session.proposal.cancel_reason = reason or normalized
                session.status = MatchSessionStatus.CANCELLED
            else:
                raise MatchmakingConflictError("Trạng thái booking không được hỗ trợ.")
            return self.session_repository.save(session)

    def sync_lifecycle_from_venue(
        self,
        session_id: UUID,
        booking_id: UUID,
        booking_status: str,
        reason: str | None = None,
    ) -> MatchmakingSession:
        """Entry point for the trusted push from venue-service (payment/check-in
        events), so booking-driven transitions notify participants immediately
        instead of waiting for the next lazy `_sync_booking` pull."""
        previous_status = self._require_session(session_id).status
        updated = self.update_lifecycle(session_id, booking_id, booking_status, reason)
        if updated.status != previous_status:
            self.notify_lifecycle(updated)
        return updated

    def _sync_booking(self, session: MatchmakingSession) -> MatchmakingSession:
        if session.proposal and session.proposal.offline_booking:
            return session
        if session.status not in {
            MatchSessionStatus.VENUE_SELECTED,
            MatchSessionStatus.BOOKING_PENDING,
            MatchSessionStatus.CONFIRMED,
            MatchSessionStatus.CHECKED_IN,
        }:
            return session
        try:
            snapshot = self.venue_client.get_matchmaking_booking(session.session_id)
        except DownstreamServiceException:
            return session
        if snapshot is None:
            return session
        if session.proposal and (
            session.proposal.booking_id != snapshot.booking_id
            or session.status.value != snapshot.status
        ):
            try:
                previous_status = session.status
                updated = self.update_lifecycle(session.session_id, snapshot.booking_id, snapshot.status)
                if updated.status != previous_status:
                    self.notify_lifecycle(updated)
                return updated
            except MatchmakingConflictError:
                return session
        return session

    def _ensure_venue_options(self, session: MatchmakingSession) -> MatchmakingSession:
        if (
            session.status == MatchSessionStatus.ACCEPTED
            and session.proposal
            and not session.proposal.venue_options
            and session.proposal.venue_search_status == VenueSearchStatus.PENDING
        ):
            self.enrich_session_venue(session.session_id)
            return self.session_repository.find_by_id(session.session_id) or session
        return session

    def _set_venue_search_status(
        self,
        session_id: UUID,
        search_status: VenueSearchStatus,
    ) -> None:
        with self.queue_repository.lock():
            current = self.session_repository.find_by_id(session_id)
            if (
                current
                and current.proposal
                and current.status == MatchSessionStatus.ACCEPTED
                and not current.proposal.venue_options
            ):
                current.proposal.venue_search_status = search_status
                self.session_repository.save(current)

    def _ensure_conversation(self, session: MatchmakingSession) -> MatchmakingSession:
        if (
            session.status not in {
                MatchSessionStatus.ACCEPTED,
                MatchSessionStatus.VENUE_SELECTED,
                MatchSessionStatus.BOOKING_PENDING,
                MatchSessionStatus.CONFIRMED,
                MatchSessionStatus.CHECKED_IN,
                MatchSessionStatus.RESULT_PENDING,
                MatchSessionStatus.DISPUTED,
            }
            or not session.proposal
            or session.proposal.conversation_id
        ):
            return session
        with self.queue_repository.lock():
            current = self.session_repository.find_by_id(session.session_id) or session
            if not current.proposal or current.proposal.conversation_id:
                return current
            self._attach_conversation(current, current.proposal.designated_booker_id)
            if current.proposal.conversation_id:
                return self.session_repository.save(current)
            return current

    def submit_result(
        self,
        session_id: UUID,
        participant_id: UUID,
        request: MatchResultRequest,
    ) -> MatchmakingSession:
        with self.queue_repository.lock():
            session = self._settle_finished_match(self._require_session(session_id))
            side = self._require_participant(session, participant_id)
            offline_confirmed = (
                session.status == MatchSessionStatus.CONFIRMED
                and session.proposal is not None
                and session.proposal.offline_booking
            )
            if not offline_confirmed and session.status not in {
                MatchSessionStatus.CHECKED_IN,
                MatchSessionStatus.RESULT_PENDING,
                MatchSessionStatus.DISPUTED,
            }:
                raise MatchmakingConflictError("Chỉ được nhập kết quả sau khi trận đấu đã check-in.")
            if not self._match_has_ended(session):
                raise MatchmakingConflictError("Chỉ được nhập kết quả sau khi trận đấu kết thúc.")
            if side.participant_id == session.participants[0].participant_id:
                first_score, second_score = request.my_score, request.opponent_score
            else:
                first_score, second_score = request.opponent_score, request.my_score
            winner_id = None
            if first_score != second_score:
                winner_id = session.participants[0 if first_score > second_score else 1].participant_id
            claim = MatchResultClaim(
                submittedBy=participant_id,
                participantOneScore=first_score,
                participantTwoScore=second_score,
                winnerId=winner_id,
            )
            # Mỗi bên một tỷ số: người cùng bên (bạn cặp, quản lý khác của CLB) nhập lại thì thay tỷ số cũ.
            session.result_claims = [
                item
                for item in session.result_claims
                if self._side_id(session, item.submitted_by) != side.participant_id
            ]
            session.result_claims.append(claim)
            if len(session.result_claims) < 2:
                session.status = MatchSessionStatus.RESULT_PENDING
                return self.session_repository.save(session)
            first_claim, second_claim = session.result_claims
            if (
                first_claim.participant_one_score != second_claim.participant_one_score
                or first_claim.participant_two_score != second_claim.participant_two_score
            ):
                session.status = MatchSessionStatus.DISPUTED
                return self.session_repository.save(session)
            return self._finalize_result(session, first_score, second_score, winner_id)

    def _finalize_result(
        self,
        session: MatchmakingSession,
        first_score: int,
        second_score: int,
        winner_id: UUID | None,
    ) -> MatchmakingSession:
        """Chốt kết quả và đóng phiên.

        Hai bên đã thống nhất tỷ số thì kết quả phải được ghi nhận, kể cả khi
        auth-service đang lỗi. Nếu không áp được ELO, phiên vẫn COMPLETED với
        `elo_applied=False` và sẽ được đồng bộ lại ở lần đọc kế tiếp — thay vì
        trả 503 khiến người chơi không bao giờ kết thúc được trận đấu.
        """
        elo_updates, elo_applied = self._apply_elo(session, winner_id, first_score, second_score)
        now = datetime.now(timezone.utc)
        session.result = MatchResult(
            participantOneScore=first_score,
            participantTwoScore=second_score,
            winnerId=winner_id,
            confirmedAt=now,
            eloUpdates=elo_updates,
            eloApplied=elo_applied,
        )
        session.status = MatchSessionStatus.COMPLETED
        session.completed_at = now
        if session.proposal:
            session.proposal.status = MatchProposalStatus.COMPLETED
        return self.session_repository.save(session)

    def _apply_elo(
        self,
        session: MatchmakingSession,
        winner_id: UUID | None,
        first_score: int | None = None,
        second_score: int | None = None,
    ) -> tuple[dict[str, int], bool]:
        """Trả về (elo_updates, đã đồng bộ được sang auth-service hay chưa)."""
        local_updates = self._calculate_elo_updates(session, winner_id)
        first, second = session.participants
        try:
            if isinstance(first, ClubMatchProfile):
                # CLB đấu CLB: ELO, thắng/hòa/thua và lịch sử trận nằm ở club-service.
                if not self.club_client:
                    return local_updates, True
                return self.club_client.apply_friendly_result(
                    session.session_id,
                    session.sport_type,
                    first.play_format,
                    (first.participant_id, first_score or 0),
                    (second.participant_id, second_score or 0),
                ), True
            if not self.auth_client:
                return local_updates, True
            updates: dict[str, int] = {}
            for side, opponent, mine, theirs in (
                (first, second, first_score, second_score),
                (second, first, second_score, first_score),
            ):
                outcome = "DRAW" if winner_id is None else ("WIN" if winner_id == side.participant_id else "LOSS")
                # Đánh đôi: mỗi người tính theo ELO của chính mình so với ELO trung bình của cặp đối thủ.
                for member in self._rated_members(side):
                    updates[str(member.user_id)] = self.auth_client.apply_match_result(
                        session.session_id, member.user_id, session.sport_type,
                        opponent.elo_rating, outcome, member.elo_rating,
                        opponent_name=opponent.name, my_score=mine, opponent_score=theirs,
                    )
            return updates, True
        except DownstreamServiceException:
            logger.warning(
                "ELO update could not be applied for session %s; result is stored and will be retried",
                session.session_id,
            )
            return local_updates, False

    def _retry_elo_sync(self, session: MatchmakingSession) -> MatchmakingSession:
        """Đồng bộ lại ELO cho phiên đã chốt nhưng auth-service từng lỗi."""
        if (
            session.status != MatchSessionStatus.COMPLETED
            or not session.result
            or session.result.elo_applied
        ):
            return session
        elo_updates, elo_applied = self._apply_elo(
            session,
            session.result.winner_id,
            session.result.participant_one_score,
            session.result.participant_two_score,
        )
        if not elo_applied:
            return session
        session.result.elo_updates = elo_updates
        session.result.elo_applied = True
        return self.session_repository.save(session)

    def submit_feedback(
        self,
        session_id: UUID,
        participant_id: UUID,
        request: OpponentFeedbackRequest,
    ) -> MatchmakingSession:
        with self.queue_repository.lock():
            session = self._require_session(session_id)
            side = self._require_participant(session, participant_id)
            if session.status != MatchSessionStatus.COMPLETED:
                raise MatchmakingConflictError("Chỉ được đánh giá sau khi kết quả đã được xác nhận.")
            if any(self._side_id(session, item.reviewer_id) == side.participant_id for item in session.feedback):
                raise MatchmakingConflictError("Bạn đã đánh giá đối thủ của trận này.")
            reviewee = next(item for item in session.participants if item.participant_id != side.participant_id)
            session.feedback.append(
                OpponentFeedback(
                    reviewerId=participant_id,
                    revieweeId=reviewee.participant_id,
                    rating=request.rating,
                    fairPlayRating=request.fair_play_rating,
                    comment=request.comment,
                )
            )
            return self.session_repository.save(session)

    def _active_session(
        self,
        participant_id: UUID,
        statuses: set[MatchSessionStatus] = SCHEDULE_RESERVING_STATUSES,
    ) -> MatchmakingSession | None:
        session = self.session_repository.find_active_by_participant(participant_id)
        if not session:
            return None
        session = self._expire_if_needed(session)
        return session if session.status in statuses else None

    def _schedule_conflict(
        self,
        participant_id: UUID,
        request: MatchmakingQueueRequest,
    ) -> MatchmakingSession | None:
        requested_start, requested_end = self._zoned_interval(
            request.play_date,
            request.start_time,
            request.end_time,
            request.timezone,
        )
        now = datetime.now(timezone.utc)
        sessions = self.session_repository.find_by_participant(participant_id, 20)
        for stored in sessions:
            session = self._expire_if_needed(stored)
            if session.status not in SCHEDULE_RESERVING_STATUSES:
                continue
            reserved_start, reserved_end = self._zoned_interval(
                session.play_date,
                session.start_time,
                session.end_time,
                session.timezone,
            )
            buffered_start = reserved_start - MATCH_BUFFER
            buffered_end = reserved_end + MATCH_BUFFER
            if buffered_end <= now:
                continue
            if requested_start < buffered_end and requested_end > buffered_start:
                return session
        return None

    @staticmethod
    def _zoned_interval(play_date, start_time, end_time, timezone_name):
        try:
            zone = ZoneInfo(timezone_name)
        except (ZoneInfoNotFoundError, TypeError):
            zone = ZoneInfo("Asia/Ho_Chi_Minh")
        start = datetime.combine(play_date, start_time, tzinfo=zone).astimezone(timezone.utc)
        end = datetime.combine(play_date, end_time, tzinfo=zone).astimezone(timezone.utc)
        return start, end

    def _expire_if_needed(self, session: MatchmakingSession) -> MatchmakingSession:
        now = datetime.now(timezone.utc)
        expires_at = session.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at <= now and session.status in {
            MatchSessionStatus.PROPOSED,
            MatchSessionStatus.ACCEPTED_BY_ONE,
        }:
            session.status = MatchSessionStatus.EXPIRED
            if session.proposal:
                session.proposal.status = MatchProposalStatus.EXPIRED
            session = self.session_repository.save(session)
            self.notify_expired(session)
        elif session.status in _UNBOOKED_STATUSES and self._match_has_ended(session):
            # Trước đây các trạng thái này không bao giờ tự đóng: kèo quá giờ vẫn "đang hoạt động",
            # chặn người chơi tìm kèo mới và bị trả lại cho mọi lượt tìm.
            session.status = MatchSessionStatus.EXPIRED
            if session.proposal:
                session.proposal.status = MatchProposalStatus.EXPIRED
                session.proposal.cancel_reason = "NOT_BOOKED"
            session = self.session_repository.save(session)
            self._notify(
                session,
                "Kèo đã hết hạn",
                "Khung giờ chơi đã qua mà kèo chưa đặt được sân nên đã được đóng lại. Bạn có thể tìm kèo mới.",
            )
        return self._retry_elo_sync(self._settle_finished_match(session))

    def _match_end(self, session: MatchmakingSession) -> datetime:
        _, match_end = self._zoned_interval(
            session.play_date,
            session.start_time,
            session.end_time,
            session.timezone,
        )
        return match_end

    def _match_has_ended(self, session: MatchmakingSession) -> bool:
        return datetime.now(timezone.utc) >= self._match_end(session)

    def _settle_finished_match(self, session: MatchmakingSession) -> MatchmakingSession:
        """Đóng vòng đời sau khi khung giờ chơi kết thúc.

        - Chưa ai check-in nhận sân: kèo không được ghi nhận, không tính ELO.
        - Đã check-in: mở cửa sổ nhập kết quả và nhắc cả hai, hết hạn thì tự chốt.
        """
        if session.status not in _RESULT_SETTLEMENT_STATUSES:
            return session
        now = datetime.now(timezone.utc)
        match_end = self._match_end(session)
        if now < match_end:
            return session
        offline = bool(session.proposal and session.proposal.offline_booking)
        if session.status == MatchSessionStatus.CONFIRMED and not offline:
            session.status = MatchSessionStatus.EXPIRED
            if session.proposal:
                session.proposal.status = MatchProposalStatus.EXPIRED
                session.proposal.cancel_reason = "NO_CHECK_IN"
            session = self.session_repository.save(session)
            self.notify_no_check_in(session)
            return session
        if session.result_deadline_at is None:
            session.result_prompt_sent_at = now
            session.result_deadline_at = match_end + timedelta(
                seconds=self.config.result_window_seconds
            )
            session = self.session_repository.save(session)
            self.notify_result_window_open(session)
        deadline = session.result_deadline_at
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        if now < deadline:
            return session
        return self._auto_settle_result(session)

    def _auto_settle_result(self, session: MatchmakingSession) -> MatchmakingSession:
        """Quá hạn nhập kết quả: một bên nhập thì bên đó được xử thắng, không bên
        nào nhập (hoặc hai bên không thống nhất được) thì tính hòa.

        Sân ngoài hệ thống không có check-in chứng minh trận đã diễn ra, nên chỉ ghi
        nhận khi hai bên nhập cùng tỷ số (đã chốt ở submit_result); quá hạn là bỏ."""
        if session.proposal and session.proposal.offline_booking:
            session.status = MatchSessionStatus.EXPIRED
            session.proposal.status = MatchProposalStatus.EXPIRED
            session.proposal.cancel_reason = "RESULT_NOT_CONFIRMED"
            session = self.session_repository.save(session)
            self._notify(
                session,
                "Kèo không được ghi nhận",
                "Hai bên chưa nhập cùng một tỷ số nên trận ở sân ngoài hệ thống không được tính ELO.",
            )
            return session
        claims = session.result_claims
        if len(claims) == 1:
            claim = claims[0]
            first_score, second_score = claim.participant_one_score, claim.participant_two_score
            winner_id = claim.winner_id
            submitter_side = self._side_id(session, claim.submitted_by)
            if winner_id is None or winner_id != submitter_side:
                # Bên vắng mặt không được hưởng lợi từ việc không phản hồi.
                winner_id = submitter_side
                first_id = session.participants[0].participant_id
                high, low = max(first_score, second_score), min(first_score, second_score)
                if high == low:
                    high, low = low + 1, low
                first_score, second_score = (
                    (high, low) if winner_id == first_id else (low, high)
                )
        else:
            first_score, second_score, winner_id = 0, 0, None
        try:
            settled = self._finalize_result(session, first_score, second_score, winner_id)
        except DownstreamServiceException:
            # Giữ nguyên phiên để thử chốt lại ở lần đọc kế tiếp.
            return session
        self.notify_result(settled, None)
        return settled

    def _attach_conversation(self, session: MatchmakingSession, creator_id: UUID) -> None:
        if not session.proposal or session.proposal.conversation_id:
            return
        try:
            session.proposal.conversation_id = self.social_client.create_match_conversation(
                session.session_id,
                list(dict.fromkeys(user_id for side in session.participants for user_id in side.user_ids())),
                creator_id,
                session.sport_type,
            )
        except (DownstreamServiceException, ValueError, KeyError):
            return

    def notify_match_found(
        self,
        session: MatchmakingSession,
        actor_id: UUID | None = None,
    ) -> None:
        self._notify(
            session,
            "Đã tìm thấy đối thủ",
            "GOAT SPORTS đã tìm thấy một đối thủ phù hợp. Hãy phản hồi trước khi kèo hết hạn.",
            skip_user=actor_id,
        )

    def notify_decision(
        self,
        session: MatchmakingSession,
        actor_id: UUID,
        decision: AcceptanceDecision,
    ) -> None:
        title = "Đối thủ đã đồng ý tham gia" if decision == AcceptanceDecision.ACCEPTED else "Đối thủ đã từ chối kèo"
        content = (
            "Kèo đã được cả hai người xác nhận. Bạn có thể mở trò chuyện và đặt sân."
            if session.status == MatchSessionStatus.ACCEPTED
            else "Mở phiên ghép kèo để xem trạng thái mới nhất."
        )
        self._notify(session, title, content, skip_side_of=actor_id)

    def notify_expired(self, session: MatchmakingSession) -> None:
        self._notify(
            session,
            "Kèo đã hết hạn phản hồi",
            "Đối thủ chưa phản hồi kịp thời gian quy định. Bạn có thể tìm đối thủ khác.",
        )

    def notify_lifecycle(
        self,
        session: MatchmakingSession,
        actor_id: UUID | None = None,
    ) -> None:
        cancel_messages = {
            "PLAYER_CANCELLED": ("Đặt sân đã bị hủy", "Người đặt cọc đã hủy đặt sân cho kèo này."),
            "PAYMENT_EXPIRED": ("Kèo đã bị hủy", "Đã hết thời gian thanh toán tiền cọc."),
            "PAYMENT_CANCELLED": ("Kèo đã bị hủy", "Thanh toán tiền cọc đã bị hủy."),
            "REFUNDED": ("Đặt sân đã bị hủy", "Đặt sân đã bị hủy và tiền cọc đã được hoàn lại."),
        }
        messages = {
            MatchSessionStatus.VENUE_SELECTED: ("Đã chọn sân thi đấu", "Người được chỉ định có thể tiếp tục đặt sân và thanh toán cọc."),
            MatchSessionStatus.BOOKING_PENDING: ("Đã tạo đơn đặt sân", "Người đặt cọc đang hoàn tất thanh toán."),
            MatchSessionStatus.CONFIRMED: ("Trận đấu đã được xác nhận", "Tiền cọc đã được ghi nhận. Vé QR check-in đã sẵn sàng."),
            MatchSessionStatus.CHECKED_IN: ("Đã check-in", "Sau trận đấu, hãy nhập kết quả để cập nhật ELO."),
            MatchSessionStatus.CANCELLED: ("Kèo đã bị hủy", "Booking đã bị hủy hoặc hết hạn thanh toán."),
        }
        cancel_reason = session.proposal.cancel_reason if session.proposal else None
        if session.proposal and session.proposal.offline_booking:
            messages[MatchSessionStatus.CONFIRMED] = (
                "Đã chốt sân thi đấu",
                "Người đặt đã liên hệ sân ngoài hệ thống. Sau trận, cả hai bên nhập cùng tỷ số để tính ELO.",
            )
            cancel_messages["PLAYER_CANCELLED"] = ("Kèo đã bị hủy", "Một bên đã hủy kèo.")
        message = (
            cancel_messages.get(cancel_reason or "")
            if session.status == MatchSessionStatus.CANCELLED
            else None
        ) or messages.get(session.status)
        if not message:
            return
        self._notify(session, message[0], message[1], skip_user=actor_id)

    def _notify_venue_search_updated(self, session_id: UUID) -> None:
        session = self.session_repository.find_by_id(session_id)
        if not session or not session.proposal:
            return
        search_status = session.proposal.venue_search_status
        title, content = {
            VenueSearchStatus.READY: (
                "Đã tìm thấy sân phù hợp",
                "GOAT SPORTS đã cập nhật các sân còn lịch trống cho kèo của bạn.",
            ),
            VenueSearchStatus.NO_AVAILABILITY: (
                "Chưa có sân phù hợp",
                "Chưa tìm thấy sân còn lịch trống trong phạm vi đã chọn.",
            ),
            VenueSearchStatus.SERVICE_UNAVAILABLE: (
                "Chưa thể kiểm tra lịch sân",
                "Dịch vụ sân tạm thời chưa phản hồi. Bạn có thể kiểm tra lại sau.",
            ),
        }.get(search_status, (None, None))
        if not title or not content:
            return
        if search_status == VenueSearchStatus.READY and all(
            item.external for item in session.proposal.venue_options
        ):
            title, content = (
                "Gợi ý sân gần nhóm",
                "Chưa có sân đối tác còn trống. Người đặt hãy gọi một sân gợi ý để chốt giờ chơi.",
            )
        self._notify(session, title, content)

    def notify_no_check_in(self, session: MatchmakingSession) -> None:
        self._notify(
            session,
            "Kèo không được ghi nhận",
            "Hai bên chưa check-in nhận sân nên trận đấu không được tính kết quả và ELO.",
        )

    def notify_result_window_open(self, session: MatchmakingSession) -> None:
        hours = max(1, self.config.result_window_seconds // 3600)
        if session.proposal and session.proposal.offline_booking:
            content = (
                f"Hãy nhập tỷ số trong {hours} giờ. Trận ở sân ngoài hệ thống chỉ tính ELO "
                "khi cả hai bên nhập cùng một tỷ số."
            )
        else:
            content = (
                f"Hãy nhập tỷ số trong {hours} giờ. Quá hạn mà bạn không nhập, "
                "đối thủ đã nhập sẽ được xử thắng."
            )
        self._notify(session, "Trận đấu đã kết thúc", content)

    def notify_result(self, session: MatchmakingSession, actor_id: UUID | None) -> None:
        messages = {
            MatchSessionStatus.RESULT_PENDING: (
                "Đối thủ đã nhập kết quả",
                "Hãy nhập kết quả của bạn để hoàn tất trận đấu và cập nhật ELO.",
            ),
            MatchSessionStatus.DISPUTED: (
                "Kết quả chưa trùng khớp",
                "Hai bên cần kiểm tra và gửi lại cùng một tỷ số.",
            ),
            MatchSessionStatus.COMPLETED: (
                "Kết quả đã được xác nhận",
                "ELO đã được cập nhật. Bạn có thể đánh giá đối thủ.",
            ),
        }
        message = messages.get(session.status)
        if not message:
            return
        if session.status != MatchSessionStatus.COMPLETED or not session.result:
            skip = actor_id if session.status == MatchSessionStatus.RESULT_PENDING else None
            self._notify(session, message[0], message[1], skip_side_of=skip)
            return
        for side in session.participants:
            for user_id in side.user_ids():
                title, body = self._rated_result_message(session, side, user_id)
                try:
                    self.notification_client.notify_matchmaking(user_id, session.session_id, title, body)
                except DownstreamServiceException:
                    continue

    def _notify(
        self,
        session: MatchmakingSession,
        title: str,
        content: str,
        *,
        skip_user: UUID | None = None,
        skip_side_of: UUID | None = None,
    ) -> None:
        """Gửi cho mọi người của cả hai bên; bỏ qua người vừa thao tác hoặc cả bên của họ."""
        skip_side = self._side_id(session, skip_side_of) if skip_side_of else None
        for side in session.participants:
            if skip_side is not None and side.participant_id == skip_side:
                continue
            for user_id in side.user_ids():
                if user_id == skip_user:
                    continue
                try:
                    self.notification_client.notify_matchmaking(user_id, session.session_id, title, content)
                except DownstreamServiceException:
                    continue

    @classmethod
    def _rated_result_message(
        cls,
        session: MatchmakingSession,
        side: MatchParticipantProfile,
        user_id: UUID,
    ) -> tuple[str, str]:
        """Thông báo kết quả cho từng người: thắng/thua, tỷ số của mình, ELO được cộng hay trừ bao nhiêu."""
        result = session.result
        is_first = side.participant_id == session.participants[0].participant_id
        mine = result.participant_one_score if is_first else result.participant_two_score
        theirs = result.participant_two_score if is_first else result.participant_one_score
        opponent = session.participants[1 if is_first else 0]
        outcome = "Hòa" if result.winner_id is None else ("Thắng" if result.winner_id == side.participant_id else "Thua")
        rated = cls._rated_members(side)
        # CLB: ELO thuộc về CLB nên mọi quản lý cùng nhận một con số.
        member = next((item for item in rated if item.user_id == user_id), rated[0])
        new_elo = result.elo_updates.get(str(member.user_id))
        if new_elo is None or not result.elo_applied:
            return (
                f"{outcome} {mine}–{theirs}",
                f"Kết quả trận với {opponent.name} đã được xác nhận. ELO sẽ được cập nhật trong ít phút.",
            )
        delta = new_elo - member.elo_rating
        change = f"−{abs(delta)}" if delta < 0 else f"+{delta}"
        whose = f"ELO của CLB {side.name}" if isinstance(side, ClubMatchProfile) else "ELO"
        return (
            f"{outcome} {mine}–{theirs} · {change} ELO",
            f"Trận với {opponent.name}: {whose} {member.elo_rating} → {new_elo}. Bạn có thể đánh giá đối thủ.",
        )

    def _require_session(self, session_id: UUID) -> MatchmakingSession:
        session = self.session_repository.find_by_id(session_id)
        if session is None:
            raise MatchmakingNotFoundError("Không tìm thấy phiên ghép kèo.")
        return session

    @staticmethod
    def _require_designated_booker(session: MatchmakingSession, participant_id: UUID) -> None:
        if not session.proposal or participant_id != session.proposal.designated_booker_id:
            raise MatchmakingForbiddenError("Chỉ người được chỉ định mới có thể đặt sân cho trận này.")

    @staticmethod
    def _calculate_elo_updates(
        session: MatchmakingSession,
        winner_id: UUID | None,
    ) -> dict[str, int]:
        first, second = session.participants
        actual_first = 0.5 if winner_id is None else float(winner_id == first.participant_id)
        updates: dict[str, int] = {}
        for side, opponent, actual in ((first, second, actual_first), (second, first, 1.0 - actual_first)):
            for member in MatchmakingService._rated_members(side):
                updates[str(member.user_id)] = calculate_new_rating(
                    session.sport_type,
                    member.elo_rating,
                    opponent.elo_rating,
                    member.match_count,
                    actual,
                )
        return updates

    @staticmethod
    def _rated_members(side: MatchParticipantProfile) -> list[SideMember]:
        """Những ai được tính ELO: từng người của bên người chơi, hoặc chính CLB."""
        if side.members and not isinstance(side, ClubMatchProfile):
            return side.members
        return [
            SideMember(
                user_id=side.participant_id,
                name=side.name,
                elo_rating=side.elo_rating,
                match_count=getattr(side, "match_count", 0),
            )
        ]

    @staticmethod
    def _side_id(session: MatchmakingSession, user_id: UUID) -> UUID | None:
        for side in session.participants:
            if user_id == side.participant_id or user_id in side.user_ids():
                return side.participant_id
        return None

    @staticmethod
    def _require_participant(
        session: MatchmakingSession,
        participant_id: UUID,
    ) -> MatchParticipantProfile:
        for profile in session.participants:
            if participant_id in profile.user_ids():
                return profile
        raise MatchmakingForbiddenError("Bạn không thuộc phiên ghép kèo này.")
