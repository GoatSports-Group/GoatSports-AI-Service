from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

from app.application.dto.request.models import (
    MatchmakingQueueRequest,
    MatchProposalUpdateRequest,
)
from app.application.exception import (
    ConflictException,
    DownstreamServiceException,
    ForbiddenException,
    ResourceNotFoundException,
)
from app.application.port.client import (
    NotificationServiceClient,
    SocialServiceClient,
    VenueServiceClient,
)
from app.application.port.persistence import (
    AiDocumentRepository,
    MatchmakingQueueRepository,
    MatchmakingSessionRepository,
)
from app.application.service.matchmaking_scoring import compatibility_score, overlapping_window
from app.application.service.model_registry import get_or_create_active_model
from app.application.service.service_config import MatchmakingConfig
from app.domain.entity.models import (
    MatchAcceptance,
    MatchmakingQueueEntry,
    MatchmakingSession,
    MatchParticipantProfile,
    MatchPreference,
    MatchProposal,
    PlayerProfile,
)
from app.domain.enums import (
    AcceptanceDecision,
    AiCapability,
    MatchProposalStatus,
    MatchSessionStatus,
    QueueStatus,
)


class MatchmakingError(Exception):
    pass


class MatchmakingNotFoundError(ResourceNotFoundException, MatchmakingError):
    pass


class MatchmakingForbiddenError(ForbiddenException, MatchmakingError):
    pass


class MatchmakingConflictError(ConflictException, MatchmakingError):
    pass


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
    ) -> None:
        self.queue_repository = queue_repository
        self.session_repository = session_repository
        self.document_repository = document_repository
        self.venue_client = venue_client
        self.social_client = social_client
        self.notification_client = notification_client
        self.config = config

    def join_queue(
        self,
        participant_id: UUID,
        request: MatchmakingQueueRequest,
    ) -> MatchmakingSession | None:
        active_session = self._active_session(participant_id)
        if active_session:
            return active_session

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=self.config.queue_ttl_seconds)
        profile = PlayerProfile(
            participant_id=participant_id,
            name=request.player_name,
            avatar_url=request.player_avatar,
            sport_type=request.sport_type,
            skill_level=request.skill_level,
            elo_rating=request.elo_rating,
            preferred_positions=request.preferred_positions,
            play_radius_km=request.max_distance_km,
            play_style=request.play_style,
            match_count=request.match_count,
            win_rate=request.win_rate,
            latitude=request.latitude,
            longitude=request.longitude,
        )
        preference = MatchPreference(
            sport_type=request.sport_type,
            play_date=request.play_date,
            start_time=request.start_time,
            end_time=request.end_time,
            timezone=request.timezone,
            max_elo_difference=request.max_elo_difference,
            max_distance_km=request.max_distance_km,
        )

        with self.queue_repository.lock():
            previous = self.queue_repository.find_by_participant(participant_id)
            if previous and previous.status == QueueStatus.QUEUED:
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
            return self._match_best_candidate(entry, profile)

    def get_status(self, participant_id: UUID) -> tuple[str, int | None, MatchmakingSession | None]:
        session = self._active_session(participant_id)
        if session:
            return session.status.value, None, session
        entry = self.queue_repository.find_by_participant(participant_id)
        if entry:
            return entry.status.value, self.queue_repository.queue_size(), None
        latest = self.session_repository.find_latest_by_participant(participant_id)
        if latest:
            return latest.status.value, None, latest
        return "NOT_IN_QUEUE", None, None

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
            return True

    def get_session(self, session_id: UUID, participant_id: UUID) -> MatchmakingSession:
        session = self._require_session(session_id)
        self._require_participant(session, participant_id)
        return self._expire_if_needed(session)

    def get_history(
        self,
        participant_id: UUID,
        limit: int = 5,
    ) -> list[MatchmakingSession]:
        sessions = self.session_repository.find_by_participant(
            participant_id,
            max(1, min(limit, 20)),
        )
        return [self._expire_if_needed(session) for session in sessions]

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
        if participant_id != session.proposal.designated_booker_id:
            raise MatchmakingForbiddenError("Chỉ người được chỉ định mới có thể cập nhật đặt sân.")
        if request.venue_id is not None:
            session.proposal.venue_id = request.venue_id
        if request.venue_court_id is not None:
            session.proposal.venue_court_id = request.venue_court_id
        if request.booking_id is not None:
            session.proposal.booking_id = request.booking_id
        if request.status is not None:
            session.proposal.status = request.status
        return self.session_repository.save(session)

    def _match_best_candidate(
        self,
        current_entry: MatchmakingQueueEntry,
        current_profile: MatchParticipantProfile,
    ) -> MatchmakingSession | None:
        best: tuple[MatchmakingQueueEntry, MatchParticipantProfile, float, float, int] | None = None
        for candidate_entry in self.queue_repository.queued_entries(self.config.scan_limit):
            if candidate_entry.queue_entry_id == current_entry.queue_entry_id:
                continue
            candidate_profile = self.queue_repository.find_profile(candidate_entry.profile_id)
            if candidate_profile is None or candidate_profile.participant_id == current_profile.participant_id:
                continue
            result = compatibility_score(
                current_profile,
                current_entry.preference,
                candidate_profile,
                candidate_entry.preference,
            )
            if result is None or result[0] < self.config.minimum_score:
                continue
            if best is None or result[0] > best[2]:
                best = (candidate_entry, candidate_profile, *result)

        if best is None:
            return None
        candidate_entry, candidate_profile, score, distance, elo_difference = best
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
            compatibility_score=score,
            distance_km=distance,
            elo_difference=elo_difference,
            model_version_id=model.model_version_id,
            matched_at=now,
            expires_at=expires_at,
            participants=[current_profile, candidate_profile],
            proposal=MatchProposal(
                designated_booker_id=current_profile.participant_id,
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
        if not session or not session.proposal or session.proposal.venue_id:
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
            recommendations, venue_court_id = self.venue_client.recommend(
                latitude,
                longitude,
                session.sport_type,
                radius_km,
                session.play_date,
                session.start_time,
                session.end_time,
                max_results=3,
            )
        except (DownstreamServiceException, ValueError, KeyError):
            return
        if not recommendations:
            return
        with self.queue_repository.lock():
            current = self.session_repository.find_by_id(session_id)
            if not current or not current.proposal or current.proposal.venue_id:
                return
            current.proposal.venue_id = recommendations[0].venue_id
            current.proposal.venue_court_id = venue_court_id
            self.session_repository.save(current)

    def _active_session(self, participant_id: UUID) -> MatchmakingSession | None:
        session = self.session_repository.find_active_by_participant(participant_id)
        if not session:
            return None
        session = self._expire_if_needed(session)
        return session if session.status in {
            MatchSessionStatus.PROPOSED,
            MatchSessionStatus.ACCEPTED_BY_ONE,
        } else None

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
            self.session_repository.save(session)
        return session

    def _attach_conversation(self, session: MatchmakingSession, creator_id: UUID) -> None:
        if not session.proposal or session.proposal.conversation_id:
            return
        try:
            session.proposal.conversation_id = self.social_client.create_match_conversation(
                session.session_id,
                [profile.participant_id for profile in session.participants],
                creator_id,
                session.sport_type,
            )
        except (DownstreamServiceException, ValueError, KeyError):
            return

    def notify_match_found(self, session: MatchmakingSession) -> None:
        for profile in session.participants:
            try:
                self.notification_client.notify_matchmaking(
                    profile.participant_id,
                    session.session_id,
                    "Đã tìm thấy đối thủ",
                    "GOAT SPORTS đã tìm thấy một đối thủ phù hợp. Hãy phản hồi trước khi kèo hết hạn.",
                )
            except DownstreamServiceException:
                continue

    def notify_decision(
        self,
        session: MatchmakingSession,
        actor_id: UUID,
        decision: AcceptanceDecision,
    ) -> None:
        recipients = [item for item in session.participants if item.participant_id != actor_id]
        title = "Đối thủ đã đồng ý tham gia" if decision == AcceptanceDecision.ACCEPTED else "Đối thủ đã từ chối kèo"
        content = (
            "Kèo đã được cả hai người xác nhận. Bạn có thể mở trò chuyện và đặt sân."
            if session.status == MatchSessionStatus.ACCEPTED
            else "Mở phiên ghép kèo để xem trạng thái mới nhất."
        )
        for profile in recipients:
            try:
                self.notification_client.notify_matchmaking(
                    profile.participant_id,
                    session.session_id,
                    title,
                    content,
                )
            except DownstreamServiceException:
                continue

    def _require_session(self, session_id: UUID) -> MatchmakingSession:
        session = self.session_repository.find_by_id(session_id)
        if session is None:
            raise MatchmakingNotFoundError("Không tìm thấy phiên ghép kèo.")
        return session

    @staticmethod
    def _require_participant(
        session: MatchmakingSession,
        participant_id: UUID,
    ) -> MatchParticipantProfile:
        for profile in session.participants:
            if profile.participant_id == participant_id:
                return profile
        raise MatchmakingForbiddenError("Bạn không thuộc phiên ghép kèo này.")
