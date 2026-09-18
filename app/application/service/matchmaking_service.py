from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.application.dto.request.models import (
    MatchBookingRegistrationRequest,
    MatchmakingQueueRequest,
    MatchProposalUpdateRequest,
    MatchResultRequest,
    MatchVenueSelectionRequest,
    OpponentFeedbackRequest,
)
from app.application.dto.response.models import MatchCandidate
from app.application.exception import (
    ConflictException,
    DownstreamServiceException,
    ForbiddenException,
    ResourceNotFoundException,
)
from app.application.port.client import (
    AuthServiceClient,
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
from app.application.service.matchmaking_scoring import compatibility_breakdown, overlapping_window
from app.application.service.model_registry import get_or_create_active_model
from app.application.service.service_config import MatchmakingConfig
from app.domain.entity.models import (
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


MATCHMAKING_FLOW_STATUSES = {
    MatchSessionStatus.PROPOSED,
    MatchSessionStatus.ACCEPTED_BY_ONE,
    MatchSessionStatus.ACCEPTED,
    MatchSessionStatus.VENUE_SELECTED,
    MatchSessionStatus.BOOKING_PENDING,
}

SCHEDULE_RESERVING_STATUSES = MATCHMAKING_FLOW_STATUSES | {
    MatchSessionStatus.CONFIRMED,
    MatchSessionStatus.CHECKED_IN,
    MatchSessionStatus.RESULT_PENDING,
    MatchSessionStatus.DISPUTED,
}

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
    ) -> None:
        self.queue_repository = queue_repository
        self.session_repository = session_repository
        self.document_repository = document_repository
        self.venue_client = venue_client
        self.social_client = social_client
        self.notification_client = notification_client
        self.config = config
        self.auth_client = auth_client

    def join_queue(
        self,
        participant_id: UUID,
        request: MatchmakingQueueRequest,
    ) -> MatchmakingSession | None:
        active_session = self._active_session(participant_id)
        if active_session:
            return active_session
        conflict = self._schedule_conflict(participant_id, request)
        if conflict:
            raise MatchmakingConflictError(
                "Lịch tìm đối thủ trùng hoặc cách dưới 60 phút với trận "
                f"{conflict.sport_type.value} lúc "
                f"{conflict.start_time.strftime('%H:%M')}–{conflict.end_time.strftime('%H:%M')} "
                f"ngày {conflict.play_date.strftime('%d/%m/%Y')}."
            )

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
            selection_mode=request.selection_mode,
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
            if request.selection_mode == MatchSelectionMode.AI:
                return self._match_best_candidate(entry, profile)
            return None

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
            return True

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
        for candidate_entry in self.queue_repository.queued_entries(self.config.scan_limit):
            if candidate_entry.queue_entry_id == current_entry.queue_entry_id:
                continue
            candidate_profile = self.queue_repository.find_profile(candidate_entry.profile_id)
            if candidate_profile is None or candidate_profile.participant_id == current_profile.participant_id:
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
            window = overlapping_window(current_entry.preference, candidate_entry.preference)
            assert window is not None
            ranked.append((candidate_entry, candidate_profile, breakdown, distance, elo_difference, window))
        ranked.sort(key=lambda item: item[2].total_score, reverse=True)
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
        except (DownstreamServiceException, ValueError, KeyError):
            self._set_venue_search_status(session_id, VenueSearchStatus.SERVICE_UNAVAILABLE)
            self._notify_venue_search_updated(session_id)
            return
        if not recommendations:
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
                )
                for item in recommendations[:3]
                if item.venue_court_id is not None
            ]
            current.proposal.venue_options = options
            current.proposal.venue_search_status = (
                VenueSearchStatus.READY if options else VenueSearchStatus.NO_AVAILABILITY
            )
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
            }:
                raise MatchmakingConflictError("Kèo không thể hủy ở trạng thái hiện tại.")
            session.status = MatchSessionStatus.CANCELLED
            if session.proposal:
                session.proposal.status = MatchProposalStatus.CANCELLED
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
            session.proposal.venue_court_id = option.venue_court_id
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
    ) -> MatchmakingSession:
        """Apply an idempotent booking event received from the trusted venue-service."""
        with self.queue_repository.lock():
            session = self._require_session(session_id)
            if not session.proposal:
                raise MatchmakingConflictError("Phiên ghép trận chưa có đề xuất booking.")
            if session.proposal.booking_id and session.proposal.booking_id != booking_id:
                raise MatchmakingConflictError("Booking không thuộc phiên ghép trận này.")
            session.proposal.booking_id = booking_id
            normalized = booking_status.upper()
            now = datetime.now(timezone.utc)
            if normalized in {"PENDING_PAYMENT", "PENDING"}:
                session.proposal.status = MatchProposalStatus.BOOKING_PENDING
                session.proposal.booking_created_at = session.proposal.booking_created_at or now
                session.status = MatchSessionStatus.BOOKING_PENDING
            elif normalized in {"CONFIRMED", "PAID"}:
                session.proposal.status = MatchProposalStatus.BOOKED
                session.proposal.booked_at = session.proposal.booked_at or now
                session.status = MatchSessionStatus.CONFIRMED
                session.confirmed_at = session.confirmed_at or now
            elif normalized in {"CHECKED_IN", "COMPLETED"}:
                session.proposal.status = MatchProposalStatus.CHECKED_IN
                session.proposal.checked_in_at = session.proposal.checked_in_at or now
                session.status = MatchSessionStatus.CHECKED_IN
                session.checked_in_at = session.checked_in_at or now
            elif normalized in {"CANCELLED", "EXPIRED", "REFUNDED"}:
                session.proposal.status = MatchProposalStatus.CANCELLED
                session.status = MatchSessionStatus.CANCELLED
            else:
                raise MatchmakingConflictError("Trạng thái booking không được hỗ trợ.")
            return self.session_repository.save(session)

    def _sync_booking(self, session: MatchmakingSession) -> MatchmakingSession:
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
            session = self._require_session(session_id)
            self._require_participant(session, participant_id)
            if session.status not in {
                MatchSessionStatus.CHECKED_IN,
                MatchSessionStatus.RESULT_PENDING,
                MatchSessionStatus.DISPUTED,
            }:
                raise MatchmakingConflictError("Chỉ được nhập kết quả sau khi trận đấu đã check-in.")
            first_id = session.participants[0].participant_id
            if participant_id == first_id:
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
            session.result_claims = [item for item in session.result_claims if item.submitted_by != participant_id]
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
            elo_updates = self._calculate_elo_updates(session, winner_id)
            if self.auth_client:
                first, second = session.participants
                first_outcome = "DRAW" if winner_id is None else ("WIN" if winner_id == first.participant_id else "LOSS")
                second_outcome = "DRAW" if winner_id is None else ("WIN" if winner_id == second.participant_id else "LOSS")
                try:
                    elo_updates = {
                        str(first.participant_id): self.auth_client.apply_match_result(
                            session.session_id, first.participant_id, session.sport_type,
                            second.elo_rating, first_outcome,
                        ),
                        str(second.participant_id): self.auth_client.apply_match_result(
                            session.session_id, second.participant_id, session.sport_type,
                            first.elo_rating, second_outcome,
                        ),
                    }
                except DownstreamServiceException:
                    session.status = MatchSessionStatus.RESULT_PENDING
                    self.session_repository.save(session)
                    raise
            now = datetime.now(timezone.utc)
            session.result = MatchResult(
                participantOneScore=first_score,
                participantTwoScore=second_score,
                winnerId=winner_id,
                confirmedAt=now,
                eloUpdates=elo_updates,
            )
            session.status = MatchSessionStatus.COMPLETED
            session.completed_at = now
            if session.proposal:
                session.proposal.status = MatchProposalStatus.COMPLETED
            return self.session_repository.save(session)

    def submit_feedback(
        self,
        session_id: UUID,
        participant_id: UUID,
        request: OpponentFeedbackRequest,
    ) -> MatchmakingSession:
        with self.queue_repository.lock():
            session = self._require_session(session_id)
            self._require_participant(session, participant_id)
            if session.status != MatchSessionStatus.COMPLETED:
                raise MatchmakingConflictError("Chỉ được đánh giá sau khi kết quả đã được xác nhận.")
            if any(item.reviewer_id == participant_id for item in session.feedback):
                raise MatchmakingConflictError("Bạn đã đánh giá đối thủ của trận này.")
            reviewee = next(item for item in session.participants if item.participant_id != participant_id)
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

    def _active_session(self, participant_id: UUID) -> MatchmakingSession | None:
        session = self.session_repository.find_active_by_participant(participant_id)
        if not session:
            return None
        session = self._expire_if_needed(session)
        return session if session.status in SCHEDULE_RESERVING_STATUSES else None

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

    def notify_match_found(
        self,
        session: MatchmakingSession,
        actor_id: UUID | None = None,
    ) -> None:
        for profile in session.participants:
            if profile.participant_id == actor_id:
                continue
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

    def notify_lifecycle(
        self,
        session: MatchmakingSession,
        actor_id: UUID | None = None,
    ) -> None:
        messages = {
            MatchSessionStatus.VENUE_SELECTED: ("Đã chọn sân thi đấu", "Người được chỉ định có thể tiếp tục đặt sân và thanh toán cọc."),
            MatchSessionStatus.BOOKING_PENDING: ("Đã tạo đơn đặt sân", "Người đặt cọc đang hoàn tất thanh toán."),
            MatchSessionStatus.CONFIRMED: ("Trận đấu đã được xác nhận", "Tiền cọc đã được ghi nhận. Vé QR check-in đã sẵn sàng."),
            MatchSessionStatus.CHECKED_IN: ("Đã check-in", "Sau trận đấu, hãy nhập kết quả để cập nhật ELO."),
            MatchSessionStatus.CANCELLED: ("Kèo đã bị hủy", "Booking đã bị hủy hoặc hết hạn thanh toán."),
        }
        message = messages.get(session.status)
        if not message:
            return
        for profile in session.participants:
            if profile.participant_id == actor_id:
                continue
            try:
                self.notification_client.notify_matchmaking(
                    profile.participant_id, session.session_id, message[0], message[1]
                )
            except DownstreamServiceException:
                continue

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
        for profile in session.participants:
            try:
                self.notification_client.notify_matchmaking(
                    profile.participant_id,
                    session.session_id,
                    title,
                    content,
                )
            except DownstreamServiceException:
                continue

    def notify_result(self, session: MatchmakingSession, actor_id: UUID) -> None:
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
        recipients = (
            session.participants
            if session.status in {MatchSessionStatus.DISPUTED, MatchSessionStatus.COMPLETED}
            else [item for item in session.participants if item.participant_id != actor_id]
        )
        for profile in recipients:
            try:
                self.notification_client.notify_matchmaking(
                    profile.participant_id, session.session_id, message[0], message[1]
                )
            except DownstreamServiceException:
                continue

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
        first_new = calculate_new_rating(
            session.sport_type,
            first.elo_rating,
            second.elo_rating,
            getattr(first, "match_count", 0),
            actual_first,
        )
        second_new = calculate_new_rating(
            session.sport_type,
            second.elo_rating,
            first.elo_rating,
            getattr(second, "match_count", 0),
            1.0 - actual_first,
        )
        return {str(first.participant_id): first_new, str(second.participant_id): second_new}

    @staticmethod
    def _require_participant(
        session: MatchmakingSession,
        participant_id: UUID,
    ) -> MatchParticipantProfile:
        for profile in session.participants:
            if profile.participant_id == participant_id:
                return profile
        raise MatchmakingForbiddenError("Bạn không thuộc phiên ghép kèo này.")
