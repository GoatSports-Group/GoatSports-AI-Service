import unittest
from contextlib import nullcontext
from datetime import datetime, time, timedelta, timezone
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from app.application.dto.request.models import (
    MatchmakingQueueRequest,
    MatchResultRequest,
    MatchVenueSelectionRequest,
)
from app.application.dto.response.models import VenueRecommendation
from app.application.exception import DownstreamServiceException, ForbiddenException
from app.application.service.matchmaking_service import MatchmakingConflictError, MatchmakingService
from app.application.service.service_config import MatchmakingConfig
from app.domain.entity.models import ClubMatchProfile, MatchProposal, SideMember
from app.domain.enums import AcceptanceDecision, MatchSessionStatus, QueueStatus, VenueSearchStatus


class FakeQueueRepository:
    def __init__(self) -> None:
        self.entries = {}
        self.profiles = {}
        self.participants = {}

    def lock(self):
        return nullcontext()

    def save(self, entry, profile):
        self.entries[entry.queue_entry_id] = entry
        self.profiles[entry.profile_id] = profile
        for key_id in profile.queue_keys():
            self.participants[key_id] = entry.queue_entry_id
        return entry

    def find_by_participant(self, participant_id):
        entry_id = self.participants.get(participant_id)
        return self.entries.get(entry_id)

    def find_profile(self, profile_id):
        return self.profiles.get(profile_id)

    def queued_entries(self, limit):
        return [entry for entry in self.entries.values() if entry.status == QueueStatus.QUEUED][:limit]

    def queue_size(self):
        return len(self.queued_entries(10_000))


def _on_side(session, user_id):
    return any(user_id == profile.participant_id or user_id in profile.user_ids() for profile in session.participants)


class FakeSessionRepository:
    def __init__(self) -> None:
        self.sessions = {}

    def save(self, session):
        self.sessions[session.session_id] = session.model_copy(deep=True)
        return session

    def find_by_id(self, session_id):
        session = self.sessions.get(session_id)
        return session.model_copy(deep=True) if session else None

    def find_active_by_participant(self, participant_id):
        for session in reversed(list(self.sessions.values())):
            if session.status in {MatchSessionStatus.PROPOSED, MatchSessionStatus.ACCEPTED_BY_ONE, MatchSessionStatus.ACCEPTED} and _on_side(
                session, participant_id
            ):
                return session.model_copy(deep=True)
        return None

    def find_latest_by_participant(self, participant_id):
        for session in reversed(list(self.sessions.values())):
            if _on_side(session, participant_id):
                return session.model_copy(deep=True)
        return None

    def find_by_participant(self, participant_id, limit, offset=0):
        return [
            session.model_copy(deep=True)
            for session in reversed(list(self.sessions.values()))
            if _on_side(session, participant_id)
        ][offset:offset + limit]


class FakeDocumentRepository:
    def __init__(self):
        self.model = None

    def find_active_model(self, _):
        return self.model

    def save_model_version(self, model):
        self.model = model
        return model


class FakeVenueClient:
    def __init__(self):
        self.recommendations = []

    def recommend(self, *_, **__):
        return self.recommendations, None

    def get_matchmaking_booking(self, *_):
        return None


class FakeSocialClient:
    conversation_id = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")

    def __init__(self) -> None:
        self.friendships: set[frozenset] = set()
        self.conversation_members = []

    def create_match_conversation(self, session_id, participant_ids, creator_id, sport_type):
        self.conversation_members = participant_ids
        return self.conversation_id

    def friend_ids(self, user_id):
        return {other for pair in self.friendships if user_id in pair for other in pair if other != user_id}


class FakeNotificationClient:
    def __init__(self) -> None:
        self.notifications = []

    def notify_matchmaking(self, receiver_id, session_id, title, content):
        self.notifications.append((receiver_id, session_id, title, content))


class FakeAuthClient:
    def __init__(self) -> None:
        self.available = True
        self.calls = []

    def apply_match_result(self, session_id, user_id, sport_type, opponent_elo, outcome, player_elo=None,
                           opponent_name=None, my_score=None, opponent_score=None):
        if not self.available:
            raise DownstreamServiceException("Auth service is unavailable")
        self.calls.append((session_id, user_id, outcome))
        self.history = getattr(self, "history", {})
        self.history[user_id] = (opponent_name, my_score, opponent_score)
        self.opponent_elos = getattr(self, "opponent_elos", {})
        self.opponent_elos[user_id] = opponent_elo
        return 1234

    def teammate(self, user_id, sport_type):
        return SideMember(user_id=user_id, name=f"Bạn {str(user_id)[:4]}", elo_rating=self.partner_elo)

    partner_elo = 1300


class FakeClubClient:
    def __init__(self) -> None:
        self.clubs = {}
        self.results = []

    def matchmaking_profile(self, club_id, play_format, latitude, longitude):
        club = self.clubs.get(club_id)
        if club is None:
            return None
        return ClubMatchProfile(
            participant_id=club_id, name=club["name"], sport_type=club.get("sport", "FOOTBALL"), elo_rating=club["elo"],
            latitude=latitude, longitude=longitude, active_member_count=club["members"],
            preferred_format=play_format, play_format=play_format, manager_ids=club["managers"],
            member_ids=club.get("member_ids", []),
        )

    def apply_friendly_result(self, session_id, sport_type, play_format, first, second):
        self.results.append((session_id, play_format, first, second))
        won = first[1] > second[1]
        return {str(first[0]): 1216 if won else 1184, str(second[0]): 1184 if won else 1216}


class MatchmakingServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.queue = FakeQueueRepository()
        self.sessions = FakeSessionRepository()
        self.social = FakeSocialClient()
        self.notifications = FakeNotificationClient()
        self.venue = FakeVenueClient()
        self.auth = FakeAuthClient()
        self.service = MatchmakingService(
            self.queue,
            self.sessions,
            FakeDocumentRepository(),
            self.venue,
            self.social,
            self.notifications,
            MatchmakingConfig(
                queue_ttl_seconds=900,
                session_ttl_seconds=600,
                minimum_score=50,
                scan_limit=200,
            ),
            self.auth,
        )

    def test_matches_two_compatible_players_and_persists_session(self):
        first_id, second_id = uuid4(), uuid4()
        first = self.service.join_queue(first_id, self.request("Người chơi A", 10.77, 106.67, 1200))
        second = self.service.join_queue(second_id, self.request("Người chơi B", 10.78, 106.68, 1260))

        self.assertIsNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(2, len(second.participants))
        self.assertEqual(MatchSessionStatus.PROPOSED, second.status)
        self.assertEqual(QueueStatus.MATCHED, self.queue.find_by_participant(first_id).status)
        self.assertIn(second.session_id, self.sessions.sessions)
        self.service.notify_match_found(second)
        self.assertEqual(2, len(self.notifications.notifications))
        self.notifications.notifications.clear()
        self.service.notify_match_found(second, second_id)
        self.assertEqual([first_id], [item[0] for item in self.notifications.notifications])

    def test_accepts_once_per_participant_and_creates_conversation_after_both_accept(self):
        first_id, second_id = uuid4(), uuid4()
        self.service.join_queue(first_id, self.request("Người chơi A", 10.77, 106.67, 1200))
        session = self.service.join_queue(second_id, self.request("Người chơi B", 10.78, 106.68, 1250))

        first_acceptance = self.service.decide(session.session_id, first_id, AcceptanceDecision.ACCEPTED)
        accepted = self.service.decide(session.session_id, second_id, AcceptanceDecision.ACCEPTED)

        self.assertEqual(MatchSessionStatus.ACCEPTED_BY_ONE, first_acceptance.status)
        self.assertEqual(MatchSessionStatus.ACCEPTED, accepted.status)
        self.assertEqual(self.social.conversation_id, accepted.proposal.conversation_id)

    def test_returns_recent_matchmaking_history_for_current_participant(self):
        first_id, second_id = uuid4(), uuid4()
        self.service.join_queue(first_id, self.request("Người chơi A", 10.77, 106.67, 1200))
        session = self.service.join_queue(second_id, self.request("Người chơi B", 10.78, 106.68, 1250))

        history = self.service.get_history(first_id, 5)

        self.assertEqual(1, len(history))
        self.assertEqual(session.session_id, history[0].session_id)

    def test_rejects_a_new_search_that_conflicts_with_confirmed_match_buffer(self):
        first_id, second_id = uuid4(), uuid4()
        self.service.join_queue(first_id, self.request("Người chơi A", 10.77, 106.67, 1200))
        session = self.service.join_queue(second_id, self.request("Người chơi B", 10.78, 106.68, 1250))
        session.status = MatchSessionStatus.CONFIRMED
        self.sessions.save(session)

        conflicting = self.request("Người chơi A", 10.77, 106.67, 1200)
        conflicting.start_time = time(20, 30)
        conflicting.end_time = time(22, 0)

        with self.assertRaises(MatchmakingConflictError):
            self.service.join_queue(first_id, conflicting)

    def test_allows_a_new_search_outside_confirmed_match_buffer(self):
        first_id, second_id = uuid4(), uuid4()
        self.service.join_queue(first_id, self.request("Người chơi A", 10.77, 106.67, 1200))
        session = self.service.join_queue(second_id, self.request("Người chơi B", 10.78, 106.68, 1250))
        session.status = MatchSessionStatus.CONFIRMED
        self.sessions.save(session)

        non_conflicting = self.request("Người chơi A", 10.77, 106.67, 1200)
        non_conflicting.start_time = time(21, 0)
        non_conflicting.end_time = time(23, 0)

        self.assertIsNone(self.service.join_queue(first_id, non_conflicting))

    def _booking_pending_session(self, booking_id):
        first_id, second_id = uuid4(), uuid4()
        self.service.join_queue(first_id, self.request("Người chơi A", 10.77, 106.67, 1200))
        session = self.service.join_queue(second_id, self.request("Người chơi B", 10.78, 106.68, 1250))
        session.proposal = MatchProposal(
            designated_booker_id=first_id,
            venue_court_id=uuid4(),
            booking_id=booking_id,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        session.status = MatchSessionStatus.BOOKING_PENDING
        self.sessions.save(session)
        return session

    def test_venue_lifecycle_sync_confirms_booking_and_notifies_both_participants(self):
        booking_id = uuid4()
        session = self._booking_pending_session(booking_id)

        updated = self.service.sync_lifecycle_from_venue(session.session_id, booking_id, "CONFIRMED")

        self.assertEqual(MatchSessionStatus.CONFIRMED, updated.status)
        self.assertEqual(
            {participant.participant_id for participant in updated.participants},
            {receiver for receiver, *_ in self.notifications.notifications},
        )

    def test_venue_lifecycle_sync_ignores_stale_out_of_order_event(self):
        booking_id = uuid4()
        session = self._booking_pending_session(booking_id)
        self.service.sync_lifecycle_from_venue(session.session_id, booking_id, "CONFIRMED")
        self.service.sync_lifecycle_from_venue(session.session_id, booking_id, "CHECKED_IN")
        self.notifications.notifications.clear()

        stale = self.service.sync_lifecycle_from_venue(session.session_id, booking_id, "CONFIRMED")

        self.assertEqual(MatchSessionStatus.CHECKED_IN, stale.status)
        self.assertEqual([], self.notifications.notifications)

    def test_venue_lifecycle_sync_records_cancel_reason_and_specific_message(self):
        booking_id = uuid4()
        session = self._booking_pending_session(booking_id)

        cancelled = self.service.sync_lifecycle_from_venue(
            session.session_id, booking_id, "EXPIRED", "PAYMENT_EXPIRED"
        )

        self.assertEqual(MatchSessionStatus.CANCELLED, cancelled.status)
        self.assertEqual("PAYMENT_EXPIRED", cancelled.proposal.cancel_reason)
        self.assertTrue(self.notifications.notifications)
        self.assertEqual(
            "Đã hết thời gian thanh toán tiền cọc.",
            self.notifications.notifications[0][3],
        )

    def test_venue_lifecycle_sync_cannot_cancel_after_check_in(self):
        booking_id = uuid4()
        session = self._booking_pending_session(booking_id)
        self.service.sync_lifecycle_from_venue(session.session_id, booking_id, "CONFIRMED")
        self.service.sync_lifecycle_from_venue(session.session_id, booking_id, "CHECKED_IN")
        self.notifications.notifications.clear()

        result = self.service.sync_lifecycle_from_venue(session.session_id, booking_id, "CANCELLED")

        self.assertEqual(MatchSessionStatus.CHECKED_IN, result.status)
        self.assertEqual([], self.notifications.notifications)

    def test_manual_mode_returns_ranked_candidates_and_allows_selection(self):
        first_id, second_id = uuid4(), uuid4()
        self.service.join_queue(first_id, self.request("Người chơi A", 10.77, 106.67, 1200))
        manual = self.request("Người chơi B", 10.78, 106.68, 1250)
        manual.selection_mode = "MANUAL"
        self.assertIsNone(self.service.join_queue(second_id, manual))

        candidates = self.service.get_candidates(second_id, 5)
        self.assertEqual(1, len(candidates))
        self.assertEqual(first_id, candidates[0].participant.participant_id)
        self.assertAlmostEqual(
            candidates[0].score_breakdown.total_score,
            candidates[0].score_breakdown.elo_score * .30
            + candidates[0].score_breakdown.skill_score * .10
            + candidates[0].score_breakdown.schedule_score * .25
            + candidates[0].score_breakdown.distance_score * .15
            + candidates[0].score_breakdown.play_style_score * .10
            + candidates[0].score_breakdown.position_score * .05
            + candidates[0].score_breakdown.experience_score * .05,
            delta=0.11,
        )
        self.assertEqual(100, candidates[0].score_breakdown.skill_score)
        self.assertEqual(70, candidates[0].score_breakdown.experience_score)
        session = self.service.select_candidate(second_id, first_id)
        self.assertEqual(MatchSessionStatus.PROPOSED, session.status)

    def test_acceptance_recommends_three_available_venues_and_booker_selects_one(self):
        first_id, second_id = uuid4(), uuid4()
        self.service.join_queue(first_id, self.request("Người chơi A", 10.77, 106.67, 1200))
        session = self.service.join_queue(second_id, self.request("Người chơi B", 10.78, 106.68, 1250))
        self.service.decide(session.session_id, first_id, AcceptanceDecision.ACCEPTED)
        accepted = self.service.decide(session.session_id, second_id, AcceptanceDecision.ACCEPTED)
        self.venue.recommendations = [
            VenueRecommendation(
                venueId=uuid4(), venueCourtId=uuid4(), venueName=f"Sân {index}",
                address="Hà Nội", distanceKm=index, rating=4.5, score=90-index,
                matchReason="Cân bằng khoảng cách", suggestedCourts=["Sân A"],
                slotStart=time(18 + index % 2, 0), slotEnd=time(19 + index % 2, 0),
            )
            for index in range(1, 4)
        ]
        self.service.enrich_session_venue(session.session_id)
        enriched = self.service.get_session(session.session_id, second_id)
        self.assertEqual(3, len(enriched.proposal.venue_options))
        self.assertEqual(2, len(self.notifications.notifications))
        self.assertTrue(all(item[2] == "Đã tìm thấy sân phù hợp" for item in self.notifications.notifications))
        option = enriched.proposal.venue_options[1]
        selected = self.service.select_venue(
            session.session_id,
            accepted.proposal.designated_booker_id,
            MatchVenueSelectionRequest(venueId=option.venue_id, venueCourtId=option.venue_court_id),
        )
        self.assertEqual(MatchSessionStatus.VENUE_SELECTED, selected.status)
        self.assertEqual(option.venue_id, selected.proposal.venue_id)
        # Khung ranh chung 18:00-20:00 chi la pham vi; gio thi dau chot theo slot trong cua san da chon.
        self.assertEqual((time(18), time(20)), (accepted.start_time, accepted.end_time))
        self.assertEqual((time(18), time(19)), (selected.start_time, selected.end_time))

    def test_records_no_available_venue_and_can_refresh_recommendations(self):
        first_id, second_id = uuid4(), uuid4()
        self.service.join_queue(first_id, self.request("Người chơi A", 10.77, 106.67, 1200))
        session = self.service.join_queue(second_id, self.request("Người chơi B", 10.78, 106.68, 1250))
        self.service.decide(session.session_id, first_id, AcceptanceDecision.ACCEPTED)
        self.service.decide(session.session_id, second_id, AcceptanceDecision.ACCEPTED)

        self.service.enrich_session_venue(session.session_id)
        unavailable = self.service.get_session(session.session_id, first_id)
        self.assertEqual(VenueSearchStatus.NO_AVAILABILITY, unavailable.proposal.venue_search_status)

        self.venue.recommendations = [
            VenueRecommendation(
                venueId=uuid4(), venueCourtId=uuid4(), venueName="Sân mới",
                address="Hà Nội", distanceKm=1, rating=4.5, score=90,
                matchReason="Còn lịch trống", suggestedCourts=["Sân A"],
            )
        ]
        refreshed = self.service.refresh_venue_options(session.session_id, second_id)
        self.assertEqual(VenueSearchStatus.READY, refreshed.proposal.venue_search_status)
        self.assertEqual(1, len(refreshed.proposal.venue_options))

    def test_recovers_missing_conversation_when_session_is_polled(self):
        first_id, second_id = uuid4(), uuid4()
        self.service.join_queue(first_id, self.request("Người chơi A", 10.77, 106.67, 1200))
        session = self.service.join_queue(second_id, self.request("Người chơi B", 10.78, 106.68, 1250))
        self.service.decide(session.session_id, first_id, AcceptanceDecision.ACCEPTED)
        accepted = self.service.decide(session.session_id, second_id, AcceptanceDecision.ACCEPTED)
        accepted.proposal.conversation_id = None
        self.sessions.save(accepted)

        recovered = self.service.get_session(session.session_id, first_id)

        self.assertEqual(self.social.conversation_id, recovered.proposal.conversation_id)

    def test_participant_can_cancel_an_accepted_session_before_booking(self):
        first_id, second_id = uuid4(), uuid4()
        self.service.join_queue(first_id, self.request("Người chơi A", 10.77, 106.67, 1200))
        session = self.service.join_queue(second_id, self.request("Người chơi B", 10.78, 106.68, 1250))
        self.service.decide(session.session_id, first_id, AcceptanceDecision.ACCEPTED)
        self.service.decide(session.session_id, second_id, AcceptanceDecision.ACCEPTED)

        cancelled = self.service.cancel_session(session.session_id, first_id)

        self.assertEqual(MatchSessionStatus.CANCELLED, cancelled.status)
        self.assertEqual("PLAYER_CANCELLED", cancelled.proposal.cancel_reason)

    def test_result_requires_matching_claims_before_completion(self):
        first_id, second_id, session = self._finished_session(MatchSessionStatus.CHECKED_IN)
        pending = self.service.submit_result(
            session.session_id, first_id, MatchResultRequest(myScore=2, opponentScore=1)
        )
        self.assertEqual(MatchSessionStatus.RESULT_PENDING, pending.status)
        completed = self.service.submit_result(
            session.session_id, second_id, MatchResultRequest(myScore=1, opponentScore=2)
        )
        self.assertEqual(MatchSessionStatus.COMPLETED, completed.status)
        self.assertEqual(first_id, completed.result.winner_id)

    def test_confirmed_result_tells_each_player_their_own_elo_change(self):
        first_id, second_id, session = self._finished_session(MatchSessionStatus.CHECKED_IN)
        self.service.submit_result(session.session_id, first_id, MatchResultRequest(myScore=21, opponentScore=15))
        completed = self.service.submit_result(
            session.session_id, second_id, MatchResultRequest(myScore=15, opponentScore=21)
        )

        self.service.notify_result(completed, second_id)

        titles = {receiver: (title, content) for receiver, _, title, content in self.notifications.notifications}
        # FakeAuthClient trả ELO mới 1234: A (1200) được +34, B (1250) bị trừ 16.
        self.assertEqual("Thắng 21–15 · +34 ELO", titles[first_id][0])
        self.assertIn("1200 → 1234", titles[first_id][1])
        self.assertEqual("Thua 15–21 · −16 ELO", titles[second_id][0])
        # Lịch sử thi đấu bên auth-service biết đối thủ và tỷ số theo góc nhìn từng người.
        self.assertEqual(("Người chơi B", 21, 15), self.auth.history[first_id])
        self.assertEqual(("Người chơi A", 15, 21), self.auth.history[second_id])

    def _finished_session(self, status: MatchSessionStatus):
        first_id, second_id = uuid4(), uuid4()
        self.service.join_queue(first_id, self.request("Người chơi A", 10.77, 106.67, 1200))
        session = self.service.join_queue(second_id, self.request("Người chơi B", 10.78, 106.68, 1250))
        # Trận vừa kết thúc 30 phút trước (không cố định "hôm qua 18h–20h": sau 20h cửa sổ 24h đã hết).
        ended = (datetime.now(ZoneInfo(session.timezone)) - timedelta(minutes=30)).replace(microsecond=0)
        started = ended - timedelta(hours=1)
        session.play_date = ended.date()
        session.start_time = started.time() if started.date() == ended.date() else time(0, 0)
        session.end_time = ended.time()
        session.proposal = MatchProposal(
            designated_booker_id=first_id,
            venue_court_id=uuid4(),
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        session.status = status
        self.sessions.save(session)
        self.notifications.notifications.clear()
        return first_id, second_id, session

    def _force_result_deadline_passed(self, session_id):
        stored = self.sessions.find_by_id(session_id)
        stored.result_deadline_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        self.sessions.save(stored)

    def test_agreed_result_is_recorded_even_when_auth_service_rejects_the_elo_update(self):
        first_id, second_id, session = self._finished_session(MatchSessionStatus.CHECKED_IN)
        self.auth.available = False
        self.service.submit_result(
            session.session_id, first_id, MatchResultRequest(myScore=2, opponentScore=1)
        )

        completed = self.service.submit_result(
            session.session_id, second_id, MatchResultRequest(myScore=1, opponentScore=2)
        )

        self.assertEqual(MatchSessionStatus.COMPLETED, completed.status)
        self.assertEqual(first_id, completed.result.winner_id)
        self.assertFalse(completed.result.elo_applied)

    def test_pending_elo_update_is_retried_on_the_next_read(self):
        first_id, second_id, session = self._finished_session(MatchSessionStatus.CHECKED_IN)
        self.auth.available = False
        self.service.submit_result(
            session.session_id, first_id, MatchResultRequest(myScore=2, opponentScore=1)
        )
        self.service.submit_result(
            session.session_id, second_id, MatchResultRequest(myScore=1, opponentScore=2)
        )
        self.auth.available = True

        synced = self.service.get_history(first_id)[0]

        self.assertTrue(synced.result.elo_applied)
        self.assertEqual(2, len(self.auth.calls))
        self.assertEqual({1234}, set(synced.result.elo_updates.values()))

    def test_pending_elo_update_is_retried_when_reading_the_finished_session_status(self):
        first_id, second_id, session = self._finished_session(MatchSessionStatus.CHECKED_IN)
        self.auth.available = False
        self.service.submit_result(
            session.session_id, first_id, MatchResultRequest(myScore=2, opponentScore=1)
        )
        self.service.submit_result(
            session.session_id, second_id, MatchResultRequest(myScore=1, opponentScore=2)
        )
        self.auth.available = True
        # Hàng chờ có TTL riêng, sau khi hết hạn `get_status` rơi vào nhánh phiên
        # gần nhất — đúng nhánh mà màn hình kết quả của người chơi đang đọc.
        self.queue.participants.clear()

        _, _, synced = self.service.get_status(first_id)

        self.assertIsNotNone(synced)
        self.assertTrue(synced.result.elo_applied)

    def test_result_cannot_be_submitted_while_the_match_is_still_running(self):
        first_id, second_id = uuid4(), uuid4()
        self.service.join_queue(first_id, self.request("Người chơi A", 10.77, 106.67, 1200))
        session = self.service.join_queue(second_id, self.request("Người chơi B", 10.78, 106.68, 1250))
        session.status = MatchSessionStatus.CHECKED_IN
        self.sessions.save(session)

        with self.assertRaises(MatchmakingConflictError):
            self.service.submit_result(
                session.session_id, first_id, MatchResultRequest(myScore=2, opponentScore=1)
            )

    def test_finished_match_without_check_in_never_reaches_the_result_step(self):
        first_id, _, session = self._finished_session(MatchSessionStatus.CONFIRMED)

        settled = self.service.get_history(first_id)[0]

        self.assertEqual(MatchSessionStatus.EXPIRED, settled.status)
        self.assertEqual("NO_CHECK_IN", settled.proposal.cancel_reason)
        self.assertIsNone(settled.result)
        self.assertEqual(2, len(self.notifications.notifications))

    def test_result_window_opens_and_notifies_both_after_a_checked_in_match_ends(self):
        first_id, _, session = self._finished_session(MatchSessionStatus.CHECKED_IN)

        settled = self.service.get_history(first_id)[0]

        self.assertEqual(MatchSessionStatus.CHECKED_IN, settled.status)
        self.assertIsNotNone(settled.result_deadline_at)
        self.assertEqual(2, len(self.notifications.notifications))
        self.assertEqual("Trận đấu đã kết thúc", self.notifications.notifications[0][2])

    def test_expired_result_window_awards_the_win_to_the_participant_who_submitted(self):
        first_id, _, session = self._finished_session(MatchSessionStatus.CHECKED_IN)
        self.service.submit_result(
            session.session_id, first_id, MatchResultRequest(myScore=1, opponentScore=2)
        )
        self._force_result_deadline_passed(session.session_id)

        settled = self.service.get_history(first_id)[0]

        self.assertEqual(MatchSessionStatus.COMPLETED, settled.status)
        self.assertEqual(first_id, settled.result.winner_id)
        winner_index = 0 if settled.participants[0].participant_id == first_id else 1
        scores = [settled.result.participant_one_score, settled.result.participant_two_score]
        self.assertGreater(scores[winner_index], scores[1 - winner_index])

    def test_expired_result_window_without_any_claim_is_settled_as_a_draw(self):
        first_id, _, session = self._finished_session(MatchSessionStatus.CHECKED_IN)
        self.service.get_history(first_id)
        self._force_result_deadline_passed(session.session_id)

        settled = self.service.get_history(first_id)[0]

        self.assertEqual(MatchSessionStatus.COMPLETED, settled.status)
        self.assertIsNone(settled.result.winner_id)
        self.assertEqual(0, settled.result.participant_one_score)
        self.assertEqual(0, settled.result.participant_two_score)

    @staticmethod
    def request(name: str, latitude: float, longitude: float, elo: int) -> MatchmakingQueueRequest:
        return MatchmakingQueueRequest(
            playerName=name,
            sportType="BADMINTON",
            skillLevel="INTERMEDIATE",
            eloRating=elo,
            latitude=latitude,
            longitude=longitude,
            playDate=datetime.now(timezone.utc).date() + timedelta(days=1),
            startTime="18:00",
            endTime="20:00",
            maxEloDifference=300,
            maxDistanceKm=15,
        )



class PlayFormatMatchmakingTests(unittest.TestCase):
    """Đánh đôi ghép cặp với cặp; bóng đá ghép CLB với CLB, một quản lý thay cả CLB."""

    def setUp(self) -> None:
        self.queue = FakeQueueRepository()
        self.sessions = FakeSessionRepository()
        self.social = FakeSocialClient()
        self.notifications = FakeNotificationClient()
        self.auth = FakeAuthClient()
        self.clubs = FakeClubClient()
        self.service = MatchmakingService(
            self.queue, self.sessions, FakeDocumentRepository(), FakeVenueClient(), self.social,
            self.notifications,
            MatchmakingConfig(queue_ttl_seconds=900, session_ttl_seconds=600, minimum_score=50, scan_limit=200),
            self.auth, self.clubs,
        )

    @staticmethod
    def request(name, sport="BADMINTON", play_format=None, **extra):
        return MatchmakingQueueRequest(
            playerName=name, sportType=sport, skillLevel="INTERMEDIATE", eloRating=1200,
            latitude=10.77, longitude=106.67,
            playDate=datetime.now(timezone.utc).date() + timedelta(days=1),
            startTime="18:00", endTime="20:00", maxEloDifference=300, maxDistanceKm=15,
            playFormat=play_format, **extra,
        )

    def _pair(self, leader, partner, name):
        self.social.friendships.add(frozenset({leader, partner}))
        return self.service.join_queue(
            leader, self.request(name, play_format="BADMINTON_DOUBLES", partnerId=partner)
        )

    def _checked_in(self, session):
        ended = (datetime.now(ZoneInfo(session.timezone)) - timedelta(minutes=30)).replace(microsecond=0)
        session.play_date, session.end_time = ended.date(), ended.time()
        session.start_time = max(time(0, 0), (ended - timedelta(hours=1)).time()) if ended.hour else time(0, 0)
        session.status = MatchSessionStatus.CHECKED_IN
        self.sessions.save(session)

    def test_team_formats_need_a_club_and_doubles_need_a_partner(self):
        with self.assertRaises(ValueError):
            self.request("A", sport="FOOTBALL")
        with self.assertRaises(ValueError):
            self.request("A", sport="FOOTBALL", play_format="FOOTBALL_7")
        with self.assertRaises(ValueError):
            self.request("A", play_format="BADMINTON_DOUBLES")
        with self.assertRaises(ValueError):
            self.request("A", play_format="FOOTBALL_7", clubId=uuid4())

    def test_partner_must_be_a_friend(self):
        with self.assertRaises(ForbiddenException):
            self.service.join_queue(
                uuid4(), self.request("A", play_format="BADMINTON_DOUBLES", partnerId=uuid4())
            )

    def test_doubles_pairs_meet_pairs_and_every_player_is_rated_on_their_own(self):
        a, b, c, d, solo = uuid4(), uuid4(), uuid4(), uuid4(), uuid4()
        self.assertIsNone(self.service.join_queue(solo, self.request("Đơn")))
        self.assertIsNone(self._pair(a, b, "An"))
        # Bạn cặp cũng thấy lượt chờ và không thể đứng vào cặp khác cùng lúc.
        self.assertEqual(QueueStatus.QUEUED, self.service.get_status(b)[0])
        self.social.friendships.add(frozenset({c, b}))
        with self.assertRaises(MatchmakingConflictError):
            self.service.join_queue(c, self.request("Cường", play_format="BADMINTON_DOUBLES", partnerId=b))

        session = self._pair(c, d, "Cường")

        self.assertIsNotNone(session)
        self.assertEqual({a, c}, {side.participant_id for side in session.participants})
        self.assertEqual(QueueStatus.QUEUED, self.queue.find_by_participant(solo).status)
        pair = next(side for side in session.participants if side.participant_id == a)
        self.assertEqual(1250, pair.elo_rating)
        self.assertTrue(pair.name.startswith("An & "))

        self.service.notify_match_found(session, c)
        self.assertEqual({a, b, d}, {item[0] for item in self.notifications.notifications})

        # Một người mỗi bên xác nhận là đủ; cuộc trò chuyện có cả bốn người.
        self.service.decide(session.session_id, b, AcceptanceDecision.ACCEPTED)
        with self.assertRaises(MatchmakingConflictError):
            self.service.decide(session.session_id, a, AcceptanceDecision.ACCEPTED)
        accepted = self.service.decide(session.session_id, d, AcceptanceDecision.ACCEPTED)
        self.assertEqual(MatchSessionStatus.ACCEPTED, accepted.status)
        self.assertEqual({a, b, c, d}, set(self.social.conversation_members))

        self._checked_in(accepted)
        # Bạn cặp nhập rồi người kia của cùng bên nhập lại: chỉ tính một tỷ số mỗi bên.
        self.service.submit_result(session.session_id, b, MatchResultRequest(myScore=19, opponentScore=21))
        pending = self.service.submit_result(session.session_id, a, MatchResultRequest(myScore=21, opponentScore=19))
        self.assertEqual(MatchSessionStatus.RESULT_PENDING, pending.status)
        self.assertEqual(1, len(pending.result_claims))
        done = self.service.submit_result(session.session_id, d, MatchResultRequest(myScore=19, opponentScore=21))

        self.assertEqual(MatchSessionStatus.COMPLETED, done.status)
        self.assertEqual(a, done.result.winner_id)
        outcomes = {user_id: outcome for _, user_id, outcome in self.auth.calls}
        self.assertEqual({a: "WIN", b: "WIN", c: "LOSS", d: "LOSS"}, outcomes)
        # Mỗi người so với ELO trung bình của cặp đối thủ.
        self.assertEqual(1250, self.auth.opponent_elos[c])
        self.assertEqual({str(a), str(b), str(c), str(d)}, set(done.result.elo_updates))

    def test_active_match_blocks_a_new_search_and_a_past_unbooked_one_expires(self):
        a, b = uuid4(), uuid4()
        self.service.join_queue(a, self.request("An"))
        session = self.service.join_queue(b, self.request("Bình"))
        self.service.decide(session.session_id, a, AcceptanceDecision.ACCEPTED)
        accepted = self.service.decide(session.session_id, b, AcceptanceDecision.ACCEPTED)
        self.assertEqual(MatchSessionStatus.ACCEPTED, accepted.status)

        # Còn kèo chưa xong thì lượt tìm môn khác bị từ chối rõ ràng, không trả lại kèo cũ.
        with self.assertRaises(MatchmakingConflictError) as raised:
            self.service.join_queue(a, self.request("An", sport="TENNIS"))
        self.assertIn("cầu lông", str(raised.exception))

        # Khung giờ đã qua mà chưa đặt sân: kèo tự hết hạn và người chơi tìm kèo mới được.
        ended = datetime.now(ZoneInfo(accepted.timezone)) - timedelta(days=1)
        accepted.play_date = ended.date()
        self.sessions.save(accepted)
        self.notifications.notifications.clear()
        self.assertIsNone(self.service.join_queue(a, self.request("An", sport="TENNIS")))
        expired = self.sessions.find_by_id(accepted.session_id)
        self.assertEqual(MatchSessionStatus.EXPIRED, expired.status)
        self.assertEqual("NOT_BOOKED", expired.proposal.cancel_reason)
        self.assertTrue(any(title == "Kèo đã hết hạn" for _, _, title, _ in self.notifications.notifications))

    def test_friends_and_clubmates_form_a_team_that_meets_only_player_teams(self):
        a, b, c, d, e, f, stranger = (uuid4() for _ in range(7))
        club = uuid4()
        self.clubs.clubs = {club: {"name": "Rổ Quận 3", "elo": 1200, "members": 8, "managers": [d],
                                   "member_ids": [d, f], "sport": "BASKETBALL"}}
        for pair in ({a, b}, {a, c}, {d, e}):
            self.social.friendships.add(frozenset(pair))

        def team(name, *mates, **extra):
            return self.request(name, sport="BASKETBALL", play_format="BASKETBALL_3X3", partnerIds=list(mates), **extra)

        # Thiếu người, người lạ, hay CLB mình không ở đều bị chặn.
        with self.assertRaises(ValueError):
            team("An", b)
        with self.assertRaises(ForbiddenException):
            self.service.join_queue(a, team("An", b, stranger))
        with self.assertRaises(ForbiddenException):
            self.service.join_queue(a, team("An", b, f, teammateClubId=club))

        self.assertIsNone(self.service.join_queue(a, team("An", b, c)))
        # Một CLB đứng chờ cùng hình thức không bao giờ gặp đội bạn bè.
        self.assertIsNone(self.service.join_queue(d, self.request(
            "Dũng", sport="BASKETBALL", play_format="BASKETBALL_3X3", clubId=club)))
        self.service.leave_queue(d)
        # e là bạn của d, f là đồng đội CLB của d.
        session = self.service.join_queue(d, team("Dũng", e, f, teammateClubId=club))

        self.assertIsNotNone(session)
        mine = next(side for side in session.participants if side.participant_id == a)
        self.assertEqual("Đội An", mine.name)
        self.assertEqual(3, len(mine.members))
        self.assertEqual(round((1200 + 1300 + 1300) / 3), mine.elo_rating)
        self.service.decide(session.session_id, c, AcceptanceDecision.ACCEPTED)
        accepted = self.service.decide(session.session_id, f, AcceptanceDecision.ACCEPTED)
        self.assertEqual({a, b, c, d, e, f}, set(self.social.conversation_members))

        self._checked_in(accepted)
        self.service.submit_result(session.session_id, b, MatchResultRequest(myScore=21, opponentScore=15))
        done = self.service.submit_result(session.session_id, e, MatchResultRequest(myScore=15, opponentScore=21))
        self.assertEqual(MatchSessionStatus.COMPLETED, done.status)
        outcomes = {user_id: outcome for _, user_id, outcome in self.auth.calls}
        self.assertEqual({a: "WIN", b: "WIN", c: "WIN", d: "LOSS", e: "LOSS", f: "LOSS"}, outcomes)

    def test_club_managers_act_for_the_whole_club(self):
        owner_x, manager_x, member_x, owner_y = uuid4(), uuid4(), uuid4(), uuid4()
        club_x, club_y, small = uuid4(), uuid4(), uuid4()
        self.clubs.clubs = {
            club_x: {"name": "FC Bến Nghé", "elo": 1200, "members": 9, "managers": [owner_x, manager_x]},
            club_y: {"name": "FC Thảo Điền", "elo": 1260, "members": 12, "managers": [owner_y]},
            small: {"name": "FC Nhỏ", "elo": 1200, "members": 4, "managers": [owner_y]},
        }

        def club_request(club_id, play_format="FOOTBALL_7"):
            return self.request("Quản lý", sport="FOOTBALL", play_format=play_format, clubId=club_id)

        with self.assertRaises(ForbiddenException):
            self.service.join_queue(member_x, club_request(club_x))
        with self.assertRaises(MatchmakingConflictError):
            self.service.join_queue(owner_y, club_request(small))
        self.assertIsNone(self.service.join_queue(owner_x, club_request(club_x)))
        # Quản lý thứ hai không đưa CLB vào hàng chờ lần nữa.
        with self.assertRaises(MatchmakingConflictError):
            self.service.join_queue(manager_x, club_request(club_x))
        # Sân 5 không gặp sân 7.
        self.assertIsNone(self.service.join_queue(owner_y, club_request(club_y, "FOOTBALL_5")))
        session = self.service.join_queue(owner_y, club_request(club_y))

        self.assertIsNotNone(session)
        self.assertEqual({club_x, club_y}, {side.participant_id for side in session.participants})
        self.assertEqual(owner_y, session.proposal.designated_booker_id)
        self.service.decide(session.session_id, manager_x, AcceptanceDecision.ACCEPTED)
        accepted = self.service.decide(session.session_id, owner_y, AcceptanceDecision.ACCEPTED)
        self.assertEqual(MatchSessionStatus.ACCEPTED, accepted.status)
        with self.assertRaises(ForbiddenException):
            self.service.get_session(session.session_id, member_x)

        self._checked_in(accepted)
        self.service.submit_result(session.session_id, manager_x, MatchResultRequest(myScore=3, opponentScore=1))
        self.notifications.notifications.clear()
        done = self.service.submit_result(session.session_id, owner_y, MatchResultRequest(myScore=1, opponentScore=3))
        self.service.notify_result(done, owner_y)

        self.assertEqual(MatchSessionStatus.COMPLETED, done.status)
        self.assertEqual(club_x, done.result.winner_id)
        self.assertEqual([], self.auth.calls)
        [(_, play_format, first, second)] = self.clubs.results
        self.assertEqual("FOOTBALL_7", play_format)
        self.assertEqual({(club_x, 3), (club_y, 1)}, {first, second})
        messages = {item[0]: item[2] for item in self.notifications.notifications}
        self.assertEqual({owner_x, manager_x, owner_y}, set(messages))
        self.assertEqual("Thắng 3–1 · +16 ELO", messages[manager_x])


if __name__ == "__main__":
    unittest.main()
