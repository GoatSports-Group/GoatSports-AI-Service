import unittest
from contextlib import nullcontext
from datetime import datetime, time, timedelta, timezone
from uuid import UUID, uuid4

from app.application.dto.request.models import (
    MatchmakingQueueRequest,
    MatchResultRequest,
    MatchVenueSelectionRequest,
)
from app.application.dto.response.models import VenueRecommendation
from app.application.service.matchmaking_service import MatchmakingConflictError, MatchmakingService
from app.application.service.service_config import MatchmakingConfig
from app.domain.entity.models import MatchProposal
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
        self.participants[profile.participant_id] = entry.queue_entry_id
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
            if session.status in {MatchSessionStatus.PROPOSED, MatchSessionStatus.ACCEPTED_BY_ONE, MatchSessionStatus.ACCEPTED} and any(
                profile.participant_id == participant_id for profile in session.participants
            ):
                return session.model_copy(deep=True)
        return None

    def find_latest_by_participant(self, participant_id):
        for session in reversed(list(self.sessions.values())):
            if any(profile.participant_id == participant_id for profile in session.participants):
                return session.model_copy(deep=True)
        return None

    def find_by_participant(self, participant_id, limit, offset=0):
        return [
            session.model_copy(deep=True)
            for session in reversed(list(self.sessions.values()))
            if any(profile.participant_id == participant_id for profile in session.participants)
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

    def create_match_conversation(self, *_, **__):
        return self.conversation_id


class FakeNotificationClient:
    def __init__(self) -> None:
        self.notifications = []

    def notify_matchmaking(self, receiver_id, session_id, title, content):
        self.notifications.append((receiver_id, session_id, title, content))


class MatchmakingServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.queue = FakeQueueRepository()
        self.sessions = FakeSessionRepository()
        self.social = FakeSocialClient()
        self.notifications = FakeNotificationClient()
        self.venue = FakeVenueClient()
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
        first_id, second_id = uuid4(), uuid4()
        self.service.join_queue(first_id, self.request("Người chơi A", 10.77, 106.67, 1200))
        session = self.service.join_queue(second_id, self.request("Người chơi B", 10.78, 106.68, 1250))
        session.status = MatchSessionStatus.CHECKED_IN
        self.sessions.save(session)
        pending = self.service.submit_result(
            session.session_id, first_id, MatchResultRequest(myScore=2, opponentScore=1)
        )
        self.assertEqual(MatchSessionStatus.RESULT_PENDING, pending.status)
        completed = self.service.submit_result(
            session.session_id, second_id, MatchResultRequest(myScore=1, opponentScore=2)
        )
        self.assertEqual(MatchSessionStatus.COMPLETED, completed.status)
        self.assertEqual(first_id, completed.result.winner_id)

    def _finished_session(self, status: MatchSessionStatus):
        first_id, second_id = uuid4(), uuid4()
        self.service.join_queue(first_id, self.request("Người chơi A", 10.77, 106.67, 1200))
        session = self.service.join_queue(second_id, self.request("Người chơi B", 10.78, 106.68, 1250))
        session.play_date = datetime.now(timezone.utc).date() - timedelta(days=1)
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


if __name__ == "__main__":
    unittest.main()
