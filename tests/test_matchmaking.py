import unittest
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from app.application.dto.request.models import (
    MatchmakingQueueRequest,
    MatchResultRequest,
    MatchVenueSelectionRequest,
)
from app.application.dto.response.models import VenueRecommendation
from app.application.service.matchmaking_service import MatchmakingService
from app.application.service.service_config import MatchmakingConfig
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

    def find_by_participant(self, participant_id, limit):
        return [
            session.model_copy(deep=True)
            for session in reversed(list(self.sessions.values()))
            if any(profile.participant_id == participant_id for profile in session.participants)
        ][:limit]


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
            candidates[0].score_breakdown.elo_score * .4
            + candidates[0].score_breakdown.schedule_score * .3
            + candidates[0].score_breakdown.distance_score * .2
            + candidates[0].score_breakdown.play_style_score * .1,
            delta=0.11,
        )
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
