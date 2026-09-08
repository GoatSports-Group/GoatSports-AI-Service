import unittest
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from app.application.dto.request.models import MatchmakingQueueRequest
from app.application.service.matchmaking_service import MatchmakingService
from app.application.service.service_config import MatchmakingConfig
from app.domain.enums import AcceptanceDecision, MatchSessionStatus, QueueStatus


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


class FakeDocumentRepository:
    def __init__(self):
        self.model = None

    def find_active_model(self, _):
        return self.model

    def save_model_version(self, model):
        self.model = model
        return model


class FakeVenueClient:
    def recommend(self, *_, **__):
        return [], None


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
        self.service = MatchmakingService(
            self.queue,
            self.sessions,
            FakeDocumentRepository(),
            FakeVenueClient(),
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

    def test_accepts_once_per_participant_and_creates_conversation_after_both_accept(self):
        first_id, second_id = uuid4(), uuid4()
        self.service.join_queue(first_id, self.request("Người chơi A", 10.77, 106.67, 1200))
        session = self.service.join_queue(second_id, self.request("Người chơi B", 10.78, 106.68, 1250))

        first_acceptance = self.service.decide(session.session_id, first_id, AcceptanceDecision.ACCEPTED)
        accepted = self.service.decide(session.session_id, second_id, AcceptanceDecision.ACCEPTED)

        self.assertEqual(MatchSessionStatus.ACCEPTED_BY_ONE, first_acceptance.status)
        self.assertEqual(MatchSessionStatus.ACCEPTED, accepted.status)
        self.assertEqual(self.social.conversation_id, accepted.proposal.conversation_id)

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
