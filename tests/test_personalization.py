import unittest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from app.application.dto.request.models import PostCandidate
from app.application.dto.response.models import VenueRecommendation
from app.application.service.activity_ingest_service import ActivityIngestService
from app.application.service.post_ranking_service import PostRankingService
from app.application.service.recommendation_service import RecommendationService
from app.domain.enums import InteractionType, SportType

NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)


class InMemoryPersonalizationRepository:
    def __init__(self):
        self.interactions = {}
        self.disabled = set()
        self.model = None

    def find_active_model(self, _):
        return self.model

    def save_model_version(self, model):
        self.model = model
        return model

    def save_interaction(self, interaction):
        self.interactions[interaction.interaction_id] = interaction
        return interaction

    def delete_interaction(self, interaction_id):
        self.interactions.pop(interaction_id, None)

    def delete_user_interactions(self, user_id):
        doomed = [key for key, item in self.interactions.items() if item.user_id == user_id]
        for key in doomed:
            del self.interactions[key]
        return len(doomed)

    def find_interactions(self, user_id=None, target_type=None, limit=20_000, since=None):
        return [
            item for item in self.interactions.values()
            if (user_id is None or item.user_id == user_id)
            and (target_type is None or item.target_type == target_type)
            and (since is None or item.occurred_at >= since)
        ][:limit]

    def is_personalization_enabled(self, user_id):
        return user_id not in self.disabled

    def set_personalization_enabled(self, user_id, enabled):
        (self.disabled.discard if enabled else self.disabled.add)(user_id)


def event(event_type, user_id, payload, occurred_at=NOW):
    return {"eventId": str(uuid4()), "eventType": event_type, "userId": str(user_id),
            "occurredAt": occurred_at.isoformat(), "payload": payload}


class ActivityIngestTests(unittest.TestCase):
    def setUp(self):
        self.repository = InMemoryPersonalizationRepository()
        self.service = ActivityIngestService(self.repository)
        self.user = uuid4()

    def test_booking_counts_while_valid_and_is_removed_when_cancelled(self):
        booking, venue = uuid4(), uuid4()
        payload = {"bookingId": str(booking), "venueId": str(venue), "sportType": "BADMINTON",
                   "status": "CONFIRMED", "minPrice": 120000, "rating": 4.5}
        self.service.handle(event("activity.booking.state.v1", self.user, payload))
        # Kafka giao lại / nạp bù chạy lại: vẫn một hành vi.
        self.service.handle(event("activity.booking.state.v1", self.user, {**payload, "status": "COMPLETED"}))
        [stored] = self.repository.interactions.values()
        self.assertEqual(InteractionType.BOOKING, stored.interaction_type)
        self.assertEqual(SportType.BADMINTON, stored.sport_type)
        self.assertEqual({"price": 120000.0, "rating": 4.5}, stored.metadata)

        self.service.handle(event("activity.booking.state.v1", self.user, {**payload, "status": "CANCELLED"}))
        self.assertEqual({}, self.repository.interactions)

    def test_venue_views_count_once_per_day(self):
        venue = str(uuid4())
        payload = {"venueId": venue, "sportTypes": ["TENNIS"]}
        self.service.handle(event("activity.venue.viewed.v1", self.user, payload))
        self.service.handle(event("activity.venue.viewed.v1", self.user, payload, NOW + timedelta(hours=2)))
        self.service.handle(event("activity.venue.viewed.v1", self.user, payload, NOW + timedelta(days=1)))
        self.assertEqual(2, len(self.repository.interactions))

    def test_unlike_and_deleted_comment_remove_the_signal_and_own_posts_are_ignored(self):
        post, author, comment = str(uuid4()), str(uuid4()), str(uuid4())
        like = {"postId": post, "authorId": author, "sport": "FOOTBALL", "reaction": "LIKE", "active": True}
        self.service.handle(event("activity.post.reaction.v1", self.user, like))
        self.service.handle(event("activity.post.comment.v1", self.user,
                                  {"postId": post, "authorId": author, "commentId": comment, "active": True}))
        self.assertEqual(2, len(self.repository.interactions))

        self.service.handle(event("activity.post.reaction.v1", self.user, {**like, "active": False}))
        self.service.handle(event("activity.post.comment.v1", self.user,
                                  {"postId": post, "authorId": author, "commentId": comment, "active": False}))
        self.assertEqual({}, self.repository.interactions)

        self.service.handle(event("activity.post.reaction.v1", self.user, {**like, "authorId": str(self.user)}))
        self.assertEqual({}, self.repository.interactions)

    def test_opting_out_erases_history_and_ignores_new_events(self):
        like = {"postId": str(uuid4()), "authorId": str(uuid4()), "reaction": "SAVE", "active": True}
        self.service.handle(event("activity.post.reaction.v1", self.user, like))
        self.service.set_personalization(self.user, False)
        self.assertEqual({}, self.repository.interactions)

        self.service.handle(event("activity.post.reaction.v1", self.user, like))
        self.assertEqual({}, self.repository.interactions)

    def test_malformed_events_are_skipped(self):
        self.service.handle({"eventType": "activity.post.reaction.v1"})
        self.service.handle(event("activity.post.reaction.v1", self.user, {"postId": "not-a-uuid"}))
        self.service.handle(event("activity.post.reaction.v1", self.user,
                                  {"postId": str(uuid4()), "reaction": "WOW", "active": True}))
        self.assertEqual({}, self.repository.interactions)


class PostRankingTests(unittest.TestCase):
    def setUp(self):
        self.repository = InMemoryPersonalizationRepository()
        self.ingest = ActivityIngestService(self.repository)
        self.ranking = PostRankingService(self.repository, snapshot_seconds=0, clock=lambda: NOW)
        self.user = uuid4()

    def like(self, user, post, author, sport="FOOTBALL"):
        self.ingest.handle(event("activity.post.reaction.v1", user, {
            "postId": str(post), "authorId": str(author), "sport": sport, "reaction": "LIKE", "active": True,
        }, NOW - timedelta(days=1)))

    @staticmethod
    def candidate(post, author, sport="FOOTBALL", hours_old=1, likes=0):
        return PostCandidate(post_id=post, author_id=author, sport=sport, like_count=likes,
                             published_at=NOW - timedelta(hours=hours_old))

    def scores(self, user, candidates):
        return {item.post_id: item.score for item in self.ranking.rank(user, candidates)}

    def test_new_user_gets_fresh_and_popular_posts_first(self):
        fresh, stale = uuid4(), uuid4()
        scores = self.scores(self.user, [self.candidate(stale, uuid4(), hours_old=96),
                                         self.candidate(fresh, uuid4(), hours_old=1)])
        self.assertGreater(scores[fresh], scores[stale])

    def test_posts_from_authors_and_sports_the_user_engages_with_rank_higher(self):
        favourite = uuid4()
        self.like(self.user, uuid4(), favourite, sport="BADMINTON")
        theirs, stranger = uuid4(), uuid4()
        scores = self.scores(self.user, [self.candidate(stranger, uuid4(), sport="TENNIS"),
                                         self.candidate(theirs, favourite, sport="BADMINTON", hours_old=6)])
        self.assertGreater(scores[theirs], scores[stranger])

    def test_posts_liked_by_similar_users_rank_higher(self):
        twin, shared_author = uuid4(), uuid4()
        shared = uuid4()
        self.like(self.user, shared, shared_author)
        self.like(twin, shared, shared_author)
        twins_pick, other = uuid4(), uuid4()
        self.like(twin, twins_pick, uuid4())
        scores = self.scores(self.user, [self.candidate(other, uuid4()), self.candidate(twins_pick, uuid4())])
        self.assertGreater(scores[twins_pick], scores[other])

    def test_opted_out_user_is_ranked_like_a_new_user(self):
        favourite = uuid4()
        self.like(self.user, uuid4(), favourite)
        self.ingest.set_personalization(self.user, False)
        theirs, fresher = uuid4(), uuid4()
        scores = self.scores(self.user, [self.candidate(theirs, favourite, hours_old=6),
                                         self.candidate(fresher, uuid4(), hours_old=1)])
        self.assertGreater(scores[fresher], scores[theirs])


class FakeVenueClient:
    def __init__(self, venues):
        self.venues = venues
        self.calls = []

    def recommend(self, *args, **kwargs):
        self.calls.append(kwargs)
        return [venue.model_copy() for venue in self.venues], None


class VenueRecommendationTests(unittest.TestCase):
    def setUp(self):
        self.repository = InMemoryPersonalizationRepository()
        self.ingest = ActivityIngestService(self.repository)
        self.user = uuid4()
        self.badminton = VenueRecommendation(venue_id=uuid4(), venue_name="Cầu lông", sport_types=["BADMINTON"],
                                             address="Q1", distance_km=6.0, rating=4.0, min_price=100000, score=0,
                                             match_reason="")
        self.multi = VenueRecommendation(venue_id=uuid4(), venue_name="Đa môn", sport_types=["BADMINTON", "TENNIS"],
                                         address="Q3", distance_km=4.0, rating=4.0, min_price=100000, score=0,
                                         match_reason="")
        self.client = FakeVenueClient([self.multi, self.badminton])
        self.service = RecommendationService(self.client, self.repository)

    def recommend(self):
        return self.service.get_recommendations(self.user, 10.7, 106.7, SportType.TENNIS, radius_km=10)

    def test_a_venue_the_user_keeps_booking_rises_and_court_lookups_are_skipped(self):
        for _ in range(3):
            self.ingest.handle(event("activity.booking.state.v1", self.user, {
                "bookingId": str(uuid4()), "venueId": str(self.badminton.venue_id), "sportType": "BADMINTON",
                "status": "COMPLETED", "minPrice": 100000, "rating": 4.0}))
        self.assertEqual(self.badminton.venue_id, self.recommend()[0].venue_id)
        self.assertFalse(self.client.calls[0]["include_courts"])
        self.assertEqual(10, self.client.calls[0]["radius_km"])

    def test_opted_out_user_gets_the_same_list_as_a_new_user(self):
        baseline = [item.venue_id for item in self.recommend()]
        for _ in range(3):
            self.ingest.handle(event("activity.booking.state.v1", self.user, {
                "bookingId": str(uuid4()), "venueId": str(self.badminton.venue_id), "sportType": "BADMINTON",
                "status": "COMPLETED"}))
        self.ingest.set_personalization(self.user, False)
        self.assertEqual(baseline, [item.venue_id for item in self.recommend()])


if __name__ == "__main__":
    unittest.main()
