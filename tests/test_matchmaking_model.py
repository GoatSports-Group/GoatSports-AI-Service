import random
import unittest
from datetime import date, datetime, time, timedelta, timezone
from uuid import uuid4

from app.application.service.match_insight_service import DISABLED, FAILED, PENDING, READY, MatchInsightService
from app.application.service.matchmaking_model import (
    FEATURES,
    MIN_SAMPLES,
    importance,
    is_learned,
    label,
    success_probability,
    train,
)
from app.domain.entity.models import (
    MatchmakingSession,
    MatchScoreBreakdown,
    OpponentFeedback,
    PlayerProfile,
)
from app.domain.enums import MatchSessionStatus


def _player(name):
    return PlayerProfile(participant_id=uuid4(), name=name, sport_type="BADMINTON", latitude=10.77, longitude=106.67)


def _session(breakdown, status, feedback=()):
    return MatchmakingSession(
        sport_type="BADMINTON", play_date=date(2026, 10, 1), start_time=time(18), end_time=time(20),
        timezone="Asia/Ho_Chi_Minh", compatibility_score=breakdown.total_score, distance_km=2.0, elo_difference=40,
        score_breakdown=breakdown, status=status, expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
        participants=[_player("An"), _player("Bình")], feedback=list(feedback),
    )


def _breakdown(**scores):
    values = {name: scores.get(name, 60.0) for name in FEATURES}
    return MatchScoreBreakdown(**values, total_score=sum(values.values()) / len(values))


class MatchmakingModelTests(unittest.TestCase):
    def test_labels_come_from_real_outcomes(self):
        good = _breakdown()
        self.assertEqual(1, label(_session(good, MatchSessionStatus.COMPLETED)))
        self.assertEqual(0, label(_session(good, MatchSessionStatus.REJECTED)))
        bad_game = OpponentFeedback(reviewer_id=uuid4(), reviewee_id=uuid4(), rating=4, fair_play_rating=2)
        self.assertEqual(0, label(_session(good, MatchSessionStatus.COMPLETED, [bad_game])))
        self.assertIsNone(label(_session(good, MatchSessionStatus.ACCEPTED)))

    def test_too_little_data_keeps_the_handwritten_weights(self):
        sessions = [_session(_breakdown(), MatchSessionStatus.COMPLETED) for _ in range(MIN_SAMPLES - 1)]
        outcome = train(sessions)
        self.assertIsNone(outcome.model)
        self.assertIn(str(MIN_SAMPLES), outcome.reason)

    def test_learns_that_the_shared_schedule_decides_whether_a_match_happens(self):
        rng = random.Random(7)
        sessions = []
        for _ in range(160):
            schedule = rng.uniform(0, 100)
            breakdown = _breakdown(
                schedule_score=schedule,
                elo_score=rng.uniform(0, 100),
                distance_score=rng.uniform(0, 100),
            )
            # Dữ liệu giả lập: kèo có khung giờ trùng nhiều thì thường diễn ra; thêm 10% nhiễu.
            happened = (schedule > 55) != (rng.random() < 0.1)
            sessions.append(_session(breakdown, MatchSessionStatus.COMPLETED if happened else MatchSessionStatus.REJECTED))

        outcome = train(sessions)

        self.assertIsNotNone(outcome.model)
        self.assertTrue(is_learned(outcome.model))
        self.assertGreater(outcome.auc, 0.8)
        weights = importance(outcome.model)
        self.assertEqual("schedule_score", max(weights, key=weights.get))
        likely = success_probability(outcome.model, _breakdown(schedule_score=95))
        unlikely = success_probability(outcome.model, _breakdown(schedule_score=10))
        self.assertGreater(likely, unlikely)


class FakeInsightRepository:
    def __init__(self, session):
        self.session = session
        self.updates = []

    def find_by_id(self, _):
        return self.session

    def find_by_participant(self, *_):
        return []

    def update_insight(self, session_id, insight, status, model):
        self.updates.append((insight, status, model))
        self.session.ai_insight, self.session.ai_insight_status, self.session.ai_insight_model = insight, status, model


class FakeLanguageModel:
    def __init__(self, name="qwen3.5:4b", reply="<think>so sánh</think>Hai bên **hợp nhau** về khung giờ."):
        self.model_name = name
        self.reply = reply
        self.prompts = []

    def complete(self, system, prompt, max_tokens=None):
        self.prompts.append(prompt)
        return self.reply


class MatchInsightTests(unittest.TestCase):
    def test_writes_a_clean_analysis_once_and_only_the_insight_fields(self):
        session = _session(_breakdown(), MatchSessionStatus.PROPOSED)
        repository, llm = FakeInsightRepository(session), FakeLanguageModel()
        service = MatchInsightService(repository, llm)

        self.assertTrue(service.claim(session))
        self.assertFalse(service.claim(session))  # đang chạy thì không xếp hàng lần hai
        self.assertEqual(PENDING, session.ai_insight_status)
        service.explain(session.session_id)

        self.assertEqual(READY, session.ai_insight_status)
        self.assertEqual("Hai bên hợp nhau về khung giờ.", session.ai_insight)
        self.assertEqual("qwen3.5:4b", session.ai_insight_model)
        self.assertIn("An", llm.prompts[0])
        self.assertFalse(service.claim(session))

    def test_drops_any_sentence_about_betting(self):
        session = _session(_breakdown(), MatchSessionStatus.PROPOSED)
        repository = FakeInsightRepository(session)
        llm = FakeLanguageModel(reply="Hai bên cân sức. Hãy cân nhắc đặt cược cho Bình trước trận! Nhớ khởi động kỹ.")
        service = MatchInsightService(repository, llm)
        service.claim(session)
        service.explain(session.session_id)
        self.assertEqual("Hai bên cân sức. Nhớ khởi động kỹ.", session.ai_insight)

    def test_drops_a_sentence_cut_off_by_the_token_limit(self):
        session = _session(_breakdown(), MatchSessionStatus.PROPOSED)
        repository = FakeInsightRepository(session)
        llm = FakeLanguageModel(reply="Hai bên trùng trọn khung giờ. Nhớ khởi động kỹ và thống nhất")
        service = MatchInsightService(repository, llm)
        service.claim(session)
        service.explain(session.session_id)
        self.assertEqual("Hai bên trùng trọn khung giờ.", session.ai_insight)

    def test_no_llm_disables_and_a_failed_call_is_recorded(self):
        session = _session(_breakdown(), MatchSessionStatus.PROPOSED)
        repository = FakeInsightRepository(session)
        self.assertFalse(MatchInsightService(repository, FakeLanguageModel(name=None)).claim(session))
        self.assertEqual(DISABLED, session.ai_insight_status)

        other = _session(_breakdown(), MatchSessionStatus.PROPOSED)
        repository = FakeInsightRepository(other)
        service = MatchInsightService(repository, FakeLanguageModel(reply=None))
        self.assertTrue(service.claim(other))
        service.explain(other.session_id)
        self.assertEqual(FAILED, other.ai_insight_status)
