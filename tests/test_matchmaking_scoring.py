import unittest
from datetime import date, time
from uuid import uuid4

from app.application.service.matchmaking_scoring import compatibility_breakdown
from app.domain.entity.models import MatchPreference, PlayerProfile
from app.domain.enums import SkillLevel, SportType


class MatchmakingScoringTests(unittest.TestCase):
    def setUp(self):
        self.preference = MatchPreference(
            sportType=SportType.FOOTBALL,
            playDate=date(2026, 9, 20),
            startTime=time(18, 0),
            endTime=time(20, 0),
            maxEloDifference=300,
            maxDistanceKm=20,
        )
        self.first = self.profile(
            SkillLevel.ADVANCED,
            {"Hậu vệ"},
            24,
            0.62,
        )

    def test_profile_fields_change_compatibility_score(self):
        similar = self.profile(
            SkillLevel.ADVANCED,
            {"Hậu vệ"},
            22,
            0.60,
        )
        different = self.profile(
            SkillLevel.BEGINNER,
            {"Tiền đạo"},
            80,
            0.20,
        )

        similar_breakdown, _, _ = compatibility_breakdown(
            self.first, self.preference, similar, self.preference
        )
        different_breakdown, _, _ = compatibility_breakdown(
            self.first, self.preference, different, self.preference
        )

        self.assertEqual(100, similar_breakdown.skill_score)
        self.assertEqual(100, similar_breakdown.position_score)
        self.assertGreater(
            similar_breakdown.experience_score,
            different_breakdown.experience_score,
        )
        self.assertGreater(
            similar_breakdown.total_score,
            different_breakdown.total_score,
        )

    @staticmethod
    def profile(skill, positions, match_count, win_rate):
        return PlayerProfile(
            participantId=uuid4(),
            name="Người chơi",
            sportType=SportType.FOOTBALL,
            skillLevel=skill,
            eloRating=1400,
            preferredPositions=positions,
            playStyle="BALANCED",
            matchCount=match_count,
            winRate=win_rate,
            latitude=10.776,
            longitude=106.700,
        )


if __name__ == "__main__":
    unittest.main()
