import unittest

from app.application.service.elo_rating import (
    calculate_new_rating,
    initial_rating,
    k_factor,
)
from app.domain.enums import SkillLevel, SportType


class EloRatingPolicyTests(unittest.TestCase):
    def test_initial_rating_uses_declared_skill_in_every_sport_pool(self):
        expected = {
            SkillLevel.BEGINNER: 1000,
            SkillLevel.INTERMEDIATE: 1200,
            SkillLevel.ADVANCED: 1500,
            SkillLevel.PRO: 1800,
        }
        for sport in SportType:
            for skill, rating in expected.items():
                self.assertEqual(rating, initial_rating(sport, skill))

    def test_provisional_rating_converges_faster(self):
        self.assertEqual(
            1220,
            calculate_new_rating(SportType.BADMINTON, 1200, 1200, 0, 1.0),
        )
        self.assertEqual(
            1216,
            calculate_new_rating(SportType.BADMINTON, 1200, 1200, 10, 1.0),
        )

    def test_team_sport_has_lower_established_volatility(self):
        self.assertEqual(24, k_factor(SportType.FOOTBALL, 10))
        self.assertEqual(32, k_factor(SportType.TENNIS, 10))
        self.assertEqual(
            1212,
            calculate_new_rating(SportType.FOOTBALL, 1200, 1200, 10, 1.0),
        )


if __name__ == "__main__":
    unittest.main()
