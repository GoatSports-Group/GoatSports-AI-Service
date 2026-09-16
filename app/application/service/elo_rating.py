from app.domain.enums import SkillLevel, SportType

ELO_SCALE = 400.0
PROVISIONAL_MATCH_LIMIT = 10
EXPERIENCED_MATCH_LIMIT = 30

INITIAL_RATINGS: dict[SkillLevel, int] = {
    SkillLevel.BEGINNER: 1000,
    SkillLevel.INTERMEDIATE: 1200,
    SkillLevel.ADVANCED: 1500,
    SkillLevel.PRO: 1800,
}

# Team results are a noisier signal of one player's strength, so their
# established ratings intentionally move more slowly than head-to-head sports.
SPORT_K_FACTORS: dict[SportType, tuple[int, int]] = {
    SportType.BADMINTON: (32, 24),
    SportType.TENNIS: (32, 24),
    SportType.PICKLEBALL: (32, 24),
    SportType.FOOTBALL: (24, 16),
    SportType.BASKETBALL: (24, 16),
    SportType.VOLLEYBALL: (24, 16),
}


def initial_rating(sport_type: SportType, skill_level: SkillLevel) -> int:
    # This lookup makes a missing per-sport configuration fail loudly.
    SPORT_K_FACTORS[sport_type]
    return INITIAL_RATINGS[skill_level]


def k_factor(sport_type: SportType, match_count: int) -> int:
    established, experienced = SPORT_K_FACTORS[sport_type]
    safe_match_count = max(0, match_count)
    if safe_match_count < PROVISIONAL_MATCH_LIMIT:
        return 40
    return established if safe_match_count < EXPERIENCED_MATCH_LIMIT else experienced


def calculate_new_rating(
    sport_type: SportType,
    current_rating: int,
    opponent_rating: int,
    match_count: int,
    actual_score: float,
) -> int:
    if not 0.0 <= actual_score <= 1.0:
        raise ValueError("Actual Elo score must be between 0 and 1")
    expected_score = 1.0 / (
        1.0 + 10.0 ** ((opponent_rating - current_rating) / ELO_SCALE)
    )
    updated = round(
        current_rating
        + k_factor(sport_type, match_count) * (actual_score - expected_score)
    )
    return max(0, updated)
