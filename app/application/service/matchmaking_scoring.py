import math
from datetime import time

from app.domain.entity.models import (
    MatchParticipantProfile,
    MatchPreference,
    MatchScoreBreakdown,
    PlayerProfile,
)
from app.domain.enums import SkillLevel

SCORE_WEIGHTS = {
    "elo": 0.30,
    "skill": 0.10,
    "schedule": 0.25,
    "distance": 0.15,
    "play_style": 0.10,
    "position": 0.05,
    "experience": 0.05,
}

SKILL_ORDER = {
    SkillLevel.BEGINNER: 0,
    SkillLevel.INTERMEDIATE: 1,
    SkillLevel.ADVANCED: 2,
    SkillLevel.PRO: 3,
}


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    value = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2.0) ** 2
    )
    return round(radius_km * 2.0 * math.atan2(math.sqrt(value), math.sqrt(1.0 - value)), 2)


def overlapping_window(first: MatchPreference, second: MatchPreference) -> tuple[time, time] | None:
    if first.play_date != second.play_date or first.sport_type != second.sport_type:
        return None
    start = max(first.start_time, second.start_time)
    end = min(first.end_time, second.end_time)
    return (start, end) if start < end else None


def compatibility_score(
    first_profile: MatchParticipantProfile,
    first_preference: MatchPreference,
    second_profile: MatchParticipantProfile,
    second_preference: MatchPreference,
) -> tuple[float, float, int] | None:
    result = compatibility_breakdown(
        first_profile,
        first_preference,
        second_profile,
        second_preference,
    )
    if result is None:
        return None
    breakdown, distance, elo_difference = result
    return breakdown.total_score, distance, elo_difference


def compatibility_breakdown(
    first_profile: MatchParticipantProfile,
    first_preference: MatchPreference,
    second_profile: MatchParticipantProfile,
    second_preference: MatchPreference,
) -> tuple[MatchScoreBreakdown, float, int] | None:
    overlap = overlapping_window(first_preference, second_preference)
    if overlap is None:
        return None

    distance = haversine_distance(
        first_profile.latitude,
        first_profile.longitude,
        second_profile.latitude,
        second_profile.longitude,
    )
    allowed_distance = min(
        first_preference.max_distance_km,
        second_preference.max_distance_km,
    )
    if distance > allowed_distance:
        return None

    elo_difference = abs(first_profile.elo_rating - second_profile.elo_rating)
    allowed_elo_difference = min(
        first_preference.max_elo_difference,
        second_preference.max_elo_difference,
    )
    if elo_difference > allowed_elo_difference:
        return None

    distance_score = 100.0 * (1.0 - distance / allowed_distance)
    elo_score = (
        100.0
        if allowed_elo_difference == 0
        else 100.0 * (1.0 - elo_difference / allowed_elo_difference)
    )
    first_minutes = _minutes(first_preference.end_time) - _minutes(first_preference.start_time)
    second_minutes = _minutes(second_preference.end_time) - _minutes(second_preference.start_time)
    overlap_minutes = _minutes(overlap[1]) - _minutes(overlap[0])
    schedule_score = 100.0 * overlap_minutes / max(first_minutes, second_minutes)

    first_style = first_profile.play_style if isinstance(first_profile, PlayerProfile) else "BALANCED"
    second_style = second_profile.play_style if isinstance(second_profile, PlayerProfile) else "BALANCED"
    if first_style == second_style:
        play_style_score = 100.0
    elif "BALANCED" in {first_style, second_style}:
        play_style_score = 70.0
    else:
        play_style_score = 40.0

    skill_score = _skill_score(first_profile, second_profile)
    position_score = _position_score(first_profile, second_profile)
    experience_score = _experience_score(first_profile, second_profile)

    score = round(
        elo_score * SCORE_WEIGHTS["elo"]
        + skill_score * SCORE_WEIGHTS["skill"]
        + schedule_score * SCORE_WEIGHTS["schedule"]
        + distance_score * SCORE_WEIGHTS["distance"]
        + play_style_score * SCORE_WEIGHTS["play_style"]
        + position_score * SCORE_WEIGHTS["position"]
        + experience_score * SCORE_WEIGHTS["experience"],
        1,
    )
    breakdown = MatchScoreBreakdown(
        eloScore=round(elo_score, 1),
        skillScore=round(skill_score, 1),
        scheduleScore=round(schedule_score, 1),
        distanceScore=round(distance_score, 1),
        playStyleScore=round(play_style_score, 1),
        positionScore=round(position_score, 1),
        experienceScore=round(experience_score, 1),
        totalScore=score,
        reasons=[
            f"Chênh lệch {elo_difference} ELO trong ngưỡng cho phép",
            f"Mức trình độ tương đồng {skill_score:.0f}%",
            f"Khung giờ trùng nhau {overlap_minutes} phút",
            f"Khoảng cách giữa hai người chơi {distance:.1f} km",
            (
                "Cùng phong cách chơi"
                if first_style == second_style
                else "Phong cách chơi có thể bổ trợ nhau"
            ),
            f"Vị trí yêu thích tương đồng {position_score:.0f}%",
            f"Kinh nghiệm và phong độ tương đồng {experience_score:.0f}%",
        ],
    )
    return breakdown, distance, elo_difference


def _minutes(value: time) -> int:
    return value.hour * 60 + value.minute


def _skill_score(
    first_profile: MatchParticipantProfile,
    second_profile: MatchParticipantProfile,
) -> float:
    if not isinstance(first_profile, PlayerProfile) or not isinstance(second_profile, PlayerProfile):
        return 70.0
    difference = abs(
        SKILL_ORDER[first_profile.skill_level] - SKILL_ORDER[second_profile.skill_level]
    )
    return (100.0, 75.0, 45.0, 20.0)[difference]


def _position_score(
    first_profile: MatchParticipantProfile,
    second_profile: MatchParticipantProfile,
) -> float:
    if not isinstance(first_profile, PlayerProfile) or not isinstance(second_profile, PlayerProfile):
        return 70.0
    first = {_normalize_position(value) for value in first_profile.preferred_positions if value.strip()}
    second = {_normalize_position(value) for value in second_profile.preferred_positions if value.strip()}
    if not first and not second:
        return 70.0
    if not first or not second:
        return 65.0
    similarity = len(first & second) / len(first | second)
    return 50.0 + 50.0 * similarity


def _experience_score(
    first_profile: MatchParticipantProfile,
    second_profile: MatchParticipantProfile,
) -> float:
    if not isinstance(first_profile, PlayerProfile) or not isinstance(second_profile, PlayerProfile):
        return 70.0
    if first_profile.match_count == 0 or second_profile.match_count == 0:
        return 70.0
    match_count_base = max(first_profile.match_count, second_profile.match_count, 10)
    match_count_score = 100.0 * (
        1.0 - abs(first_profile.match_count - second_profile.match_count) / match_count_base
    )
    win_rate_score = 100.0 * (1.0 - abs(first_profile.win_rate - second_profile.win_rate))
    return max(0.0, (match_count_score + win_rate_score) / 2.0)


def _normalize_position(value: str) -> str:
    return " ".join(value.casefold().split())
