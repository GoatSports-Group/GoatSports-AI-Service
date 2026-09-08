import math
from datetime import time

from app.domain.entity.models import MatchParticipantProfile, MatchPreference


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
    if overlapping_window(first_preference, second_preference) is None:
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
    overlap = overlapping_window(first_preference, second_preference)
    assert overlap is not None
    overlap_minutes = _minutes(overlap[1]) - _minutes(overlap[0])
    schedule_score = 100.0 * overlap_minutes / max(first_minutes, second_minutes)
    score = round(elo_score * 0.55 + distance_score * 0.30 + schedule_score * 0.15, 1)
    return score, distance, elo_difference


def _minutes(value: time) -> int:
    return value.hour * 60 + value.minute
