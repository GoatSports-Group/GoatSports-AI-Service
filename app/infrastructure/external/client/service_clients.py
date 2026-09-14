from __future__ import annotations

import logging
from datetime import date, time
from uuid import UUID

import httpx

from app.application.dto.response.models import MatchBookingSnapshot, VenueRecommendation
from app.application.exception import DownstreamServiceException
from app.application.service.matchmaking_scoring import haversine_distance
from app.domain.enums import SportType
from app.infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)


def _gateway_headers(settings: Settings, user_id: UUID | None = None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if settings.GATEWAY_SECRET:
        headers["X-Gateway-Secret"] = settings.GATEWAY_SECRET
    if user_id is not None:
        headers["X-User-Id"] = str(user_id)
    return headers


def _response_data(response: httpx.Response):
    """Unwrap the shared Spring response envelope and retain plain-response support."""
    payload = response.json()
    if isinstance(payload, dict) and "statusCode" in payload and "data" in payload:
        return payload["data"]
    return payload


class HttpVenueServiceClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def recommend(
        self,
        latitude: float,
        longitude: float,
        sport_type: SportType,
        radius_km: float,
        play_date: date | None = None,
        start_time: time | None = None,
        end_time: time | None = None,
        max_results: int = 5,
    ) -> tuple[list[VenueRecommendation], UUID | None]:
        try:
            return self._recommend(
                latitude,
                longitude,
                sport_type,
                radius_km,
                play_date,
                start_time,
                end_time,
                max_results,
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exception:
            raise DownstreamServiceException("Venue service is unavailable") from exception

    def get_matchmaking_booking(self, session_id: UUID) -> MatchBookingSnapshot | None:
        try:
            with httpx.Client(timeout=self.settings.HTTP_TIMEOUT_SECONDS) as client:
                response = client.get(
                    f"{self.settings.VENUE_SERVICE_URL}/internal/matchmaking/bookings/{session_id}"
                )
                if response.status_code in {400, 404}:
                    return None
                response.raise_for_status()
                return MatchBookingSnapshot.model_validate(response.json())
        except (httpx.HTTPError, TypeError, ValueError) as exception:
            raise DownstreamServiceException("Venue service is unavailable") from exception

    def _recommend(
        self,
        latitude: float,
        longitude: float,
        sport_type: SportType,
        radius_km: float,
        play_date: date | None,
        start_time: time | None,
        end_time: time | None,
        max_results: int,
    ) -> tuple[list[VenueRecommendation], UUID | None]:
        with httpx.Client(timeout=self.settings.HTTP_TIMEOUT_SECONDS) as client:
            response = client.get(
                f"{self.settings.VENUE_SERVICE_URL}/api/v1/venues",
                headers=_gateway_headers(self.settings),
                params={
                    "sportType": sport_type.value,
                    "latitude": latitude,
                    "longitude": longitude,
                    "radiusKm": radius_km,
                    "page": 0,
                    "size": max_results,
                },
            )
            response.raise_for_status()
            body = _response_data(response)
            venues = body.get("items", []) if isinstance(body, dict) else body
            recommendations: list[VenueRecommendation] = []
            courts_by_venue: dict[UUID, UUID | None] = {}
            for venue in venues:
                venue_id = UUID(str(venue["venueId"]))
                distance = venue.get("distanceKm")
                if distance is None and venue.get("latitude") is not None:
                    distance = haversine_distance(
                        latitude,
                        longitude,
                        float(venue["latitude"]),
                        float(venue["longitude"]),
                    )
                distance = float(distance or 0)
                rating = float(venue.get("averageRating") or 0)
                court_names, available_court = self._courts(
                    client, venue_id, sport_type, play_date, start_time, end_time
                )
                courts_by_venue[venue_id] = available_court
                recommendations.append(
                    VenueRecommendation(
                        venue_id=venue_id,
                        venue_court_id=available_court,
                        venue_name=venue.get("name", "Sân thể thao"),
                        sport_types=venue.get("sportTypes") or [],
                        address=", ".join(
                            str(part)
                            for part in (venue.get("address"), venue.get("district"), venue.get("city"))
                            if part
                        ),
                        distance_km=round(distance, 2),
                        rating=rating,
                        min_price=venue.get("minPrice"),
                        max_price=venue.get("maxPrice"),
                        score=round(max(0.0, min(100.0, rating * 15 + 30 - distance * 2)), 1),
                        match_reason=(
                            f"Phù hợp môn chơi, được đánh giá {rating:.1f}/5 "
                            f"và cách vị trí nhóm {distance:.1f} km."
                        ),
                        suggested_courts=court_names,
                    )
                )
            recommendations.sort(key=lambda item: item.score, reverse=True)
            selected = recommendations[:max_results]
            selected_court = courts_by_venue.get(selected[0].venue_id) if selected else None
            return selected, selected_court

    def _courts(
        self,
        client: httpx.Client,
        venue_id: UUID,
        sport_type: SportType,
        play_date: date | None,
        start_time: time | None,
        end_time: time | None,
    ) -> tuple[list[str], UUID | None]:
        response = client.get(
            f"{self.settings.VENUE_SERVICE_URL}/api/v1/venues/{venue_id}/courts",
            headers=_gateway_headers(self.settings),
        )
        response.raise_for_status()
        response_data = _response_data(response)
        matching = [
            court
            for court in response_data
            if court.get("active", True) and court.get("sportType") == sport_type.value
        ]
        names = [court.get("name", "Sân") for court in matching]
        if not play_date or not start_time or not end_time:
            return names, UUID(str(matching[0]["venueCourtId"])) if matching else None
        for court in matching:
            court_id = UUID(str(court["venueCourtId"]))
            slot_response = client.get(
                f"{self.settings.VENUE_SERVICE_URL}/api/v1/venue-courts/{court_id}/slots",
                headers=_gateway_headers(self.settings),
                params={"date": play_date.isoformat()},
            )
            slot_response.raise_for_status()
            slots = _response_data(slot_response)
            if any(
                slot.get("status") == "AVAILABLE"
                and time.fromisoformat(slot["startTime"]) <= start_time
                and time.fromisoformat(slot["endTime"]) >= end_time
                for slot in slots
            ):
                return names, court_id
        return names, None


class HttpSocialServiceClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def create_match_conversation(
        self,
        session_id: UUID,
        participant_ids: list[UUID],
        creator_id: UUID,
        sport_type: SportType,
    ) -> UUID:
        try:
            with httpx.Client(timeout=self.settings.HTTP_TIMEOUT_SECONDS) as client:
                response = client.post(
                    f"{self.settings.SOCIAL_SERVICE_URL}/api/v1/social/conversations/group",
                    headers=_gateway_headers(self.settings, creator_id),
                    json={
                        "type": "MATCH",
                        "name": f"Kèo {sport_type.value}",
                        "contextId": str(session_id),
                        "creatorId": str(creator_id),
                        "members": [{"userId": str(user_id)} for user_id in participant_ids],
                    },
                )
                response.raise_for_status()
                return UUID(str(_response_data(response)["conversationId"]))
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exception:
            raise DownstreamServiceException("Social service is unavailable") from exception


class HttpNotificationServiceClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def notify_matchmaking(
        self,
        receiver_id: UUID,
        session_id: UUID,
        title: str,
        content: str,
    ) -> None:
        try:
            with httpx.Client(timeout=self.settings.HTTP_TIMEOUT_SECONDS) as client:
                response = client.post(
                    f"{self.settings.NOTIFICATION_SERVICE_URL}/internal/notifications",
                    json={
                        "receiverId": str(receiver_id),
                        "title": title,
                        "content": content,
                        "type": "MATCHMAKING",
                        "sourceService": "AI",
                        "referenceType": "MATCHMAKING_SESSION",
                        "referenceId": str(session_id),
                    },
                )
                response.raise_for_status()
        except httpx.HTTPError as exception:
            raise DownstreamServiceException("Notification service is unavailable") from exception


class HttpAuthServiceClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def apply_match_result(
        self,
        session_id: UUID,
        user_id: UUID,
        sport_type: SportType,
        opponent_elo: int,
        outcome: str,
    ) -> int:
        if not self.settings.AUTH_SERVICE_URL:
            raise DownstreamServiceException("Auth service URL is not configured")
        try:
            with httpx.Client(timeout=self.settings.HTTP_TIMEOUT_SECONDS) as client:
                response = client.post(
                    f"{self.settings.AUTH_SERVICE_URL}/internal/match-results",
                    json={
                        "sessionId": str(session_id),
                        "userId": str(user_id),
                        "sportType": sport_type.value,
                        "opponentElo": opponent_elo,
                        "outcome": outcome,
                    },
                )
                response.raise_for_status()
                return int(response.json()["newElo"])
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exception:
            raise DownstreamServiceException("Auth service is unavailable") from exception


class OpenAiCompatibleLanguageModelClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def model_name(self) -> str | None:
        return self.settings.LLM_MODEL or None

    def generate(self, question: str, context: str) -> str | None:
        if not self.settings.LLM_BASE_URL or not self.settings.LLM_MODEL:
            return None
        headers = {"Content-Type": "application/json"}
        if self.settings.LLM_API_KEY:
            headers["Authorization"] = f"Bearer {self.settings.LLM_API_KEY}"
        try:
            with httpx.Client(timeout=self.settings.LLM_TIMEOUT_SECONDS) as client:
                response = client.post(
                    f"{self.settings.LLM_BASE_URL}/chat/completions",
                    headers=headers,
                    json={
                        "model": self.settings.LLM_MODEL,
                        "temperature": self.settings.LLM_TEMPERATURE,
                        "max_tokens": self.settings.LLM_MAX_TOKENS,
                        "reasoning_effort": self.settings.LLM_REASONING_EFFORT,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "Bạn là trợ lý thể thao GoatSports. Chỉ trả lời dựa trên ngữ cảnh "
                                    "được cung cấp, không bịa thông tin và trích dẫn [Nguồn N]."
                                ),
                            },
                            {"role": "user", "content": f"Ngữ cảnh:\n{context}\n\nCâu hỏi: {question}"},
                        ],
                    },
                )
                response.raise_for_status()
                return str(response.json()["choices"][0]["message"]["content"]).strip()
        except (httpx.HTTPError, KeyError, IndexError, TypeError) as exception:
            logger.warning("LLM generation failed; using extractive fallback: %s", exception)
            return None


# Compatibility names used by the former flat package.
VenueServiceClient = HttpVenueServiceClient
SocialServiceClient = HttpSocialServiceClient
NotificationServiceClient = HttpNotificationServiceClient
