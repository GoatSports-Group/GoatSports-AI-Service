from __future__ import annotations

import logging
import math
from datetime import date, time
from uuid import UUID

import httpx

from app.application.dto.response.models import (
    DemandHistorySnapshot,
    MatchBookingSnapshot,
    VenueRecommendation,
)
from app.application.exception import (
    ConflictException,
    DownstreamServiceException,
    ResourceNotFoundException,
)
from app.application.port.client import ChatTurn, ToolCall
from app.application.service.matchmaking_scoring import haversine_distance
from app.domain.entity.models import ClubMatchProfile, MatchVenueOption, SideMember
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


def _log_failure(call: str, exception: Exception) -> None:
    """Ghi ro service nao tu choi va vi sao; neu khong, moi loi deu chi thanh mot dong 503 chung chung."""
    if isinstance(exception, httpx.HTTPStatusError):
        logger.error("%s failed: %s %s -> %s %s", call, exception.request.method, exception.request.url,
                     exception.response.status_code, exception.response.text[:300])
    else:
        logger.error("%s failed: %s: %s", call, type(exception).__name__, exception)


def _response_data(response: httpx.Response):
    """Unwrap the shared Spring response envelope and retain plain-response support."""
    payload = response.json()
    if isinstance(payload, dict) and "statusCode" in payload and "data" in payload:
        return payload["data"]
    return payload


def _error_message(response: httpx.Response, fallback: str) -> str:
    try:
        message = response.json().get("message")
    except ValueError:
        return fallback
    return message if isinstance(message, str) and message.strip() else fallback


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
        include_courts: bool = True,
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
                include_courts,
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exception:
            raise DownstreamServiceException("Venue service is unavailable") from exception

    def get_matchmaking_booking(self, session_id: UUID) -> MatchBookingSnapshot | None:
        try:
            with httpx.Client(timeout=self.settings.HTTP_TIMEOUT_SECONDS) as client:
                response = client.get(
                    f"{self.settings.VENUE_SERVICE_URL}/internal/matchmaking/bookings/{session_id}",
                    headers=_gateway_headers(self.settings),
                )
                if response.status_code in {400, 404}:
                    return None
                response.raise_for_status()
                return MatchBookingSnapshot.model_validate(_response_data(response))
        except (httpx.HTTPError, TypeError, ValueError) as exception:
            raise DownstreamServiceException("Venue service is unavailable") from exception

    def nearby_external_venues(
        self,
        latitude: float,
        longitude: float,
        sport_type: SportType,
        radius_km: float,
        max_results: int = 3,
    ) -> list[MatchVenueOption]:
        lat_delta = radius_km / 111.0
        lng_delta = radius_km / (111.0 * max(math.cos(math.radians(latitude)), 0.01))
        try:
            with httpx.Client(timeout=self.settings.HTTP_TIMEOUT_SECONDS) as client:
                response = client.get(
                    f"{self.settings.VENUE_SERVICE_URL}/api/v1/venues/external",
                    headers=_gateway_headers(self.settings),
                    params={
                        "south": latitude - lat_delta,
                        "west": longitude - lng_delta,
                        "north": latitude + lat_delta,
                        "east": longitude + lng_delta,
                        "sportType": sport_type.value,
                        "limit": 200,
                    },
                )
                response.raise_for_status()
                venues = _response_data(response) or []
            nearby = []
            for venue in venues:
                distance = haversine_distance(
                    latitude, longitude, float(venue["latitude"]), float(venue["longitude"])
                )
                if distance <= radius_km:
                    nearby.append((distance, venue))
            nearby.sort(key=lambda item: item[0])
            return [
                MatchVenueOption(
                    venueId=UUID(str(venue["externalVenueId"])),
                    venueName=venue.get("name") or "Sân thể thao",
                    address=(venue.get("address") or "")[:500],
                    distanceKm=round(distance, 2),
                    score=round(max(0.0, 60 - distance * 4), 1),
                    matchReason=(
                        f"Sân ngoài hệ thống, cách vị trí nhóm {distance:.1f} km. "
                        "Hãy gọi sân để hỏi giờ trống và đặt trực tiếp."
                    ),
                    external=True,
                    phone=venue.get("phone"),
                    latitude=float(venue["latitude"]),
                    longitude=float(venue["longitude"]),
                )
                for distance, venue in nearby[:max_results]
            ]
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exception:
            _log_failure("nearby_external_venues", exception)
            raise DownstreamServiceException("Venue service is unavailable") from exception

    def owner_demand_history(self, owner_id: UUID, venue_id: UUID, hours: int) -> DemandHistorySnapshot:
        try:
            with httpx.Client(timeout=self.settings.HTTP_TIMEOUT_SECONDS) as client:
                response = client.get(
                    f"{self.settings.VENUE_SERVICE_URL}/internal/owner-demand/{venue_id}",
                    params={"ownerId": str(owner_id), "hours": hours},
                    headers=_gateway_headers(self.settings),
                )
                if response.status_code in {400, 403, 404}:
                    raise ResourceNotFoundException(_error_message(response, "Không tìm thấy cơ sở của bạn."))
                response.raise_for_status()
                return DemandHistorySnapshot.model_validate(_response_data(response))
        except (httpx.HTTPError, TypeError, ValueError) as exception:
            _log_failure("owner_demand_history", exception)
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
        include_courts: bool = True,
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
                # Gợi ý ở trang chủ không cần sân con / giờ trống: bỏ một lượt gọi venue-service cho mỗi sân.
                court_names, available_court, slot = (
                    self._courts(client, venue_id, sport_type, play_date, start_time, end_time)
                    if include_courts else ([], None, None)
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
                        total_reviews=int(venue.get("totalReviews") or 0),
                        min_price=venue.get("minPrice"),
                        max_price=venue.get("maxPrice"),
                        score=round(max(0.0, min(100.0, rating * 15 + 30 - distance * 2)), 1),
                        match_reason=(
                            (f"Còn trống {slot[0].strftime('%H:%M')}–{slot[1].strftime('%H:%M')}. " if slot else "")
                            + f"Được đánh giá {rating:.1f}/5 và cách vị trí nhóm {distance:.1f} km."
                        ),
                        suggested_courts=court_names,
                        slot_start=slot[0] if slot else None,
                        slot_end=slot[1] if slot else None,
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
    ) -> tuple[list[str], UUID | None, tuple[time, time] | None]:
        """San va slot som nhat con trong nam tron trong [start_time, end_time].

        Khung gio la khoang ranh cua hai ben, khong phai do dai tran: tran dau dai dung mot slot cua san
        (chu san tu dat 30-240 phut), nen khung 18:00-20:00 khop slot 18:00-19:00 hoac 19:00-20:00.
        """
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
            return names, (UUID(str(matching[0]["venueCourtId"])) if matching else None), None
        best: tuple[time, time, UUID] | None = None
        for court in matching:
            court_id = UUID(str(court["venueCourtId"]))
            slot_response = client.get(
                f"{self.settings.VENUE_SERVICE_URL}/api/v1/venue-courts/{court_id}/slots",
                headers=_gateway_headers(self.settings),
                params={"date": play_date.isoformat()},
            )
            slot_response.raise_for_status()
            for slot in _response_data(slot_response):
                slot_start = time.fromisoformat(slot["startTime"])
                slot_end = time.fromisoformat(slot["endTime"])
                if (
                    slot.get("status") == "AVAILABLE"
                    and start_time <= slot_start < slot_end <= end_time
                    and (best is None or slot_start < best[0])
                ):
                    best = (slot_start, slot_end, court_id)
        if best is None:
            return names, None, None
        return names, best[2], (best[0], best[1])


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

    def publish_highlight(self, payload: dict) -> UUID:
        try:
            with httpx.Client(timeout=self.settings.HTTP_TIMEOUT_SECONDS) as client:
                response = client.post(
                    f"{self.settings.SOCIAL_SERVICE_URL}/internal/posts/highlights",
                    headers=_gateway_headers(self.settings),
                    json=payload,
                )
                if 400 <= response.status_code < 500:
                    # Lỗi nghiệp vụ (vd. đã khoe trận này rồi): đưa nguyên lời nhắn về cho người dùng.
                    raise ConflictException(_error_message(response, "Không đăng được lên Cộng đồng."))
                response.raise_for_status()
                return UUID(str(_response_data(response)["postId"]))
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exception:
            _log_failure("publish_highlight", exception)
            raise DownstreamServiceException("Cộng đồng đang bận, vui lòng thử lại sau ít phút.") from exception

    def friend_ids(self, user_id: UUID) -> set[UUID]:
        try:
            with httpx.Client(timeout=self.settings.HTTP_TIMEOUT_SECONDS) as client:
                response = client.get(
                    f"{self.settings.SOCIAL_SERVICE_URL}/api/v1/social/friends",
                    headers=_gateway_headers(self.settings, user_id),
                    params={"userId": str(user_id)},
                )
                response.raise_for_status()
                friendships = _response_data(response) or []
        except (httpx.HTTPError, TypeError, ValueError) as exception:
            _log_failure("friend_ids", exception)
            raise DownstreamServiceException("Social service is unavailable") from exception
        me = str(user_id)
        ids: set[UUID] = set()
        # Danh sách trả cả lời mời đang chờ, nên phải lọc đúng quan hệ đã chấp nhận.
        for item in friendships:
            if item.get("status") != "ACCEPTED":
                continue
            other = item.get("addresseeId") if item.get("requesterId") == me else item.get("requesterId")
            if other and other != me:
                ids.add(UUID(str(other)))
        return ids


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
        player_elo: int | None = None,
        opponent_name: str | None = None,
        my_score: int | None = None,
        opponent_score: int | None = None,
    ) -> int:
        if not self.settings.AUTH_SERVICE_URL:
            raise DownstreamServiceException("Auth service URL is not configured")
        url = f"{self.settings.AUTH_SERVICE_URL}/internal/match-results"
        try:
            with httpx.Client(timeout=self.settings.HTTP_TIMEOUT_SECONDS) as client:
                response = client.post(
                    url,
                    headers=_gateway_headers(self.settings, user_id),
                    json={
                        "sessionId": str(session_id),
                        "userId": str(user_id),
                        "sportType": sport_type.value,
                        "opponentElo": opponent_elo,
                        "outcome": outcome,
                        "playerElo": player_elo,
                        # Chỉ để auth-service ghi lịch sử thi đấu (đấu với ai, tỷ số).
                        "opponentName": opponent_name,
                        "myScore": my_score,
                        "opponentScore": opponent_score,
                    },
                )
                response.raise_for_status()
                return int(_response_data(response)["newElo"])
        except httpx.HTTPStatusError as exception:
            # Ghi cả status lẫn body: nếu không, lỗi 4xx của auth-service bị nuốt
            # thành một câu 503 chung chung, không thể chẩn đoán được.
            logger.error(
                "apply_match_result failed: POST %s -> %s %s (session=%s, user=%s, outcome=%s)",
                url,
                exception.response.status_code,
                exception.response.text[:500],
                session_id,
                user_id,
                outcome,
            )
            raise DownstreamServiceException("Auth service is unavailable") from exception
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exception:
            logger.error(
                "apply_match_result failed: POST %s -> %s: %s (session=%s, user=%s)",
                url,
                type(exception).__name__,
                exception,
                session_id,
                user_id,
            )
            raise DownstreamServiceException("Auth service is unavailable") from exception

    def teammate(self, user_id: UUID, sport_type: SportType) -> SideMember:
        try:
            with httpx.Client(timeout=self.settings.HTTP_TIMEOUT_SECONDS) as client:
                headers = _gateway_headers(self.settings)
                profile = client.post(
                    f"{self.settings.AUTH_SERVICE_URL}/internal/sport-profiles/ensure",
                    headers=headers,
                    json={"userId": str(user_id), "sportType": sport_type.value},
                )
                profile.raise_for_status()
                user = client.get(f"{self.settings.AUTH_SERVICE_URL}/internal/users/{user_id}", headers=headers)
                user.raise_for_status()
                user_data = _response_data(user)
                return SideMember(
                    user_id=user_id,
                    name=user_data.get("fullName") or user_data.get("username") or "Người chơi",
                    elo_rating=int(_response_data(profile)["eloRating"]),
                )
        except (httpx.HTTPError, KeyError, TypeError, ValueError, AttributeError) as exception:
            _log_failure("teammate", exception)
            raise DownstreamServiceException("Auth service is unavailable") from exception


class HttpClubServiceClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def matchmaking_profile(
        self, club_id: UUID, play_format: str, latitude: float, longitude: float
    ) -> ClubMatchProfile | None:
        try:
            with httpx.Client(timeout=self.settings.HTTP_TIMEOUT_SECONDS) as client:
                response = client.get(
                    f"{self.settings.CLUB_SERVICE_URL}/internal/clubs/{club_id}/matchmaking-profile",
                    headers=_gateway_headers(self.settings),
                )
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                data = _response_data(response)
                return ClubMatchProfile(
                    participant_id=UUID(str(data["clubId"])),
                    name=data["name"],
                    avatar_url=data.get("logoUrl"),
                    sport_type=SportType(data["sportType"]),
                    elo_rating=int(data.get("eloRating") or 1200),
                    latitude=latitude,
                    longitude=longitude,
                    active_member_count=int(data.get("activeMemberCount") or 0),
                    preferred_format=play_format,
                    play_format=play_format,
                    match_count=int(data.get("matchCount") or 0),
                    win_rate=float(data.get("winRate") or 0),
                    manager_ids=[UUID(str(item)) for item in data.get("managerIds") or []],
                    member_ids=[UUID(str(item)) for item in data.get("memberIds") or []],
                )
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exception:
            _log_failure("club matchmaking_profile", exception)
            raise DownstreamServiceException("Club service is unavailable") from exception

    def apply_friendly_result(
        self,
        session_id: UUID,
        sport_type: SportType,
        play_format: str | None,
        first: tuple[UUID, int],
        second: tuple[UUID, int],
    ) -> dict[str, int]:
        try:
            with httpx.Client(timeout=self.settings.HTTP_TIMEOUT_SECONDS) as client:
                response = client.post(
                    f"{self.settings.CLUB_SERVICE_URL}/internal/clubs/friendly-results",
                    headers=_gateway_headers(self.settings),
                    json={
                        "sessionId": str(session_id),
                        "sportType": sport_type.value,
                        "playFormat": play_format,
                        "first": {"clubId": str(first[0]), "score": first[1]},
                        "second": {"clubId": str(second[0]), "score": second[1]},
                    },
                )
                response.raise_for_status()
                return {str(key): int(value) for key, value in _response_data(response).items()}
        except (httpx.HTTPError, AttributeError, TypeError, ValueError) as exception:
            _log_failure("apply_friendly_result", exception)
            raise DownstreamServiceException("Club service is unavailable") from exception


class OpenAiCompatibleLanguageModelClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _extras(self) -> dict:
        think = self.settings.LLM_THINK
        if think in {"true", "false"}:
            return {"think": think == "true"}
        return {"think": False} if ":11434" in self.settings.LLM_BASE_URL else {}

    @property
    def model_name(self) -> str | None:
        return self.settings.LLM_MODEL or None

    def complete(self, system: str, prompt: str, max_tokens: int | None = None) -> str | None:
        if not self.settings.LLM_BASE_URL or not self.settings.LLM_MODEL:
            return None
        headers = {"Content-Type": "application/json"}
        if self.settings.LLM_API_KEY:
            headers["Authorization"] = f"Bearer {self.settings.LLM_API_KEY}"
        try:
            with httpx.Client(timeout=self.settings.MATCH_INSIGHT_TIMEOUT_SECONDS) as client:
                response = client.post(
                    f"{self.settings.LLM_BASE_URL}/chat/completions",
                    headers=headers,
                    json={
                        "model": self.settings.LLM_MODEL,
                        "temperature": self.settings.LLM_TEMPERATURE,
                        "max_tokens": max_tokens or self.settings.LLM_MAX_TOKENS,
                        "reasoning_effort": self.settings.LLM_REASONING_EFFORT,
                        **self._extras(),
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                    },
                )
                response.raise_for_status()
                return str(response.json()["choices"][0]["message"]["content"]).strip()
        except (httpx.HTTPError, KeyError, IndexError, TypeError) as exception:
            logger.warning("LLM completion failed: %s", exception)
            return None



    def chat(self, messages: list[dict], tools: list[dict] | None, max_tokens: int) -> ChatTurn | None:
        if not self.settings.LLM_BASE_URL or not self.settings.LLM_MODEL:
            return None
        headers = {"Content-Type": "application/json"}
        if self.settings.LLM_API_KEY:
            headers["Authorization"] = f"Bearer {self.settings.LLM_API_KEY}"
        body = {
            "model": self.settings.LLM_MODEL,
            "temperature": self.settings.LLM_TEMPERATURE,
            "max_tokens": max_tokens,
            "reasoning_effort": self.settings.LLM_REASONING_EFFORT,
            **self._extras(),
            "messages": messages,
        }
        if tools:
            body["tools"] = tools
        try:
            with httpx.Client(timeout=self.settings.LLM_TIMEOUT_SECONDS) as client:
                response = client.post(f"{self.settings.LLM_BASE_URL}/chat/completions", headers=headers, json=body)
                response.raise_for_status()
                message = response.json()["choices"][0]["message"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exception:
            logger.warning("LLM chat failed: %s", exception)
            return None
        calls = [
            ToolCall(
                call_id=str(call.get("id") or f"call_{index}"),
                name=str(call["function"]["name"]),
                arguments=str(call["function"].get("arguments") or "{}"),
            )
            for index, call in enumerate(message.get("tool_calls") or [])
            if call.get("function", {}).get("name")
        ]
        content = message.get("content")
        return ChatTurn(content=str(content).strip() if content else None, tool_calls=calls)


# Compatibility names used by the former flat package.
VenueServiceClient = HttpVenueServiceClient
SocialServiceClient = HttpSocialServiceClient
NotificationServiceClient = HttpNotificationServiceClient
