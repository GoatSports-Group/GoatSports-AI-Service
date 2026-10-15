"""Đọc dữ liệu thật cho trợ lý GOAT AI từ venue-service và club-service (chỉ đọc).

Gọi thẳng service (không qua gateway) kèm X-Gateway-Secret như mọi lời gọi nội bộ khác. "Đơn của tôi" chuyển
tiếp nguyên danh tính gateway đã gắn cho câu hỏi chatbot, nên venue-service tự kiểm quyền như khi người chơi tự xem.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import httpx

from app.application.port.client import CallerIdentity, PlatformDataError
from app.infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)

# Trợ lý đang chờ trả lời người dùng: dịch vụ chậm thì báo lỗi nhanh thay vì treo cả câu trả lời.
_TIMEOUT_SECONDS = 8.0


class HttpPlatformDataClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def search_venues(self, keyword, sport_type, district, max_price, sort, limit) -> tuple[list[dict[str, Any]], int]:
        params = {"page": 0, "size": limit}
        for name, value in (("keyword", keyword), ("sportType", sport_type), ("district", district),
                            ("maxPrice", max_price), ("sort", sort)):
            if value is not None:
                params[name] = value
        body = self._get(f"{self.settings.VENUE_SERVICE_URL}/api/v1/venues", params)
        return _items(body), _total(body)

    def free_slots(self, venue_keyword: str, play_date: date, sport_type: str | None) -> dict[str, Any] | None:
        venues, _ = self.search_venues(venue_keyword, sport_type, None, None, None, 3)
        if not venues:
            return None
        venue = venues[0]
        base = self.settings.VENUE_SERVICE_URL
        courts = [
            court
            for court in self._get(f"{base}/api/v1/venues/{venue['venueId']}/courts") or []
            if court.get("active", True) and (sport_type is None or court.get("sportType") == sport_type)
        ]
        result = []
        for court in courts[:8]:
            slots = self._get(f"{base}/api/v1/venue-courts/{court['venueCourtId']}/slots", {"date": play_date.isoformat()}) or []
            free = sorted(
                ({"start": slot["startTime"], "end": slot["endTime"], "price": slot.get("pricePerHour")}
                 for slot in slots if slot.get("status") == "AVAILABLE"),
                key=lambda slot: slot["start"],
            )
            result.append({"court": court.get("name", "Sân"), "slots": free})
        return {"venue_id": venue["venueId"], "venue_name": venue.get("name", "Cơ sở"), "courts": result}

    def search_tournaments(self, sport_type, status, keyword, limit) -> list[dict[str, Any]]:
        params = {"page": 0, "size": limit}
        for name, value in (("sportType", sport_type), ("status", status), ("keyword", keyword)):
            if value is not None:
                params[name] = value
        return _items(self._get(f"{self.settings.CLUB_SERVICE_URL}/api/v1/tournaments/search", params))

    def my_bookings(self, identity: CallerIdentity, status: str, limit: int) -> list[dict[str, Any]]:
        body = self._get(
            f"{self.settings.VENUE_SERVICE_URL}/api/v1/bookings/my-history",
            {"status": status, "page": 0, "size": limit},
            _identity_headers(identity),
        )
        return _items(body)

    def my_clubs(self, identity: CallerIdentity, activity_limit: int):
        base = f"{self.settings.CLUB_SERVICE_URL}/api/v1/clubs/me"
        clubs = _items(self._get(base, None, _identity_headers(identity)))
        activities = _items(self._get(f"{base}/activities", {"limit": activity_limit}, _identity_headers(identity)))
        return clubs, activities

    def search_clubs(self, sport_type, city, keyword, limit):
        params = {"page": 0, "size": limit}
        for name, value in (("sportType", sport_type), ("city", city), ("keyword", keyword)):
            if value is not None:
                params[name] = value
        body = self._get(f"{self.settings.CLUB_SERVICE_URL}/api/v1/clubs/search", params)
        return _items(body), _total(body)

    def _get(self, url: str, params: dict | None = None, extra_headers: dict | None = None):
        headers = {"X-Gateway-Secret": self.settings.GATEWAY_SECRET, **(extra_headers or {})}
        try:
            with httpx.Client(timeout=_TIMEOUT_SECONDS) as client:
                response = client.get(url, params=params, headers=headers)
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exception:
            logger.warning("Assistant data call failed for %s: %s", url.split("?")[0], exception)
            raise PlatformDataError(str(exception)) from exception
        if isinstance(payload, dict) and "statusCode" in payload and "data" in payload:
            return payload["data"]
        return payload


def _identity_headers(identity: CallerIdentity) -> dict[str, str]:
    """Chuyển tiếp nguyên danh tính gateway đã gắn: service đích tự kiểm quyền như khi người dùng tự gọi."""
    headers = {
        "X-User-Id": str(identity.user_id),
        "X-User-Email": identity.email or "",
        "X-User-Username": identity.username or "",
        "X-User-Roles": identity.roles or "",
        "X-User-Authorities": identity.authorities or "",
    }
    return {key: value for key, value in headers.items() if value}


def _total(body: Any) -> int:
    """Tổng số bản ghi khớp (không chỉ trang đầu): PageResult{total}, Spring Page{totalElements|page.totalElements}."""
    if isinstance(body, dict):
        for key in ("total", "totalElements"):
            if isinstance(body.get(key), int):
                return body[key]
        page = body.get("page") or body.get("meta")
        if isinstance(page, dict):
            for key in ("totalElements", "total"):
                if isinstance(page.get(key), int):
                    return page[key]
    return len(_items(body))


def _items(body: Any) -> list[dict[str, Any]]:
    """Các kiểu trang khác nhau của các service: PageResult{items}, BaseListResponse{result}, Spring Page{content}."""
    if isinstance(body, list):
        return body
    if isinstance(body, dict):
        for key in ("items", "result", "content"):
            if isinstance(body.get(key), list):
                return body[key]
    return []
