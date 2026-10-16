"""Cổng cho trợ lý GOAT AI: gọi LLM kèm công cụ, và đọc dữ liệu thật của nền tảng (chỉ đọc)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol
from uuid import UUID


@dataclass(frozen=True)
class CallerIdentity:
    """Danh tính người hỏi do gateway gắn (đã kiểm X-Gateway-Secret). Không bao giờ lấy từ câu hỏi hay từ LLM."""

    user_id: UUID
    email: str | None = None
    username: str | None = None
    roles: str | None = None
    authorities: str | None = None


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: str  # JSON do LLM sinh, phải kiểm lại trước khi dùng


@dataclass(frozen=True)
class ChatTurn:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass(frozen=True)
class ExternalPassage:
    """Một đoạn kiến thức chung lấy từ nguồn ngoài hệ thống (vd. Wikipedia): chỉ để bù, không thay tài liệu hệ thống."""

    title: str
    summary: str
    url: str
    source: str = "Wikipedia tiếng Việt"


class ExternalKnowledgeClient(Protocol):
    def lookup(self, query: str, limit: int, language: str = "vi") -> list[ExternalPassage]:
        """language: "vi" hoặc "en" (bài tiếng Việt thường thiếu, vd. luật bóng đá 5 người chỉ có bài tiếng Anh).
        Rỗng khi không tìm thấy; ném PlatformDataError khi nguồn ngoài không dùng được."""
        ...


class PlatformDataError(Exception):
    """Service nguồn lỗi hoặc không phản hồi: trợ lý nói rõ là chưa lấy được dữ liệu, không đoán."""


class PlatformDataClient(Protocol):
    def search_venues(
        self,
        keyword: str | None,
        sport_type: str | None,
        district: str | None,
        max_price: float | None,
        sort: str | None,
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Các sân của trang đầu và TỔNG số sân khớp bộ lọc. sort: None (mặc định) | "price" | "rating"."""
        ...

    def free_slots(self, venue_keyword: str, play_date: date, sport_type: str | None) -> dict[str, Any] | None:
        """Cơ sở khớp tên đầu tiên và các khung giờ còn trống theo từng sân con; None nếu không tìm thấy cơ sở."""
        ...

    def search_tournaments(
        self, sport_type: str | None, status: str | None, keyword: str | None, limit: int
    ) -> list[dict[str, Any]]: ...

    def my_bookings(self, identity: CallerIdentity, status: str, limit: int) -> list[dict[str, Any]]: ...

    def my_clubs(self, identity: CallerIdentity, activity_limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """CLB người hỏi đang là thành viên và các hoạt động CLB sắp tới của họ."""
        ...

    def search_clubs(
        self, sport_type: str | None, city: str | None, keyword: str | None, limit: int
    ) -> tuple[list[dict[str, Any]], int]: ...

    def my_sport_profiles(self, identity: CallerIdentity) -> list[dict[str, Any]]:
        """Hồ sơ thể thao của người hỏi (môn, trình độ, ELO, thành phố, giờ rảnh) từ auth-service."""
        ...
