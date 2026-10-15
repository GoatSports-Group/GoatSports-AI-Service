from __future__ import annotations

from typing import Protocol


class ConversationStore(Protocol):
    """Trí nhớ ngắn của trợ lý (vài lượt gần nhất, tự hết hạn) và giới hạn số câu hỏi mỗi phút."""

    def load(self, key: str) -> list[dict[str, str]]:
        """Các lượt cũ dạng {"role": "user"|"assistant", "content": ...}, cũ trước."""
        ...

    def append(self, key: str, question: str, answer: str) -> None: ...

    def allow(self, key: str, limit_per_minute: int) -> bool:
        """Tính thêm một câu hỏi; False khi đã vượt giới hạn trong phút hiện tại."""
        ...
