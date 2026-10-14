from __future__ import annotations

from typing import Protocol


class TextEmbedder(Protocol):
    """Biến chữ thành vector đã chuẩn hóa (độ dài 1), để so độ gần nghĩa bằng tích vô hướng."""

    @property
    def model_name(self) -> str:
        """Tên kèm phiên bản; đổi tên thì kho tri thức được nhúng lại lúc khởi động."""
        ...

    @property
    def relevance_floor(self) -> float:
        """Độ gần nghĩa tối thiểu để một đoạn được coi là liên quan (mỗi model có thang điểm riêng)."""
        ...

    def embed_query(self, text: str) -> list[float]: ...

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...
