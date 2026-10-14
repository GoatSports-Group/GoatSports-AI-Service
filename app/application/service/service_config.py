from dataclasses import dataclass


@dataclass(frozen=True)
class MatchmakingConfig:
    queue_ttl_seconds: int
    session_ttl_seconds: int
    minimum_score: float
    scan_limit: int
    result_window_seconds: int = 86_400


@dataclass(frozen=True)
class VectorSearchConfig:
    enabled: bool
    index_name: str


@dataclass(frozen=True)
class RagConfig:
    max_context_chunks: int
    # Phần điểm khớp từ khóa; phần còn lại là độ gần nghĩa (embedding).
    lexical_weight: float = 0.3
