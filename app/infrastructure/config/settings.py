import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Goat Sports AI Service")
    API_V1_STR: str = "/api/v1/ai"
    PORT: int = _env_int("PORT", _env_int("AI_SERVICE_PORT", 7081))
    HOST: str = os.getenv("HOST", "0.0.0.0")

    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DATABASE: str = os.getenv("AI_MONGODB_DATABASE", "sports_ai")
    MONGODB_VECTOR_SEARCH_ENABLED: bool = _env_bool("MONGODB_VECTOR_SEARCH_ENABLED", False)
    MONGODB_VECTOR_INDEX: str = os.getenv("MONGODB_VECTOR_INDEX", "embedding_vector_index")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    VENUE_SERVICE_URL: str = os.getenv("VENUE_SERVICE_URL", "http://localhost:7072")
    SOCIAL_SERVICE_URL: str = os.getenv("SOCIAL_SERVICE_URL", "http://localhost:7080")
    NOTIFICATION_SERVICE_URL: str = os.getenv("NOTIFICATION_SERVICE_URL", "http://localhost:7073")
    HTTP_TIMEOUT_SECONDS: float = _env_float("HTTP_TIMEOUT_SECONDS", 5.0)

    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "").rstrip("/")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "")
    LLM_TEMPERATURE: float = _env_float("LLM_TEMPERATURE", 0.2)
    LLM_TIMEOUT_SECONDS: float = _env_float("LLM_TIMEOUT_SECONDS", 120.0)
    LLM_MAX_TOKENS: int = _env_int("LLM_MAX_TOKENS", 256)
    LLM_REASONING_EFFORT: str = os.getenv("LLM_REASONING_EFFORT", "none")
    RAG_MAX_CONTEXT_CHUNKS: int = _env_int("RAG_MAX_CONTEXT_CHUNKS", 5)
    RAG_MIN_RELEVANCE_SCORE: float = _env_float("RAG_MIN_RELEVANCE_SCORE", 0.08)

    MATCHMAKING_QUEUE_TTL_SECONDS: int = _env_int("MATCHMAKING_QUEUE_TTL_SECONDS", 900)
    MATCHMAKING_SESSION_TTL_SECONDS: int = _env_int("MATCHMAKING_SESSION_TTL_SECONDS", 600)
    MATCHMAKING_MIN_SCORE: float = _env_float("MATCHMAKING_MIN_SCORE", 50.0)
    MATCHMAKING_SCAN_LIMIT: int = _env_int("MATCHMAKING_SCAN_LIMIT", 200)

settings = Settings()
