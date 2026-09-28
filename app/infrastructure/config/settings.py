import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "")


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    PROJECT_NAME: str = "GOATSPORTS-AI-SERVICE"
    API_V1_STR: str = "/api/v1/ai"
    PORT: int = _env_int("AI_SERVICE_PORT", 7081)
    HOST: str = os.getenv("AI_SERVICE_HOST", "[IP_ADDRESS]")

    MONGODB_URI: str = os.getenv("AI_MONGODB_URI")
    MONGODB_DATABASE: str = os.getenv("AI_MONGODB_DATABASE", "")
    MONGODB_VECTOR_SEARCH_ENABLED: bool = _env_bool("MONGODB_VECTOR_SEARCH_ENABLED", True)
    MONGODB_VECTOR_INDEX: str = os.getenv("MONGODB_VECTOR_INDEX", "")
    MONGODB_VECTOR_SEARCH_RETRY_SECONDS: float = _env_float(
        "MONGODB_VECTOR_SEARCH_RETRY_SECONDS", 60.0
    )
    REDIS_URL: str = os.getenv("REDIS_URL", "")

    VENUE_SERVICE_URL: str = os.getenv("VENUE_SERVICE_URL", "")
    SOCIAL_SERVICE_URL: str = os.getenv("SOCIAL_SERVICE_URL", "")
    NOTIFICATION_SERVICE_URL: str = os.getenv("NOTIFICATION_SERVICE_URL", "")
    AUTH_SERVICE_URL: str = os.getenv("AUTH_SERVICE_URL", "")
    CLUB_SERVICE_URL: str = os.getenv("CLUB_SERVICE_URL", "")
    GATEWAY_SECRET: str = os.getenv("GATEWAY_SECRET", "")
    HTTP_TIMEOUT_SECONDS: float = _env_float("HTTP_TIMEOUT_SECONDS", 30.0)

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
    MATCHMAKING_RESULT_WINDOW_SECONDS: int = _env_int("MATCHMAKING_RESULT_WINDOW_SECONDS", 86400)

    IDENTITY_VERIFICATION_TTL_SECONDS: int = _env_int(
        "IDENTITY_VERIFICATION_TTL_SECONDS", 3600
    )
    IDENTITY_DOWNLOAD_TIMEOUT_SECONDS: float = _env_float(
        "IDENTITY_DOWNLOAD_TIMEOUT_SECONDS", 10.0
    )
    IDENTITY_MAX_IMAGE_BYTES: int = _env_int("IDENTITY_MAX_IMAGE_BYTES", 2 * 1024 * 1024)
    IDENTITY_ALLOWED_OBJECT_HOST_SUFFIXES: tuple[str, ...] = tuple(
        value.strip().lower()
        for value in os.getenv(
            "IDENTITY_ALLOWED_OBJECT_HOST_SUFFIXES",
            "",
        ).split(",")
        if value.strip()
    )
    IDENTITY_MIN_IMAGE_EDGE: int = _env_int("IDENTITY_MIN_IMAGE_EDGE", 480)
    IDENTITY_MIN_BLUR_SCORE: float = _env_float("IDENTITY_MIN_BLUR_SCORE", 45.0)
    IDENTITY_BLUR_TARGET: float = _env_float("IDENTITY_BLUR_TARGET", 180.0)
    IDENTITY_MAX_GLARE_RATIO: float = _env_float("IDENTITY_MAX_GLARE_RATIO", 0.12)
    IDENTITY_FACE_MATCH_THRESHOLD: float = _env_float(
        "IDENTITY_FACE_MATCH_THRESHOLD", 0.45
    )
    IDENTITY_OCR_TEXT_DETECTION_MODEL: str = os.getenv(
        "IDENTITY_OCR_TEXT_DETECTION_MODEL", ""
    )
    IDENTITY_OCR_TEXT_RECOGNITION_MODEL: str = os.getenv(
        "IDENTITY_OCR_TEXT_RECOGNITION_MODEL", ""
    )
    IDENTITY_OCR_ENABLE_MKLDNN: bool = _env_bool("IDENTITY_OCR_ENABLE_MKLDNN", False)
    IDENTITY_OCR_CPU_THREADS: int = _env_int("IDENTITY_OCR_CPU_THREADS", 8)
    IDENTITY_OCR_RECOGNITION_BATCH_SIZE: int = _env_int(
        "IDENTITY_OCR_RECOGNITION_BATCH_SIZE", 32
    )
    IDENTITY_OCR_MAX_SIDE: int = _env_int("IDENTITY_OCR_MAX_SIDE", 960)
    IDENTITY_OCR_WORKER_PROCESSES: int = _env_int("IDENTITY_OCR_WORKER_PROCESSES", 1)
    IDENTITY_VISION_WARMUP_ENABLED: bool = _env_bool("IDENTITY_VISION_WARMUP_ENABLED", True)
    FACE_DETECTOR_MODEL_PATH: str = os.getenv(
        "FACE_DETECTOR_MODEL_PATH", ""
    )
    FACE_RECOGNITION_MODEL_PATH: str = os.getenv(
        "FACE_RECOGNITION_MODEL_PATH", ""
    )
    FACE_ATTRIBUTE_MODEL_PATH: str = os.getenv(
        "FACE_ATTRIBUTE_MODEL_PATH", ""
    )
    FACE_ATTRIBUTE_GLASSES_THRESHOLD: float = _env_float(
        "FACE_ATTRIBUTE_GLASSES_THRESHOLD", 0.55
    )
    FACE_ATTRIBUTE_MASK_THRESHOLD: float = _env_float("FACE_ATTRIBUTE_MASK_THRESHOLD", 0.55)
    FACE_ATTRIBUTE_EYE_OPEN_THRESHOLD: float = _env_float(
        "FACE_ATTRIBUTE_EYE_OPEN_THRESHOLD", 0.45
    )
    LIVENESS_MODEL_PATH: str = os.getenv(
        "LIVENESS_MODEL_PATH", ""
    )
    LIVENESS_LIVE_CLASS_INDEX: int = _env_int("LIVENESS_LIVE_CLASS_INDEX", 0)
    LIVENESS_THRESHOLD: float = _env_float("LIVENESS_THRESHOLD", 0.80)

settings = Settings()