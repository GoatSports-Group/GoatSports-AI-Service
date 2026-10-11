from importlib.util import find_spec
from pathlib import Path

from fastapi import APIRouter

from app.application.dto.response.models import AiCapabilityStatus
from app.domain.enums import AiCapability
from app.infrastructure.config.container import demand_forecast_service
from app.infrastructure.config.settings import settings

router = APIRouter(prefix="/capabilities", tags=["AI Capabilities"])


@router.get("", response_model=list[AiCapabilityStatus], summary="Kiểm tra năng lực AI")
def get_capabilities() -> list[AiCapabilityStatus]:
    return capability_statuses()


def capability_statuses() -> list[AiCapabilityStatus]:
    """Kiểm tra thật cấu hình LLM, thư viện và file model — không ghi READY cứng."""
    rag_ready = bool(settings.LLM_BASE_URL and settings.LLM_MODEL)
    identity_dependencies_ready = all(
        find_spec(module) is not None for module in ("cv2", "numpy", "onnxruntime", "paddleocr")
    )
    identity_ready = identity_dependencies_ready and all(
        Path(path).is_file()
        for path in (
            settings.FACE_DETECTOR_MODEL_PATH,
            settings.FACE_RECOGNITION_MODEL_PATH,
            settings.LIVENESS_MODEL_PATH,
            settings.FACE_ATTRIBUTE_MODEL_PATH,
        )
    )
    return [
        AiCapabilityStatus(
            capability=AiCapability.MATCHMAKING,
            algorithm="Per-sport Elo + profile compatibility + Haversine + schedule overlap",
            storage="Redis queue + MongoDB session",
            status="READY",
            detail="Ghép người chơi theo môn, trình độ, vị trí và thời gian rảnh.",
        ),
        AiCapabilityStatus(
            capability=AiCapability.VENUE_RECOMMENDATION,
            algorithm="Cosine preference similarity + geographic ranking",
            storage="MongoDB interactions",
            status="READY",
            detail="Rerank tối đa 5 cụm sân theo thói quen, khoảng cách và đánh giá.",
        ),
        AiCapabilityStatus(
            capability=AiCapability.SEMANTIC_SEARCH,
            algorithm="Hybrid TF-IDF + vector embedding cosine",
            storage="MongoDB semantic documents + embedding records",
            status="READY",
            detail="Tìm sân, bài viết, hoạt động CLB và giải đấu theo ngữ nghĩa.",
        ),
        AiCapabilityStatus(
            capability=AiCapability.FEED_RANKING,
            algorithm="User-user collaborative filtering",
            storage="MongoDB user interactions",
            status="READY",
            detail="Xếp hạng bài viết từ view, like và comment, có cold-start fallback.",
        ),
        AiCapabilityStatus(
            capability=AiCapability.RAG,
            algorithm="Hybrid retrieval + grounded LLM generation",
            storage="MongoDB knowledge chunks + embedding records",
            status="READY" if rag_ready else "DEGRADED",
            detail=(
                "LLM đã được cấu hình."
                if rag_ready
                else "Chưa cấu hình LLM; hệ thống dùng câu trả lời trích xuất có nguồn."
            ),
        ),
        AiCapabilityStatus(
            capability=AiCapability.DEMAND_FORECAST,
            algorithm="Chronos-Bolt tiny (pretrained time-series transformer, ONNX), zero-shot",
            storage="MongoDB venue demand forecasts + model versions",
            status="READY" if demand_forecast_service.forecaster else "DEGRADED",
            detail=(
                "Dự báo tỉ lệ lấp đầy theo giờ 7 ngày tới, khoảng 10–90%, chấm điểm trên tuần vừa qua."
                if demand_forecast_service.forecaster
                else "Chưa có file mô hình Chronos-Bolt (chạy scripts/export_chronos_bolt.py)."
            ),
        ),
        AiCapabilityStatus(
            capability=AiCapability.IDENTITY_VERIFICATION,
            algorithm="OpenCV + PaddleOCR + YuNet/SFace + facial attributes + passive liveness",
            storage="Short-lived Redis verification session",
            status="READY" if identity_ready else "DEGRADED",
            detail=(
                "OCR CCCD, kiểm tra chất lượng, liveness và đối sánh khuôn mặt đang sẵn sàng."
                if identity_ready
                else "Thiếu thư viện vision hoặc model YuNet, SFace, facial attributes, passive liveness."
            ),
        ),
    ]
