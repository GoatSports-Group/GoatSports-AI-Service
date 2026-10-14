import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.infrastructure.config.container import (
    matchmaking_model_service,
    rag_chatbot_service,
    shutdown_identity_verification,
    warm_up_identity_verification,
)
from app.infrastructure.config.database import database_clients
from app.infrastructure.config.settings import settings
from app.presentation.controller import (
    admin_status,
    capabilities,
    chatbot,
    content_moderation,
    demand_forecast,
    identity_verification,
    matchmaking,
    personalization,
    recommendations,
    semantic_search,
)
from app.presentation.exception import register_exception_handlers
from app.presentation.metrics import register_metrics

logger = logging.getLogger(__name__)


async def _train_matchmaking_model_periodically(interval: int) -> None:
    """Học lại trọng số xếp hạng từ kết quả kèo thật; chưa đủ dữ liệu thì giữ trọng số mặc định."""
    await asyncio.sleep(60)
    while True:
        try:
            await asyncio.to_thread(matchmaking_model_service.train)
        except Exception:  # noqa: BLE001 - job nền không được làm sập service
            logger.warning("Matchmaking model training failed", exc_info=True)
        await asyncio.sleep(interval)


def _sync_knowledge_base() -> None:
    """Nạp các file Markdown trong thư mục knowledge/ vào kho tri thức chatbot; lỗi thì chatbot vẫn chạy với kho cũ."""
    directory = Path(settings.RAG_KNOWLEDGE_DIR)
    if not directory.is_dir():
        logger.warning("Knowledge directory %s not found; chatbot keeps its current knowledge base", directory)
        return
    try:
        report = rag_chatbot_service.sync_directory(directory)
        logger.info(
            "Knowledge base synced: %d indexed, %d unchanged, %d removed",
            len(report.indexed), len(report.unchanged), len(report.removed),
        )
    except Exception:  # noqa: BLE001 - kho tri thức hỏng không được làm sập service
        logger.exception("Knowledge base sync failed")


@asynccontextmanager
async def lifespan(_: FastAPI):
    database_clients.connect()
    trainer = None
    try:
        await asyncio.to_thread(warm_up_identity_verification)
        logger.info("Identity verification models are ready")
        await asyncio.to_thread(_sync_knowledge_base)
        if settings.MATCHMAKING_TRAIN_INTERVAL_SECONDS > 0:
            trainer = asyncio.create_task(
                _train_matchmaking_model_periodically(settings.MATCHMAKING_TRAIN_INTERVAL_SECONDS)
            )
        yield
    finally:
        if trainer:
            trainer.cancel()
        await asyncio.to_thread(shutdown_identity_verification)
        database_clients.close()


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    lifespan=lifespan,
)

for router in (
    matchmaking.router,
    recommendations.router,
    chatbot.router,
    personalization.router,
    semantic_search.router,
    demand_forecast.router,
    identity_verification.router,
    capabilities.router,
    admin_status.router,
    content_moderation.router,
):
    app.include_router(router, prefix=settings.API_V1_STR)

register_exception_handlers(app)
register_metrics(app)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    database_clients.mongo_client.admin.command("ping")
    database_clients.redis.ping()
    return {"status": "UP", "service": settings.PROJECT_NAME}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)