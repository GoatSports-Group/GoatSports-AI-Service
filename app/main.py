from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.infrastructure.config.database import database_clients
from app.infrastructure.config.settings import settings
from app.presentation.controller import (
    capabilities,
    chatbot,
    demand_forecast,
    matchmaking,
    personalization,
    recommendations,
    semantic_search,
)
from app.presentation.exception import register_exception_handlers


@asynccontextmanager
async def lifespan(_: FastAPI):
    database_clients.connect()
    yield
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
    capabilities.router,
):
    app.include_router(router, prefix=settings.API_V1_STR)

register_exception_handlers(app)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    database_clients.mongo_client.admin.command("ping")
    database_clients.redis.ping()
    return {"status": "UP", "service": settings.PROJECT_NAME}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)