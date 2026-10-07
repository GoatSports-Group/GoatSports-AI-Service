"""Metrics Prometheus cùng tên với Micrometer của các service Spring (http_server_requests_seconds,
process_uptime_seconds) để dashboard và alert rule trên Grafana dùng chung cho ai-service.

Đường dẫn giống Spring (/actuator/prometheus): Prometheus scrape trực tiếp trong goatsports-network,
còn gateway đã chặn /*/actuator/prometheus nên không lộ ra ngoài.
"""

import time

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, Histogram, generate_latest

METRICS_PATH = "/actuator/prometheus"

_REQUESTS = Histogram(
    "http_server_requests_seconds",
    "Thời gian xử lý HTTP request",
    ("method", "uri", "status"),
)
_STARTED_AT = time.time()
Gauge("process_uptime_seconds", "Thời gian chạy của process").set_function(lambda: time.time() - _STARTED_AT)


def register_metrics(app: FastAPI) -> None:
    @app.middleware("http")
    async def record_request(request: Request, call_next):
        started = time.perf_counter()
        status = 500  # exception không được handler nào bắt -> tính là 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            # uri là mẫu route (/api/v1/matches/{id}) để không bùng số series; không khớp route -> UNKNOWN như Micrometer.
            uri = getattr(request.scope.get("route"), "path", "UNKNOWN")
            _REQUESTS.labels(request.method, uri, str(status)).observe(time.perf_counter() - started)

    @app.get(METRICS_PATH, include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
