"""Metrics Prometheus cùng tên với Micrometer của các service Spring (http_server_requests_seconds,
process_uptime_seconds) để dashboard và alert rule trên Grafana dùng chung cho ai-service.

Đường dẫn giống Spring (/actuator/prometheus): Prometheus scrape trực tiếp trong goatsports-network,
còn gateway đã chặn /*/actuator/prometheus nên không lộ ra ngoài.
"""

import os
import sys
import time

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, Gauge, Histogram, generate_latest
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily

METRICS_PATH = "/actuator/prometheus"

_REQUESTS = Histogram(
    "http_server_requests_seconds",
    "Thời gian xử lý HTTP request",
    ("method", "uri", "status"),
)
_STARTED_AT = time.time()
Gauge("process_uptime_seconds", "Thời gian chạy của process").set_function(lambda: time.time() - _STARTED_AT)


def _resident_memory_bytes() -> float | None:
    """RAM đang chiếm (working set) trên Windows, qua GetProcessMemoryInfo; hệ khác không đo được thì None."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class _Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

    # Khai báo kiểu rõ ràng: mặc định ctypes truyền handle -1 (process hiện tại) thành int 32-bit, sai trên Windows 64-bit.
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = wintypes.HANDLE
    get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory_info.argtypes = (wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD)
    get_memory_info.restype = wintypes.BOOL

    counters = _Counters()
    counters.cb = ctypes.sizeof(counters)
    if not get_memory_info(get_current_process(), ctypes.byref(counters), counters.cb):
        return None
    return float(counters.WorkingSetSize)


class PortableProcessCollector:
    """CPU / RAM cho máy không có /proc (chạy ai-service trực tiếp trên Windows lúc dev).

    ProcessCollector mặc định của prometheus_client chỉ đọc /proc (Linux) nên trên Windows không xuất gì, và trang
    Giám sát hiện "—" ở cột CPU / Bộ nhớ của ai-service. Tên metric giống hệt bản Linux để truy vấn dùng chung.
    """

    def collect(self):
        times = os.times()
        cpu = CounterMetricFamily("process_cpu_seconds", "Tổng thời gian CPU (user + system) của process")
        cpu.add_metric([], times.user + times.system)
        yield cpu
        rss = _resident_memory_bytes()
        if rss is not None:
            yield GaugeMetricFamily("process_resident_memory_bytes", "RAM process đang chiếm", value=rss)


# Linux (Docker) đã có ProcessCollector mặc định; đăng ký thêm sẽ trùng tên metric.
if not os.path.exists("/proc/self/stat"):
    REGISTRY.register(PortableProcessCollector())


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
