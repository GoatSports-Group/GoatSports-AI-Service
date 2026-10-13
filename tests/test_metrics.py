import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.presentation.metrics import METRICS_PATH, register_metrics


class PrometheusMetricsTests(unittest.TestCase):
    def test_records_requests_by_route_template_with_micrometer_names(self):
        app = FastAPI()
        register_metrics(app)

        @app.get("/probe/{item_id}")
        def probe(item_id: str) -> dict[str, str]:
            return {"id": item_id}

        client = TestClient(app)
        client.get("/probe/1")
        client.get("/probe/2")
        client.get("/khong-ton-tai")

        body = client.get(METRICS_PATH).text

        self.assertIn('http_server_requests_seconds_count{method="GET",status="200",uri="/probe/{item_id}"} 2.0', body)
        self.assertIn('http_server_requests_seconds_count{method="GET",status="404",uri="UNKNOWN"} 1.0', body)
        self.assertIn("http_server_requests_seconds_bucket", body)
        self.assertIn("process_uptime_seconds", body)
        # CPU có trên mọi hệ điều hành (ProcessCollector trên Linux, PortableProcessCollector nơi khác).
        self.assertIn("process_cpu_seconds_total", body)

    def test_portable_collector_reports_cpu_and_windows_memory(self):
        import sys

        from app.presentation.metrics import PortableProcessCollector

        names = {family.name: family for family in PortableProcessCollector().collect()}

        self.assertGreater(names["process_cpu_seconds"].samples[0].value, 0)
        if sys.platform == "win32":
            self.assertGreater(names["process_resident_memory_bytes"].samples[0].value, 10_000_000)


if __name__ == "__main__":
    unittest.main()
