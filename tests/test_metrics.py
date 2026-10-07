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


if __name__ == "__main__":
    unittest.main()
