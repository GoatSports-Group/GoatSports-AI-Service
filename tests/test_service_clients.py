import unittest
from uuid import uuid4

import httpx

from app.infrastructure.config.settings import Settings
from app.infrastructure.external.client.service_clients import _gateway_headers, _response_data


class ServiceClientContractTests(unittest.TestCase):
    def test_unwraps_spring_response_envelope(self):
        response = httpx.Response(
            200,
            json={"statusCode": 200, "data": {"items": [{"venueId": "venue-1"}]}},
        )

        self.assertEqual({"items": [{"venueId": "venue-1"}]}, _response_data(response))

    def test_preserves_plain_service_response(self):
        response = httpx.Response(200, json=[{"status": "AVAILABLE"}])

        self.assertEqual([{"status": "AVAILABLE"}], _response_data(response))

    def test_gateway_headers_include_service_secret_and_actor(self):
        actor_id = uuid4()
        settings = Settings(GATEWAY_SECRET="service-secret")

        self.assertEqual(
            {
                "X-Gateway-Secret": "service-secret",
                "X-User-Id": str(actor_id),
            },
            _gateway_headers(settings, actor_id),
        )


if __name__ == "__main__":
    unittest.main()
