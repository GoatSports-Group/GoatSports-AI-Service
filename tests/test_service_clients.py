import unittest
from unittest.mock import MagicMock, patch
from uuid import uuid4

import httpx

from app.infrastructure.config.settings import Settings
from app.infrastructure.external.client.service_clients import (
    HttpVenueServiceClient,
    _gateway_headers,
    _response_data,
)


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

    def test_matchmaking_booking_unwraps_spring_envelope(self):
        session_id = uuid4()
        booking_id = uuid4()
        player_id = uuid4()
        court_id = uuid4()
        response = httpx.Response(
            200,
            request=httpx.Request("GET", "http://venue/internal/matchmaking/bookings"),
            json={
                "statusCode": 200,
                "data": {
                    "bookingId": str(booking_id),
                    "matchmakingSessionId": str(session_id),
                    "playerId": str(player_id),
                    "venueCourtId": str(court_id),
                    "status": "CONFIRMED",
                },
            },
        )
        http_client = MagicMock()
        http_client.__enter__.return_value = http_client
        http_client.get.return_value = response
        settings = Settings(VENUE_SERVICE_URL="http://venue", GATEWAY_SECRET="service-secret")

        with patch(
            "app.infrastructure.external.client.service_clients.httpx.Client",
            return_value=http_client,
        ):
            snapshot = HttpVenueServiceClient(settings).get_matchmaking_booking(session_id)

        self.assertIsNotNone(snapshot)
        self.assertEqual(booking_id, snapshot.booking_id)
        self.assertEqual("CONFIRMED", snapshot.status)
        http_client.get.assert_called_once_with(
            f"http://venue/internal/matchmaking/bookings/{session_id}",
            headers={"X-Gateway-Secret": "service-secret"},
        )


if __name__ == "__main__":
    unittest.main()
