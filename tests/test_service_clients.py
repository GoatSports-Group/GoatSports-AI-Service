import unittest
from unittest.mock import MagicMock, patch
from uuid import uuid4

import httpx

from app.domain.enums import SportType
from app.infrastructure.config.settings import Settings
from app.infrastructure.external.client.service_clients import (
    HttpNotificationServiceClient,
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

    def test_external_venues_are_the_nearest_inside_the_radius(self):
        near, far_corner = uuid4(), uuid4()
        response = httpx.Response(
            200,
            request=httpx.Request("GET", "http://venue/api/v1/venues/external"),
            json=[
                # Góc khung bản đồ: vẫn nằm trong bbox nhưng xa hơn bán kính 5 km.
                {"externalVenueId": str(far_corner), "name": "Góc khung", "latitude": 10.84, "longitude": 106.74},
                {"externalVenueId": str(near), "name": "Sân gần", "latitude": 10.781, "longitude": 106.701,
                 "phone": "0909 000 000"},
            ],
        )
        http_client = MagicMock()
        http_client.__enter__.return_value = http_client
        http_client.get.return_value = response

        with patch(
            "app.infrastructure.external.client.service_clients.httpx.Client",
            return_value=http_client,
        ):
            options = HttpVenueServiceClient(Settings(VENUE_SERVICE_URL="http://venue")).nearby_external_venues(
                10.78, 106.70, SportType.BADMINTON, 5
            )

        self.assertEqual([near], [option.venue_id for option in options])
        self.assertTrue(options[0].external)
        self.assertEqual("0909 000 000", options[0].phone)
        params = http_client.get.call_args.kwargs["params"]
        self.assertEqual("BADMINTON", params["sportType"])
        self.assertAlmostEqual(10.78 + 5 / 111, params["north"])

    def test_matchmaking_notifications_carry_the_gateway_secret(self):
        # notification-service trả 403 cho mọi request thiếu secret; thiếu là ghép trận mất realtime.
        http_client = MagicMock()
        http_client.__enter__.return_value = http_client
        http_client.post.return_value = httpx.Response(
            201, request=httpx.Request("POST", "http://notification/internal/notifications")
        )
        settings = Settings(NOTIFICATION_SERVICE_URL="http://notification", GATEWAY_SECRET="service-secret")

        with patch(
            "app.infrastructure.external.client.service_clients.httpx.Client",
            return_value=http_client,
        ):
            HttpNotificationServiceClient(settings).notify_matchmaking(uuid4(), uuid4(), "Tiêu đề", "Nội dung")

        self.assertEqual(
            {"X-Gateway-Secret": "service-secret"}, http_client.post.call_args.kwargs["headers"]
        )


if __name__ == "__main__":
    unittest.main()
