import unittest
from datetime import date, time
from uuid import uuid4

import httpx

from app.domain.enums import SportType
from app.infrastructure.config.settings import Settings
from app.infrastructure.external.client.service_clients import HttpVenueServiceClient


class VenueSlotWithinFreeWindowTests(unittest.TestCase):
    """Khung gio la khoang ranh cua hai ben; tran dau la mot slot trong nam tron trong khung do."""

    def setUp(self):
        self.early, self.late = uuid4(), uuid4()
        slots = {
            self.early: [
                {"startTime": "17:00:00", "endTime": "18:00:00", "status": "AVAILABLE"},
                {"startTime": "18:00:00", "endTime": "19:00:00", "status": "BOOKED"},
                {"startTime": "19:30:00", "endTime": "20:30:00", "status": "AVAILABLE"},
            ],
            self.late: [
                {"startTime": "19:00:00", "endTime": "20:00:00", "status": "AVAILABLE"},
                {"startTime": "18:30:00", "endTime": "19:30:00", "status": "AVAILABLE"},
            ],
        }
        courts = [
            {"venueCourtId": str(self.early), "name": "Sân 1", "sportType": "BADMINTON", "active": True},
            {"venueCourtId": str(self.late), "name": "Sân 2", "sportType": "BADMINTON", "active": True},
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/courts"):
                return httpx.Response(200, json={"statusCode": 200, "data": courts})
            court_id = request.url.path.split("/")[-2]
            return httpx.Response(200, json={"statusCode": 200, "data": slots[next(k for k in slots if str(k) == court_id)]})

        self.http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://venue")
        self.client = HttpVenueServiceClient(Settings())

    def test_picks_the_earliest_free_slot_fully_inside_the_window(self):
        names, court, slot = self.client._courts(
            self.http, uuid4(), SportType.BADMINTON, date(2026, 10, 1), time(18, 0), time(20, 0)
        )
        self.assertEqual(["Sân 1", "Sân 2"], names)
        self.assertEqual(self.late, court)
        self.assertEqual((time(18, 30), time(19, 30)), slot)

    def test_no_slot_inside_the_window_means_no_court(self):
        _, court, slot = self.client._courts(
            self.http, uuid4(), SportType.BADMINTON, date(2026, 10, 1), time(20, 0), time(21, 0)
        )
        self.assertIsNone(court)
        self.assertIsNone(slot)


if __name__ == "__main__":
    unittest.main()
