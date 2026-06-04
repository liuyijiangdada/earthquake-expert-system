#!/usr/bin/env python3
import unittest
from unittest.mock import MagicMock, patch

from core.amap_client import AmapClient
from core.coord_convert import wgs84_to_gcj02


class TestCoordConvert(unittest.TestCase):
    def test_wgs84_to_gcj02_offset(self):
        lon, lat = 104.0668, 30.5728
        glon, glat = wgs84_to_gcj02(lon, lat)
        self.assertNotAlmostEqual(lon, glon, places=3)
        self.assertNotAlmostEqual(lat, glat, places=3)


class TestAmapClient(unittest.TestCase):
    def test_unavailable_without_key(self):
        c = AmapClient("")
        self.assertFalse(c.available)

    @patch("core.amap_client.requests.get")
    def test_geocode(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "status": "1",
            "geocodes": [{"location": "104.066800,30.572800"}],
        }
        mock_get.return_value = mock_resp

        c = AmapClient("fake-key")
        geo = c.geocode("成都市", city="成都")
        self.assertIsNotNone(geo)
        self.assertAlmostEqual(geo[0], 104.0668, places=3)

    def test_public_static_map_path(self):
        c = AmapClient("fake-key")
        path = c.public_static_map_path(104.06, 30.57, zoom=12)
        self.assertTrue(path.startswith("/api/amap/static-map?"))
        self.assertIn("lon=", path)
