#!/usr/bin/env python3
import unittest

from core.geo_utils import haversine_km, match_cities_in_text


class TestGeoUtils(unittest.TestCase):
    def test_haversine(self):
        d = haversine_km(30.57, 104.07, 30.67, 104.12)
        self.assertGreater(d, 0)
        self.assertLess(d, 20)

    def test_match_cities(self):
        aliases = {"成都": ["成都"]}
        self.assertEqual(match_cities_in_text("成都避难所", aliases), ["成都"])
