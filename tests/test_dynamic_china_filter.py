#!/usr/bin/env python3
import unittest

from core.earthquake_feed import (
    EarthquakeFeedConfig,
    _in_china_bbox,
    _parse_ceic_payload,
    merge_feed_items,
)


class TestChinaBbox(unittest.TestCase):
    def setUp(self):
        self.cfg = EarthquakeFeedConfig(
            china_min_lat=18.0,
            china_max_lat=54.0,
            china_min_lon=73.0,
            china_max_lon=135.0,
        )

    def test_mainland_inside(self):
        self.assertTrue(_in_china_bbox(self.cfg, 30.0, 104.0))

    def test_outside_us(self):
        self.assertFalse(_in_china_bbox(self.cfg, 37.0, -122.0))


class TestCeicParse(unittest.TestCase):
    def test_parse_wrapped_json(self):
        raw = '({"shuju":[{"M":"4.8","O_TIME":"2026-06-07 10:00:00","EPI_LAT":"30.1","EPI_LON":"103.2","EPI_DEPTH":"10","LOCATION_C":"四川某县","NEW_DID":"CD202606070001"}],"jieguo":"最近24小时","num":1})'
        rows = _parse_ceic_payload(raw)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["M"], "4.8")


class TestMergeFeed(unittest.TestCase):
    def test_ceic_preferred_on_duplicate(self):
        ceic = [{
            "id": "ceic_1",
            "magnitude": 4.8,
            "latitude": 30.1,
            "longitude": 103.2,
            "time": "2026-06-07 10:00:00",
            "location": "四川",
            "source": "ceic",
        }]
        usgs = [{
            "id": "usgs_1",
            "magnitude": 4.8,
            "latitude": 30.11,
            "longitude": 103.21,
            "time": "2026-06-07 10:00:00",
            "location": "Sichuan",
            "source": "usgs",
        }]
        merged = merge_feed_items([("usgs", usgs), ("ceic", ceic)], max_items=5, prefer_ceic=True)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["source"], "ceic")


if __name__ == "__main__":
    unittest.main()
