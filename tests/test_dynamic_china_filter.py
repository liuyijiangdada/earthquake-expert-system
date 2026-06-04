#!/usr/bin/env python3
import unittest

from core.dynamic_retriever import DynamicRetriever


class TestChinaBbox(unittest.TestCase):
    def setUp(self):
        self.dr = DynamicRetriever(
            type(
                "Cfg",
                (),
                {
                    "DYNAMIC_CHINA_FILTER_ENABLED": True,
                    "DYNAMIC_CHINA_MIN_LAT": 18.0,
                    "DYNAMIC_CHINA_MAX_LAT": 54.0,
                    "DYNAMIC_CHINA_MIN_LON": 73.0,
                    "DYNAMIC_CHINA_MAX_LON": 135.0,
                },
            )()
        )

    def test_mainland_inside(self):
        self.assertTrue(self.dr._in_china_bbox(30.0, 104.0))

    def test_outside_us(self):
        self.assertFalse(self.dr._in_china_bbox(37.0, -122.0))
