#!/usr/bin/env python3
import os
import unittest

from config.config import Config
from core.shelter_service import ShelterService


class TestShelterService(unittest.TestCase):
    def setUp(self):
        self.svc = ShelterService(Config())

    def test_find_by_city(self):
        if not self.svc.enabled:
            self.skipTest("shelters.json not loaded")
        hits = self.svc.find_nearest("我在成都，最近的避难所在哪？")
        self.assertTrue(hits)
        self.assertEqual(hits[0]["city"], "成都")
        self.assertIn("navigation_url", hits[0])

    def test_context_text(self):
        if not self.svc.enabled:
            self.skipTest("shelters.json not loaded")
        hits = self.svc.find_nearest("宜宾避难所")
        text = self.svc.to_context_text(hits)
        self.assertIn("避难所", text)


if __name__ == "__main__":
    unittest.main()
