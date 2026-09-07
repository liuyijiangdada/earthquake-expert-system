#!/usr/bin/env python3
"""三阶段分类器单测：词表、多轮历史与紧急度。"""

import unittest

from core.phase_classifier import Phase, PhaseClassifier
from core.knowledge_signals import KnowledgeSignals
from core.scheduler import Scheduler


class TestPhaseClassifier(unittest.TestCase):
    def setUp(self):
        self.clf = PhaseClassifier()

    def test_aftershock_is_post(self):
        result = self.clf.classify("余震还会来吗")
        self.assertEqual(result.phase, Phase.POST)

    def test_just_now_magnitude_is_during(self):
        result = self.clf.classify("刚才地震了几级")
        self.assertEqual(result.phase, Phase.DURING)
        self.assertTrue(result.need_dynamic)

    def test_epicenter_alone_is_during(self):
        result = self.clf.classify("震中在哪")
        self.assertEqual(result.phase, Phase.DURING)

    def test_history_keeps_followup_during(self):
        history = [{"role": "user", "content": "刚才地震了"}]
        result = self.clf.classify("震中在哪", history=history)
        self.assertEqual(result.phase, Phase.DURING)

    def test_history_lifts_ambiguous_followup(self):
        self.assertEqual(self.clf.classify("多大？").phase, Phase.GENERAL)
        result = self.clf.classify(
            "多大？", history=[{"role": "user", "content": "刚才地震了"}]
        )
        self.assertEqual(result.phase, Phase.DURING)

    def test_history_does_not_inflate_urgency(self):
        history = [{"role": "user", "content": "刚才地震了"}]
        result = self.clf.classify("震中在哪", history=history)
        current_only = self.clf.classify("震中在哪")
        self.assertEqual(result.urgency, current_only.urgency)

    def test_prepare_now_is_pre(self):
        result = self.clf.classify("现在家里怎么准备应急包")
        self.assertEqual(result.phase, Phase.PRE)


class TestSchedulerPhaseGuards(unittest.TestCase):
    def setUp(self):
        self.scheduler = Scheduler(
            type(
                "Cfg",
                (),
                {
                    "SCHEDULER_DYNAMIC_CONFIDENCE_THRESHOLD": 0.4,
                    "SCHEDULER_STATIC_CONFIDENCE_THRESHOLD": 0.9,
                    "SCHEDULER_URGENCY_HIGH_THRESHOLD": 0.5,
                    "SCHEDULER_URGENCY_CRITICAL_THRESHOLD": 0.7,
                },
            )()
        )
        self.clf = PhaseClassifier()

    def test_during_high_static_keeps_dynamic(self):
        phase = self.clf.classify("刚才地震了几级")
        signals = KnowledgeSignals(static_confidence=0.95, dynamic_availability=0.9)
        d = self.scheduler.decide(phase, signals)
        self.assertTrue(d.use_dynamic)

    def test_pre_does_not_enable_dynamic(self):
        phase = self.clf.classify("现在家里怎么准备应急包")
        self.assertEqual(phase.phase, Phase.PRE)
        d = self.scheduler.decide(phase)
        self.assertFalse(d.use_dynamic)

    def test_pre_high_urgency_still_no_dynamic(self):
        phase = self.clf.classify("现在马上紧急怎么准备应急包")
        self.assertEqual(phase.phase, Phase.PRE)
        self.assertGreater(phase.urgency, 0.7)
        self.assertTrue(phase.need_dynamic)
        d = self.scheduler.decide(phase)
        self.assertFalse(d.use_dynamic)


if __name__ == "__main__":
    unittest.main()
