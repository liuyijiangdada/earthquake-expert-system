#!/usr/bin/env python3
"""调度器与知识信号策略单测。"""

import unittest

from core.knowledge_signals import KnowledgeSignals, _calc_static_confidence
from core.phase_classifier import Phase, PhaseResult
from core.scheduler import Scheduler


class TestStaticConfidence(unittest.TestCase):
    def test_full_hits_near_one(self):
        s = KnowledgeSignals(
            kg_region_hit=True,
            kg_emergency_hit=True,
            kg_magnitude_hit=True,
            rag_top_score=0.85,
        )
        self.assertGreaterEqual(_calc_static_confidence(s), 0.9)


class TestSchedulerSignals(unittest.TestCase):
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

    def test_high_static_suppresses_dynamic(self):
        phase = PhaseResult(phase=Phase.PRE, confidence=0.8, urgency=0.2)
        signals = KnowledgeSignals(static_confidence=0.95, dynamic_availability=0.8)
        d = self.scheduler.decide(phase, signals)
        self.assertFalse(d.use_dynamic)

    def test_during_high_static_keeps_dynamic(self):
        phase = PhaseResult(
            phase=Phase.DURING, confidence=0.9, urgency=0.4, need_dynamic=True
        )
        signals = KnowledgeSignals(static_confidence=0.95, dynamic_availability=0.9)
        d = self.scheduler.decide(phase, signals)
        self.assertTrue(d.use_dynamic)

    def test_critical_urgency_enables_dynamic(self):
        phase = PhaseResult(phase=Phase.DURING, confidence=0.9, urgency=0.85, need_dynamic=True)
        signals = KnowledgeSignals(static_confidence=0.3, dynamic_availability=0.9)
        d = self.scheduler.decide(phase, signals)
        self.assertTrue(d.use_dynamic)

    def test_unavailable_dynamic_turned_off(self):
        phase = PhaseResult(phase=Phase.DURING, confidence=0.9, urgency=0.9, need_dynamic=True)
        signals = KnowledgeSignals(static_confidence=0.2, dynamic_availability=0.1)
        d = self.scheduler.decide(phase, signals)
        self.assertFalse(d.use_dynamic)
        self.assertIn("实时数据暂不可用", d.validity_hint)

    def test_medium_static_confidence_hint(self):
        phase = PhaseResult(phase=Phase.DURING, confidence=0.8, urgency=0.3)
        signals = KnowledgeSignals(static_confidence=0.54, dynamic_availability=1.0)
        d = self.scheduler.decide(phase, signals)
        self.assertIn("本地知识库", d.reliability_hint)
