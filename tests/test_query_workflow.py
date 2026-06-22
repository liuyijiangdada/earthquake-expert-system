#!/usr/bin/env python3
"""LangGraph 问答编排单测（不加载大模型）。"""

import unittest
from unittest.mock import MagicMock

from core.phase_classifier import PhaseClassifier
from core.scheduler import Scheduler
from services.context_builder import QueryContextBuilder, QueryContextDeps
from workflows.query_graph import QueryWorkflowDeps, build_query_workflow, run_context_workflow


class _Cfg:
    KG_CONTEXT_ENABLED = False
    RAG_ENABLED = False
    PHASE_CLASSIFIER_ENABLED = True
    SCHEDULER_ENABLED = True
    VALIDITY_HINT_ENABLED = True
    MULTIMODAL_INJECT_PROMPT = False
    SCHEDULER_DYNAMIC_CONFIDENCE_THRESHOLD = 0.4
    SCHEDULER_STATIC_CONFIDENCE_THRESHOLD = 0.9
    SCHEDULER_URGENCY_HIGH_THRESHOLD = 0.5
    SCHEDULER_URGENCY_CRITICAL_THRESHOLD = 0.7


class TestQueryWorkflow(unittest.TestCase):
    def setUp(self):
        cfg = _Cfg()
        builder = QueryContextBuilder(
            QueryContextDeps(
                config=cfg,
                kg=MagicMock(),
                emergency_rag=None,
                phase_classifier=PhaseClassifier(),
                scheduler=Scheduler(cfg),
                dynamic_retriever=None,
                multimodal_output=None,
            )
        )
        self.deps = QueryWorkflowDeps(
            context_builder=builder,
            apply_layer3=lambda prompt, _q, _p, meta: prompt,
            run_llm=lambda _prompt: "室内应就地避险，保护头颈。",
            sanitize=lambda text, _q: text,
            guard=lambda text, _meta: text,
            finalize_media=lambda meta, _q, _p: meta.setdefault("media_resources", []),
        )

    def test_context_workflow_returns_prompt_and_phase(self):
        prompt, meta, phase = run_context_workflow(
            self.deps,
            "地震发生时室内怎么办",
        )
        self.assertIn("【问题】", prompt)
        self.assertIn("地震发生时室内怎么办", prompt)
        self.assertIn("phase", meta)
        self.assertIn(phase, ("震前", "震中", "震后", "通用"))

    def test_full_workflow_invokes_llm_and_enrich(self):
        workflow = build_query_workflow(self.deps, include_generate=True)
        result = workflow.invoke(
            {
                "input_text": "家庭应急包应准备什么",
                "history": None,
                "for_vision": False,
                "skip_generate": False,
            }
        )
        self.assertIn("室内应就地避险", result["response"])
        self.assertIn("media_resources", result["debug_meta"])

    def test_graph_node_order(self):
        workflow = build_query_workflow(self.deps, include_generate=False)
        result = workflow.invoke(
            {
                "input_text": "震后如何申请救助",
                "history": None,
                "for_vision": False,
                "skip_generate": True,
            }
        )
        self.assertTrue(result["prompt"])
        self.assertEqual(result["debug_meta"]["phase"], "震后")


if __name__ == "__main__":
    unittest.main()
