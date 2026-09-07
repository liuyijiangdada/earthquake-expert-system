#!/usr/bin/env python3
"""评测评分、快照、基线组装与人工评测汇总单测。"""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

from core.dynamic_retriever import DynamicRetriever
from core.dynamic_snapshot import apply_snapshot_to_retriever, load_snapshot
from core.eval_runtime import (
    JsonEvalKG,
    apply_baseline_to_stack,
    build_eval_stack,
    spec_by_id,
)
from core.eval_scoring import score_response, score_response_normalized

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))
from summarize_human_eval import cohen_kappa, summarize  # noqa: E402


class TestEvalScoring(unittest.TestCase):
    def test_b0_legacy_cap_vs_normalized_zero(self):
        prompt = "【知识图谱】\n（本路径已关闭）\n\n【参考资料】\n（本路径已关闭）\n\n【问题】\n家庭应急包该准备什么？"
        response = "家庭应急包应准备水、食物、急救药品、手电筒、收音机、重要证件复印件等物品。" * 3
        legacy = score_response(response, prompt, {}, "震前", kg_on=False, rag_on=False)
        norm = score_response_normalized(response, prompt, {}, "震前")
        self.assertLessEqual(legacy.factual, 0.42)
        self.assertGreater(legacy.factual, 0.0)
        self.assertEqual(norm.factual, 0.0)

    def test_normalized_equal_weight_rag_not_kg_bonus(self):
        rag = "室内避险应伏地遮挡手抓牢，远离玻璃门窗，不要乘坐电梯。"
        prompt = (
            f"【知识图谱】\n（无）\n\n【参考资料】\n{rag}\n\n【问题】\n室内应该躲哪里？"
        )
        response = rag + "主震后有序撤离到开阔地。"
        legacy = score_response(response, prompt, {}, "震中", kg_on=False, rag_on=True)
        both = score_response(
            response,
            prompt.replace("（无）", "汶川2008年8.0级地震发生在四川"),
            {},
            "震中",
            kg_on=True,
            rag_on=True,
        )
        norm_rag = score_response_normalized(response, prompt, {}, "震中")
        self.assertGreater(norm_rag.factual, 0.3)
        self.assertGreater(both.factual, legacy.factual)


class TestDynamicSnapshot(unittest.TestCase):
    def test_load_and_apply(self):
        result = load_snapshot()
        self.assertGreaterEqual(len(result.items), 1)
        retriever = DynamicRetriever()
        apply_snapshot_to_retriever(retriever, Path("data/eval/dynamic_feed_snapshot.json"))
        got = retriever.fetch_recent_earthquakes(force=True)
        self.assertEqual(got.items[0]["location"], result.items[0]["location"])
        retriever._enabled = False
        off = retriever.fetch_recent_earthquakes()
        self.assertIn("关闭", off.error or "")


class TestBaselineAssemble(unittest.TestCase):
    def setUp(self):
        from config.config import Config

        Config.RAG_USE_MEMORY_RAG = True
        Config.RAG_ENABLED = True
        self.stack = build_eval_stack(
            Config,
            snapshot_path=Path("data/eval/dynamic_feed_snapshot.json"),
            force_json_kg=True,
            skip_rag=True,
        )

    def test_b3_ns_has_no_phase_banner(self):
        apply_baseline_to_stack(self.stack, spec_by_id("B3-ns"))
        prompt, meta, tag = self.stack.builder.prepare("室内应该躲哪里？")
        self.assertEqual(tag, "")
        self.assertNotIn("当前判定为【", prompt)
        self.assertFalse(meta.get("schedule_reasoning"))

    def test_b4_during_injects_snapshot(self):
        apply_baseline_to_stack(self.stack, spec_by_id("B4"))
        prompt, meta, _ = self.stack.builder.prepare("刚才地震多大？震中在哪？")
        self.assertIn("【动态信息", prompt)
        self.assertGreater(int(meta.get("dynamic_items_count") or 0), 0)

    def test_b4_pre_no_dynamic(self):
        apply_baseline_to_stack(self.stack, spec_by_id("B4"))
        prompt, meta, _ = self.stack.builder.prepare("家庭应急包该准备什么？")
        self.assertNotIn("泸定", prompt)
        self.assertFalse(meta.get("dynamic_source"))

    def test_json_kg_region(self):
        kg = JsonEvalKG()
        hits = kg.query_earthquakes_by_region("四川")
        self.assertTrue(hits)

    def test_keyword_rag_hits_indoor(self):
        from core.eval_runtime import KeywordRAG
        from config.config import Config

        rag = KeywordRAG.from_config(Config)
        self.assertIsNotNone(rag)
        hits = rag.search("室内应该躲哪里？", top_k=3)
        self.assertTrue(hits)
        self.assertGreater(hits[0]["score"], 0)


class TestHumanEvalKappa(unittest.TestCase):
    def test_perfect_kappa(self):
        self.assertEqual(cohen_kappa([0, 1, 2, 1], [0, 1, 2, 1], (0, 1, 2)), 1.0)

    def test_summarize_empty_raters(self):
        with tempfile.TemporaryDirectory() as td:
            sheet = Path(td) / "s.csv"
            with sheet.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "id",
                        "phase",
                        "factual_r1",
                        "factual_r2",
                        "completeness_r1",
                        "completeness_r2",
                        "safety_r1",
                        "safety_r2",
                    ],
                )
                w.writeheader()
                w.writerow(
                    {
                        "id": "H01",
                        "phase": "震中",
                        "factual_r1": "2",
                        "factual_r2": "2",
                        "completeness_r1": "4",
                        "completeness_r2": "3",
                        "safety_r1": "1",
                        "safety_r2": "1",
                    }
                )
            out = Path(td) / "sum.json"
            rec = summarize(sheet, out)
            self.assertEqual(rec["agreement"]["kappa_factual"], 1.0)
            self.assertEqual(rec["agreement"]["kappa_safety"], 1.0)


if __name__ == "__main__":
    unittest.main()
