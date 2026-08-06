#!/usr/bin/env python3
"""回答质量检测与 RAG 降级单测。"""

import unittest

from services.response_guard import (
    build_confident_dynamic_answer,
    build_rag_fallback_text,
    guard_response,
    has_degenerate_punctuation,
    has_hedging_language,
    is_garbled_response,
    is_quake_fact_query,
    sanitize_repeated_punctuation,
    strip_eval_prefix,
)


class TestResponseGuard(unittest.TestCase):
    def test_strip_eval_prefix(self):
        self.assertEqual(strip_eval_prefix("参考答案：\n1. 趴下"), "1. 趴下")

    def test_detect_garbled(self):
        bad = "参考答案：\n1号蹲牙，严责历史 中以下搜/绵阳市丹心图强"
        self.assertTrue(is_garbled_response(bad))
        good = "1. 保持冷静，迅速判断所处环境：优先选择「伏地、遮挡、手抓牢」。"
        self.assertFalse(is_garbled_response(good))

    def test_rag_fallback_format(self):
        hits = [{
            "title": "振动发生时室内如何避险",
            "text": "振动发生时室内如何避险\n应急避险\n保持冷静，迅速判断所处环境。",
        }]
        out = build_rag_fallback_text(hits)
        self.assertIn("根据本地应急知识库", out)
        self.assertIn("1.", out)

    def test_guard_uses_rag_on_garbled(self):
        meta = {
            "static_confidence": 0.54,
            "rag_fallback_text": "根据本地应急知识库「振动发生时室内如何避险」：\n1. 保持冷静。",
        }
        bad = "参考答案：\n1号蹲牙，严责历史 中以下搜/绵阳市丹心图强"
        out = guard_response(bad, meta)
        self.assertIn("根据本地应急知识库", out)
        self.assertEqual(meta.get("response_fallback"), "rag")

    def test_guard_keeps_good_answer(self):
        meta = {"static_confidence": 0.95, "rag_fallback_text": "fallback"}
        good = "室内避险请遵循「伏地、遮挡、手抓牢」：趴下护头，远离窗户。"
        out = guard_response(good, meta)
        self.assertEqual(out, good)

    def test_detect_exclaim_degeneration(self):
        bad = "总之，地震是一种常见的自然灾害，我们应该" + ("!" * 80)
        self.assertTrue(has_degenerate_punctuation(bad))
        self.assertTrue(is_garbled_response(bad))
        self.assertNotIn("!!!", sanitize_repeated_punctuation(bad))

    def test_guard_fallback_on_exclaim_run(self):
        meta = {
            "static_confidence": 0.54,
            "rag_fallback_text": "根据本地应急知识库「地震基础知识」：\n1. 震级每增加1级，能量约增加32倍。",
        }
        bad = "我们应该" + ("!" * 100)
        out = guard_response(bad, meta)
        self.assertIn("根据本地应急知识库", out)
        self.assertEqual(meta.get("response_fallback"), "rag")

    def test_quake_fact_query_and_confident_answer(self):
        self.assertTrue(is_quake_fact_query("刚才地震多大，震中在哪？"))
        self.assertTrue(has_hedging_language("目前无法准确确定此次地震的震级"))
        items = [{
            "magnitude": 5.2,
            "location": "日本本州南部附近",
            "time": "2026-08-02 15:20:00",
            "depth": 10,
            "source": "usgs",
        }]
        ans = build_confident_dynamic_answer(items, fetched_at="2026-08-02 15:25")
        self.assertIn("5.2级", ans)
        self.assertIn("日本本州南部附近", ans)
        self.assertNotIn("无法", ans)

    def test_guard_rewrites_hedging_with_dynamic(self):
        meta = {
            "user_query": "刚才地震多大，震中在哪？",
            "dynamic_items": [{
                "magnitude": 5.2,
                "location": "日本本州南部附近",
                "time": "2026-08-02 15:20:00",
                "depth": 10,
                "source": "usgs",
            }],
            "dynamic_fetched_at": "2026-08-02 15:25",
        }
        hedge = (
            "目前无法准确确定此次地震的震级及震中位置，因为该内容尚未得到权威机构的确认。"
            "但是根据实时数据显示，本次地震发生在日本本州岛的南部地区，"
            "具体震级需要参考后续发布的正式报告结果。"
        )
        out = guard_response(hedge, meta)
        self.assertIn("5.2级", out)
        self.assertIn("日本本州南部附近", out)
        self.assertEqual(meta.get("response_fallback"), "dynamic_confident")


if __name__ == "__main__":
    unittest.main()
