#!/usr/bin/env python3
"""回答质量检测与 RAG 降级单测。"""

import unittest

from services.response_guard import (
    build_rag_fallback_text,
    guard_response,
    is_garbled_response,
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


if __name__ == "__main__":
    unittest.main()
