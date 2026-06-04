#!/usr/bin/env python3
"""
已弃用：原 BLIP 两阶段视觉方案。
请使用 llm.qwen_vl_handler（Qwen2-VL 端到端多模态）。
"""

from llm.qwen_vl_handler import QwenVLHandler, qwen_vl_handler

__all__ = ["QwenVLHandler", "qwen_vl_handler", "VisionHandler"]


class VisionHandler(QwenVLHandler):
    """兼容旧导入名，行为与 QwenVLHandler 相同。"""

    def describe_image_for_earthquake(self, image, user_question: str) -> str:
        raise NotImplementedError(
            "BLIP 描述接口已移除，请使用 QwenVLHandler.generate(image, prompt)"
        )
