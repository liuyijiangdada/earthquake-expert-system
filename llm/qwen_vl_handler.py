#!/usr/bin/env python3
"""
端到端多模态推理：Qwen2-VL 直接理解图片并生成回答（替代 BLIP 转文字 + 文本 LLM）。
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from PIL import Image

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = logging.getLogger(__name__)

_VL_IMPORTS: Optional[dict] = None
_VL_IMPORT_ERROR: Optional[BaseException] = None


def _ensure_vl_imports() -> dict:
    global _VL_IMPORTS, _VL_IMPORT_ERROR
    if _VL_IMPORTS is not None:
        return _VL_IMPORTS
    if _VL_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Qwen2-VL 依赖未就绪，请执行: pip install qwen-vl-utils torchvision"
            f"（详情: {_VL_IMPORT_ERROR}）"
        ) from _VL_IMPORT_ERROR
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info

        _VL_IMPORTS = {
            "Qwen2VLForConditionalGeneration": Qwen2VLForConditionalGeneration,
            "AutoProcessor": AutoProcessor,
            "process_vision_info": process_vision_info,
        }
        return _VL_IMPORTS
    except ImportError as e:
        _VL_IMPORT_ERROR = e
        raise RuntimeError(
            "Qwen2-VL 依赖未就绪，请执行: pip install qwen-vl-utils torchvision"
            f"（详情: {e}）"
        ) from e


class QwenVLHandler:
    """懒加载 Qwen2-VL，仅在首次图片问答时占用显存。"""

    def __init__(self, config=None):
        self._config = config
        self._model = None
        self._processor = None
        self._loaded = False
        self._load_error: Optional[str] = None

        if config:
            self.model_name = getattr(
                config, "QWEN_VL_MODEL_NAME", "Qwen/Qwen2-VL-2B-Instruct"
            )
            self.max_new_tokens = int(getattr(config, "QWEN_VL_MAX_NEW_TOKENS", 384))
            self.temperature = float(getattr(config, "QWEN_VL_TEMPERATURE", 0.5))
            self.top_p = float(getattr(config, "QWEN_VL_TOP_P", 0.9))
            self.do_sample = bool(getattr(config, "QWEN_VL_DO_SAMPLE", True))
            self.min_pixels = int(getattr(config, "QWEN_VL_MIN_PIXELS", 256 * 28 * 28))
            self.max_pixels = int(getattr(config, "QWEN_VL_MAX_PIXELS", 512 * 28 * 28))
            self.max_image_edge = int(getattr(config, "QWEN_VL_MAX_IMAGE_EDGE", 768))
            self.max_context_chars = int(
                getattr(config, "QWEN_VL_MAX_CONTEXT_CHARS", 6000)
            )
        else:
            self.model_name = "Qwen/Qwen2-VL-2B-Instruct"
            self.max_new_tokens = 384
            self.temperature = 0.5
            self.top_p = 0.9
            self.do_sample = True
            self.min_pixels = 256 * 28 * 28
            self.max_pixels = 512 * 28 * 28
            self.max_image_edge = 768
            self.max_context_chars = 6000

        self.device = self._get_device()
        self.dtype = (
            torch.float16 if self.device.type in {"cuda", "mps"} else torch.float32
        )

    @staticmethod
    def _resize_image(image: Image.Image, max_edge: int) -> Image.Image:
        """缩小过长边，降低视觉注意力显存占用。"""
        w, h = image.size
        if max(w, h) <= max_edge:
            return image
        scale = max_edge / float(max(w, h))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    @staticmethod
    def _get_device() -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def load(self) -> None:
        if self._loaded:
            return

        vl = _ensure_vl_imports()
        AutoProcessor = vl["AutoProcessor"]
        Qwen2VLForConditionalGeneration = vl["Qwen2VLForConditionalGeneration"]

        logger.info("加载 Qwen-VL 模型: %s (device=%s)", self.model_name, self.device)
        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                local_files_only=True,
                trust_remote_code=True,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map=None,
                local_files_only=True,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        except OSError as e:
            raise RuntimeError(
                f"本地未找到模型 {self.model_name}，请先下载：\n"
                f"  export HF_ENDPOINT=https://hf-mirror.com  # 国内可选\n"
                f"  huggingface-cli download {self.model_name}"
            ) from e
        self._model.to(self.device)
        self._model.eval()
        self._loaded = True
        logger.info("Qwen-VL 模型加载成功")

    def generate(
        self,
        image: Image.Image,
        user_text: str,
        system_prompt: str = "",
    ) -> str:
        """
        图片 + 文本端到端生成。user_text 可包含 RAG/KG 等上下文。
        """
        if not self._loaded:
            self.load()

        process_vision_info = _ensure_vl_imports()["process_vision_info"]

        system = system_prompt or (
            "你是地震应急领域的多模态专家助手，请结合图片与文字上下文，"
            "用简洁、准确的中文回答用户问题。"
        )
        combined_text = f"{system}\n\n{user_text}" if user_text else system
        if len(combined_text) > self.max_context_chars:
            combined_text = combined_text[: self.max_context_chars] + "\n…（上下文已截断）"

        rgb = self._resize_image(image.convert("RGB"), self.max_image_edge)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": rgb,
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    },
                    {"type": "text", "text": combined_text},
                ],
            },
        ]

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": 1.1,
        }
        if self.do_sample:
            gen_kwargs.update(
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
            )
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        decoded = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return (decoded[0] if decoded else "").strip()


def build_qwen_vl_handler(config=None) -> QwenVLHandler:
    return QwenVLHandler(config)


qwen_vl_handler = QwenVLHandler()
