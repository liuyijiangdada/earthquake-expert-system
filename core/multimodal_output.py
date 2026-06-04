#!/usr/bin/env python3
"""多模态输出模块：根据用户问题和阶段，匹配并返回图片、链接等多模态资源。"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MediaResource:
    id: str = ""
    type: str = ""
    url: str = ""
    caption: str = ""
    source: str = ""
    phase: str = ""
    matched_keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "url": self.url,
            "caption": self.caption,
            "source": self.source,
            "phase": self.phase,
        }


class MultimodalOutput:
    def __init__(self, config=None):
        self._resources: List[Dict[str, Any]] = []
        self._enabled = True
        self._max_per_type = 2

        if config:
            self._enabled = getattr(config, "MULTIMODAL_OUTPUT_ENABLED", True)
            self._max_per_type = getattr(config, "MULTIMODAL_MAX_PER_TYPE", 2)

        self._load_resources()

    def _load_resources(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "multimodal_resources.json",
        )
        if not os.path.isfile(path):
            logger.warning("多模态资源文件不存在: %s", path)
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._resources = data.get("resources", [])
            logger.info("已加载 %d 个多模态资源", len(self._resources))
        except Exception as e:
            logger.error("加载多模态资源失败: %s", e)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def match(self, user_text: str, phase_tag: str = "") -> List[MediaResource]:
        if not self._enabled or not self._resources:
            return []

        if not user_text or not user_text.strip():
            return []

        text_lower = user_text.lower()
        scored: List[tuple] = []

        for res in self._resources:
            keywords = res.get("keywords", [])
            res_phase = res.get("phase", "")

            kw_hits = sum(1 for kw in keywords if kw in text_lower)
            if kw_hits == 0:
                continue

            phase_bonus = 2.0 if (phase_tag and res_phase == phase_tag) else 0.0
            score = kw_hits * 1.0 + phase_bonus

            matched_kw = [kw for kw in keywords if kw in text_lower]
            scored.append((score, res, matched_kw))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: List[MediaResource] = []
        type_counts: Dict[str, int] = {}

        for score, res, matched_kw in scored:
            res_type = res.get("type", "link")
            if type_counts.get(res_type, 0) >= self._max_per_type:
                continue

            mr = MediaResource(
                id=res.get("id", ""),
                type=res_type,
                url=res.get("url", ""),
                caption=res.get("caption", ""),
                source=res.get("source", ""),
                phase=res.get("phase", ""),
                matched_keywords=matched_kw,
            )
            results.append(mr)
            type_counts[res_type] = type_counts.get(res_type, 0) + 1

        return results

    def match_as_dicts(self, user_text: str, phase_tag: str = "") -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.match(user_text, phase_tag)]

    def build_media_section(self, user_text: str, phase_tag: str = "") -> str:
        resources = self.match(user_text, phase_tag)
        if not resources:
            return ""

        lines = ["【相关多媒体资源】"]
        for r in resources:
            if r.type == "image":
                lines.append(f"  图片：{r.caption}（{r.url}）")
            elif r.type == "link":
                lines.append(f"  链接：{r.caption}（{r.url}）")
            elif r.type == "video":
                lines.append(f"  视频：{r.caption}（{r.url}）")

        return "\n".join(lines) + "\n"
