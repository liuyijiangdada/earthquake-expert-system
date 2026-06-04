#!/usr/bin/env python3
"""三阶段问题分类器：根据用户问句判断所属地震阶段（震前/震中/震后）及知识时效需求。"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)


class Phase(str, Enum):
    PRE = "震前"
    DURING = "震中"
    POST = "震后"
    GENERAL = "通用"


@dataclass
class PhaseResult:
    phase: Phase = Phase.GENERAL
    confidence: float = 0.0
    urgency: float = 0.0
    need_dynamic: bool = False
    matched_keywords: List[str] = field(default_factory=list)
    reasoning: str = ""


_PRE_KEYWORDS = {
    "high": ["应急包", "准备", "演练", "预防", "防震", "预案", "物资储备", "应急准备",
             "家庭应急", "避难所", "设防", "抗震", "预警系统", "应急演练"],
    "medium": ["如何准备", "怎么预防", "地震前", "震前", "平时", "日常", "科普",
               "知识", "学习", "培训", "了解", "知道"],
    "low": ["什么", "定义", "概念", "类型", "分类", "原因", "形成", "原理"],
}

_DURING_KEYWORDS = {
    "high": ["现在", "刚才", "正在", "当前", "刚刚发生", "震级多大", "几级",
             "哪里地震", "震中在哪", "地震了", "地动了", "摇了"],
    "medium": ["怎么办", "避险", "躲避", "逃生", "自救", "保护", "躲", "跑",
               "室内", "室外", "高楼", "学校", "正在地震", "地震发生时"],
    "low": ["余震", "主震", "烈度", "震感", "晃动"],
}

_POST_KEYWORDS = {
    "high": ["恢复", "重建", "安置", "救助", "补偿", "鉴定", "评估", "复课",
             "灾后", "震后", "返回", "回家", "安全鉴定"],
    "medium": ["损失", "伤亡", "受灾", "救援", "安置点", "临时", "过渡",
               "政策", "补贴", "申请", "保险", "理赔"],
    "low": ["清理", "修复", "消毒", "防疫", "心理", "创伤"],
}

_URGENCY_BOOSTERS = [
    "现在", "马上", "立刻", "紧急", "急", "快", "刚才", "刚刚",
    "正在", "当前", "救命", "危险",
]


class PhaseClassifier:
    def __init__(self, custom_pre_keywords=None, custom_during_keywords=None,
                 custom_post_keywords=None):
        self.pre_kw = self._merge_keywords(_PRE_KEYWORDS, custom_pre_keywords)
        self.during_kw = self._merge_keywords(_DURING_KEYWORDS, custom_during_keywords)
        self.post_kw = self._merge_keywords(_POST_KEYWORDS, custom_post_keywords)

    @staticmethod
    def _merge_keywords(base: dict, extra: Optional[dict]) -> dict:
        merged = {k: list(v) for k, v in base.items()}
        if extra:
            for level, words in extra.items():
                merged.setdefault(level, []).extend(words)
        return merged

    def classify(self, text: str) -> PhaseResult:
        try:
            if not text or not text.strip():
                return PhaseResult(phase=Phase.GENERAL, confidence=0.0, urgency=0.0)

            text_lower = text.lower()
            scores = {
                Phase.PRE: self._score_phase(text_lower, self.pre_kw),
                Phase.DURING: self._score_phase(text_lower, self.during_kw),
                Phase.POST: self._score_phase(text_lower, self.post_kw),
            }

            matched_kw = {Phase.PRE: [], Phase.DURING: [], Phase.POST: []}
            for phase, kw_dict in [(Phase.PRE, self.pre_kw), (Phase.DURING, self.during_kw),
                                    (Phase.POST, self.post_kw)]:
                for level, words in kw_dict.items():
                    for w in words:
                        if w in text_lower:
                            matched_kw[phase].append(w)

            best_phase = max(scores, key=scores.get)
            best_score = scores[best_phase]
            total_score = sum(scores.values()) + 1e-9

            confidence = best_score / total_score if total_score > 1e-9 else 0.0

            if best_score < 0.3:
                best_phase = Phase.GENERAL
                confidence = 0.0

            urgency = self._calc_urgency(text_lower)

            need_dynamic = (
                best_phase == Phase.DURING
                or (best_phase == Phase.POST and urgency > 0.5)
                or urgency > 0.7
            )

            reasoning = self._build_reasoning(best_phase, scores, matched_kw, urgency)

            return PhaseResult(
                phase=best_phase,
                confidence=min(confidence, 1.0),
                urgency=urgency,
                need_dynamic=need_dynamic,
                matched_keywords=matched_kw.get(best_phase, []),
                reasoning=reasoning,
            )
        except Exception as e:
            logger.error("阶段分类异常: %s", e, exc_info=True)
            return PhaseResult(
                phase=Phase.GENERAL,
                confidence=0.0,
                urgency=0.0,
                reasoning=f"分类异常：{type(e).__name__}，已降级为通用",
            )

    def _score_phase(self, text: str, kw_dict: dict) -> float:
        score = 0.0
        for level, words in kw_dict.items():
            weight = {"high": 3.0, "medium": 2.0, "low": 1.0}.get(level, 1.0)
            for w in words:
                if w in text:
                    score += weight
        return score

    def _calc_urgency(self, text: str) -> float:
        hits = sum(1 for b in _URGENCY_BOOSTERS if b in text)
        q_marks = text.count("?") + text.count("？")
        ex_marks = text.count("!") + text.count("！")
        raw = hits * 0.25 + q_marks * 0.05 + ex_marks * 0.1
        return min(raw, 1.0)

    def _build_reasoning(self, phase: Phase, scores: dict,
                         matched: dict, urgency: float) -> str:
        parts = []
        if phase != Phase.GENERAL:
            parts.append(f"判定为{phase.value}阶段")
            kw_list = matched.get(phase, [])
            if kw_list:
                parts.append(f"命中关键词：{'、'.join(kw_list[:5])}")
        else:
            parts.append("未匹配明确阶段关键词，判定为通用")

        if urgency > 0.5:
            parts.append(f"紧急度较高({urgency:.1f})")
        return "；".join(parts)
