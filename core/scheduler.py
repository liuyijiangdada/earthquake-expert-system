#!/usr/bin/env python3
"""不确定性感知调度器：根据阶段、置信度、紧急度决定知识源路由策略。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from core.phase_classifier import Phase, PhaseResult

logger = logging.getLogger(__name__)


@dataclass
class ScheduleDecision:
    use_kg: bool = True
    use_rag: bool = True
    use_dynamic: bool = False
    kg_priority: str = "normal"
    rag_priority: str = "normal"
    dynamic_priority: str = "low"
    prompt_suffix: str = ""
    validity_hint: str = ""
    phase_label: str = "通用"
    reasoning: str = ""


class Scheduler:
    def __init__(self, config=None):
        self._dynamic_confidence_threshold = getattr(
            config, "SCHEDULER_DYNAMIC_CONFIDENCE_THRESHOLD", 0.4
        )
        self._urgency_high_threshold = getattr(
            config, "SCHEDULER_URGENCY_HIGH_THRESHOLD", 0.5
        )
        self._urgency_critical_threshold = getattr(
            config, "SCHEDULER_URGENCY_CRITICAL_THRESHOLD", 0.7
        )

    def decide(self, phase_result: PhaseResult) -> ScheduleDecision:
        try:
            phase = phase_result.phase
            confidence = phase_result.confidence
            urgency = phase_result.urgency
            need_dynamic = phase_result.need_dynamic

            d = ScheduleDecision()
            d.phase_label = phase.value

            if phase == Phase.PRE:
                d = self._schedule_pre(d, confidence, urgency)
            elif phase == Phase.DURING:
                d = self._schedule_during(d, confidence, urgency)
            elif phase == Phase.POST:
                d = self._schedule_post(d, confidence, urgency)
            else:
                d = self._schedule_general(d, confidence, urgency)

            if need_dynamic and not d.use_dynamic:
                if urgency > self._urgency_high_threshold:
                    d.use_dynamic = True
                    d.dynamic_priority = "high"

            d.reasoning = self._build_reasoning(phase_result, d)
            return d
        except Exception as e:
            logger.error("调度决策异常: %s", e, exc_info=True)
            fallback = ScheduleDecision()
            fallback.reasoning = f"调度异常：{type(e).__name__}，已降级为全源默认策略"
            return fallback

    def _schedule_pre(self, d: ScheduleDecision, confidence: float,
                      urgency: float) -> ScheduleDecision:
        d.use_kg = True
        d.use_rag = True
        d.use_dynamic = False
        d.kg_priority = "normal"
        d.rag_priority = "high"
        d.dynamic_priority = "low"
        d.prompt_suffix = (
            "当前为震前防御阶段，请侧重科普知识与预防准备建议。"
        )
        d.validity_hint = "（知识有效期：长期有效，除非规范修订）"
        return d

    def _schedule_during(self, d: ScheduleDecision, confidence: float,
                         urgency: float) -> ScheduleDecision:
        d.use_kg = True
        d.use_rag = True
        d.use_dynamic = True
        d.kg_priority = "high"
        d.rag_priority = "normal"
        d.dynamic_priority = "high"

        if urgency >= self._urgency_critical_threshold:
            d.prompt_suffix = (
                "紧急！当前为震中应急阶段，请优先给出可执行的安全指令，"
                "使用简短明确的句式，避免冗长解释。"
            )
            d.validity_hint = (
                "（数据更新于{fetched_at}，余震序列可能变化，"
                "建议{refresh_minutes}分钟后再次查询）"
            )
        else:
            d.prompt_suffix = (
                "当前为震中应急阶段，请结合实时数据与应急知识给出避险指引。"
            )
            d.validity_hint = "（实时数据可能变化，建议定期刷新查询）"

        return d

    def _schedule_post(self, d: ScheduleDecision, confidence: float,
                       urgency: float) -> ScheduleDecision:
        d.use_kg = True
        d.use_rag = True
        d.use_dynamic = urgency > self._urgency_high_threshold
        d.kg_priority = "normal"
        d.rag_priority = "high"
        d.dynamic_priority = "medium" if d.use_dynamic else "low"
        d.prompt_suffix = (
            "当前为震后恢复阶段，请侧重灾后恢复指引、政策信息与安全评估建议。"
        )
        d.validity_hint = "（政策信息请以官方最新发布为准）"
        return d

    def _schedule_general(self, d: ScheduleDecision, confidence: float,
                          urgency: float) -> ScheduleDecision:
        d.use_kg = True
        d.use_rag = True
        d.use_dynamic = urgency > self._urgency_critical_threshold
        d.kg_priority = "normal"
        d.rag_priority = "normal"
        d.dynamic_priority = "medium" if d.use_dynamic else "low"

        if urgency > self._urgency_high_threshold:
            d.prompt_suffix = "请优先给出简洁明确的回答。"
        else:
            d.prompt_suffix = ""

        d.validity_hint = ""
        return d

    def _build_reasoning(self, phase_result: PhaseResult,
                         d: ScheduleDecision) -> str:
        sources = []
        if d.use_kg:
            sources.append(f"知识图谱({d.kg_priority})")
        if d.use_rag:
            sources.append(f"向量检索({d.rag_priority})")
        if d.use_dynamic:
            sources.append(f"动态数据({d.dynamic_priority})")

        return (
            f"阶段={d.phase_label}, "
            f"置信度={phase_result.confidence:.2f}, "
            f"紧急度={phase_result.urgency:.2f}, "
            f"启用知识源=[{', '.join(sources)}]"
        )
