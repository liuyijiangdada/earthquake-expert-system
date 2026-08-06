#!/usr/bin/env python3
"""不确定性感知调度器：根据阶段、置信度、紧急度与知识源探测结果决定路由策略。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from core.phase_classifier import Phase, PhaseResult

logger = logging.getLogger(__name__)

try:
    from core.knowledge_signals import KnowledgeSignals
except ImportError:
    KnowledgeSignals = None  # type: ignore


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
    static_confidence: float = 0.0
    dynamic_availability: float = 0.0
    reliability_hint: str = ""


class Scheduler:
    def __init__(self, config=None):
        self._dynamic_confidence_threshold = getattr(
            config, "SCHEDULER_DYNAMIC_CONFIDENCE_THRESHOLD", 0.4
        )
        self._static_confidence_threshold = getattr(
            config, "SCHEDULER_STATIC_CONFIDENCE_THRESHOLD", 0.9
        )
        self._urgency_high_threshold = getattr(
            config, "SCHEDULER_URGENCY_HIGH_THRESHOLD", 0.5
        )
        self._urgency_critical_threshold = getattr(
            config, "SCHEDULER_URGENCY_CRITICAL_THRESHOLD", 0.7
        )

    def decide(
        self,
        phase_result: PhaseResult,
        signals: Optional["KnowledgeSignals"] = None,
    ) -> ScheduleDecision:
        try:
            phase = phase_result.phase
            confidence = phase_result.confidence
            urgency = phase_result.urgency
            need_dynamic = phase_result.need_dynamic

            d = ScheduleDecision()
            d.phase_label = phase.value

            if signals is not None:
                d.static_confidence = signals.static_confidence
                d.dynamic_availability = signals.dynamic_availability

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

            if signals is not None:
                d = self._apply_signal_policy(d, phase_result, signals)

            d.reasoning = self._build_reasoning(phase_result, d, signals)
            return d
        except Exception as e:
            logger.error("调度决策异常: %s", e, exc_info=True)
            fallback = ScheduleDecision()
            fallback.reasoning = f"调度异常：{type(e).__name__}，已降级为全源默认策略"
            return fallback

    def _apply_signal_policy(
        self,
        d: ScheduleDecision,
        phase_result: PhaseResult,
        signals: "KnowledgeSignals",
    ) -> ScheduleDecision:
        urgency = phase_result.urgency
        sc = signals.static_confidence
        da = signals.dynamic_availability

        if (
            urgency >= self._urgency_critical_threshold
            and da >= self._dynamic_confidence_threshold
        ):
            d.use_dynamic = True
            d.dynamic_priority = "high"
            d.reliability_hint = "高紧急度且实时数据可用，优先结合动态信息作答。"
        elif sc >= self._static_confidence_threshold:
            if urgency < self._urgency_critical_threshold:
                d.use_dynamic = False
                d.dynamic_priority = "low"
            suffix = (
                "本地知识库匹配度较高；如需最新震情可追问「最新」「刚才」等关键词。"
            )
            d.prompt_suffix = f"{d.prompt_suffix} {suffix}".strip()
            d.reliability_hint = "静态知识置信度高，回答以本地知识库为主。"
        elif signals.prefers_dynamic and da >= self._dynamic_confidence_threshold:
            d.use_dynamic = True
            d.dynamic_priority = "high"
            d.reliability_hint = "问题时效性强，已启用动态数据源。"
        elif sc < 0.35 and da < self._dynamic_confidence_threshold:
            d.rag_priority = "high"
            d.reliability_hint = "本地匹配较弱，已提高向量检索权重。"
        elif sc < self._static_confidence_threshold and not d.reliability_hint:
            d.reliability_hint = "已依据本地知识库生成回答。"

        if d.use_dynamic and da < self._dynamic_confidence_threshold:
            d.use_dynamic = False
            d.dynamic_priority = "low"
            extra = "实时数据暂不可用，以下回答主要依据本地知识库。"
            d.validity_hint = f"{d.validity_hint} {extra}".strip()
            d.reliability_hint = "动态源不可用，已降级为静态知识。"

        return d

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
                "紧急！当前为震中应急阶段：若有【动态信息】，先明确写出震级、地点与时间，语气肯定；"
                "再给出可执行避险指令，句式简短，禁止“无法确定/尚未确认”等犹豫表述。"
            )
            d.validity_hint = (
                "（速报更新于{fetched_at}；如需核对可稍后刷新查询）"
            )
        else:
            d.prompt_suffix = (
                "当前为震中应急阶段：优先、肯定地引用【动态信息】中的震级与震中；"
                "再补充避险要点。禁止“无法准确确定”“等待正式报告”等推诿句式。"
            )
            d.validity_hint = "（以上为速报口径，可刷新获取最新条目）"

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

    def _build_reasoning(
        self,
        phase_result: PhaseResult,
        d: ScheduleDecision,
        signals: Optional["KnowledgeSignals"] = None,
    ) -> str:
        sources = []
        if d.use_kg:
            sources.append(f"知识图谱({d.kg_priority})")
        if d.use_rag:
            sources.append(f"向量检索({d.rag_priority})")
        if d.use_dynamic:
            sources.append(f"动态数据({d.dynamic_priority})")

        parts = [
            f"阶段={d.phase_label}",
            f"阶段置信度={phase_result.confidence:.2f}",
            f"紧急度={phase_result.urgency:.2f}",
        ]
        if signals is not None:
            parts.append(f"静态置信度={d.static_confidence:.2f}")
            parts.append(f"动态可用性={d.dynamic_availability:.2f}")
            if signals.temporal_validity:
                parts.append(f"知识时效={signals.temporal_validity}")
        parts.append(f"启用知识源=[{', '.join(sources)}]")
        if d.reliability_hint:
            parts.append(d.reliability_hint)
        return "；".join(parts)
