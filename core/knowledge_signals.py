#!/usr/bin/env python3
"""调度前知识源探测：静态置信度、动态可用性、时效标签。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

from config.constants import match_region_in_text
from core.phase_classifier import Phase, PhaseResult

_TEMPORAL_DYNAMIC = frozenset({"实时", "政策更新"})
_VALIDITY_RE = re.compile(r"时效[：:]\s*(\S+)")


@dataclass
class KnowledgeSignals:
    static_confidence: float = 0.0
    dynamic_availability: float = 0.0
    rag_top_score: float = 0.0
    kg_region_hit: bool = False
    kg_emergency_hit: bool = False
    kg_magnitude_hit: bool = False
    temporal_validity: str = ""
    prefers_dynamic: bool = False
    rag_hits: List[dict] = field(default_factory=list)
    dynamic_result: Any = None
    kg_region_snippet: str = ""
    kg_emergency_snippet: str = ""


def _extract_temporal_validity(text: str) -> str:
    if not text:
        return ""
    m = _VALIDITY_RE.search(text)
    return m.group(1) if m else ""


def _calc_static_confidence(signals: KnowledgeSignals) -> float:
    score = 0.0
    if signals.kg_region_hit:
        score += 0.28
    if signals.kg_magnitude_hit:
        score += 0.22
    if signals.kg_emergency_hit:
        score += 0.28
    if signals.rag_top_score > 0.25:
        score += min(signals.rag_top_score, 1.0) * 0.35
    elif signals.rag_top_score > 0:
        score += signals.rag_top_score * 0.15
    return min(score, 1.0)


def _should_probe_dynamic(
    input_text: str,
    phase_tag: str,
    phase_result: PhaseResult,
    signals: KnowledgeSignals,
) -> bool:
    if phase_result.need_dynamic:
        return True
    if phase_result.phase in (Phase.DURING, Phase.POST):
        return True
    if signals.prefers_dynamic:
        return True
    dynamic_kw = ("最新", "实时", "刚才", "现在", "多大", "震中", "余震")
    return any(k in input_text for k in dynamic_kw)


def compute_knowledge_signals(
    input_text: str,
    phase_tag: str,
    phase_result: Optional[PhaseResult],
    *,
    config: Any,
    kg: Any = None,
    emergency_rag: Any = None,
    dynamic_retriever: Any = None,
) -> KnowledgeSignals:
    """在调度决策前探测各知识源可用性与静态匹配置信度。"""
    signals = KnowledgeSignals()
    if not phase_result:
        return signals

    if getattr(config, "RAG_ENABLED", True) and emergency_rag is not None:
        top_k = int(getattr(config, "RAG_TOP_K", 5))
        try:
            hits = emergency_rag.search(input_text, top_k=top_k)
            signals.rag_hits = hits or []
            if hits:
                signals.rag_top_score = float(hits[0].get("score", 0))
                signals.temporal_validity = _extract_temporal_validity(
                    hits[0].get("text", "")
                )
        except Exception:
            pass

    if getattr(config, "KG_CONTEXT_ENABLED", True) and kg is not None:
        region = match_region_in_text(input_text)
        if region:
            try:
                region_results = kg.query_earthquakes_by_region(region)
                if region_results:
                    signals.kg_region_hit = True
                    lines = ["【知识图谱信息】\n"]
                    for i, eq in enumerate(region_results[:3]):
                        lines.append(f"{i+1}. {eq['location']}地震：\n")
                        lines.append(f"   时间：{eq['time']}\n")
                        lines.append(f"   震级：{eq['magnitude']}级\n")
                        lines.append(f"   深度：{eq['depth']}公里\n")
                        lines.append(f"   烈度：{eq['intensity']}\n")
                        lines.append(f"   描述：{eq['description']}\n\n")
                    signals.kg_region_snippet = "".join(lines)
            except Exception:
                pass

        if "震级" in input_text and ("大于" in input_text or "高于" in input_text):
            m = re.search(r"(大于|高于)(\d+\.?\d*)", input_text)
            if m:
                try:
                    if kg.query_earthquakes_by_magnitude(float(m.group(2))):
                        signals.kg_magnitude_hit = True
                except Exception:
                    pass

        try:
            emg = kg.query_emergency_context(input_text, phase_tag=phase_tag)
            if emg:
                signals.kg_emergency_hit = True
                signals.kg_emergency_snippet = emg
        except Exception:
            pass

    signals.prefers_dynamic = (
        signals.temporal_validity in _TEMPORAL_DYNAMIC
        or "实时" in input_text
        or "最新" in input_text
    )

    signals.static_confidence = _calc_static_confidence(signals)

    if (
        getattr(config, "DYNAMIC_RETRIEVAL_ENABLED", True)
        and dynamic_retriever is not None
        and dynamic_retriever.enabled
        and _should_probe_dynamic(input_text, phase_tag, phase_result, signals)
    ):
        try:
            dr = dynamic_retriever.fetch_for_phase(phase_tag, input_text)
            signals.dynamic_result = dr
            if getattr(dr, "error", None):
                signals.dynamic_availability = 0.0
            elif dr.items:
                signals.dynamic_availability = 1.0 if getattr(dr, "is_fresh", True) else 0.55
            else:
                signals.dynamic_availability = 0.15
        except Exception:
            signals.dynamic_availability = 0.0

    return signals
