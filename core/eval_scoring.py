#!/usr/bin/env python3
"""消融自动评分：旧 grounding 口径与证据归一化口径并列。"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

PHASE_KEYWORDS = {
    "震前": ("应急", "准备", "物资", "预案", "演练", "科普", "储备", "检查", "预警"),
    "震中": ("避险", "躲避", "撤离", "护头", "室外", "室内", "余震", "避难", "电梯", "疏散"),
    "震后": ("安全", "鉴定", "重建", "补贴", "恢复", "心理", "防疫", "理赔", "安置", "评估"),
}

_EMPTY_MARKERS = ("（无）", "（本路径已关闭）", "无需启用", "无相关", "暂不可用")


@dataclass
class Scores:
    factual: float
    completeness: float
    format_ok: int


@dataclass
class DualScores:
    legacy: Scores
    normalized: Scores


def _tokens(text: str) -> set:
    return set(re.findall(r"[\u4e00-\u9fff]{2,}", text or ""))


def extract_prompt_section(prompt: str, marker: str, end_markers: List[str]) -> str:
    if marker not in (prompt or ""):
        return ""
    start = prompt.index(marker) + len(marker)
    end = len(prompt)
    for em in end_markers:
        idx = prompt.find(em, start)
        if idx != -1:
            end = min(end, idx)
    return prompt[start:end].strip()


def extract_refs(prompt: str) -> Tuple[str, str, str]:
    kg_ref = extract_prompt_section(
        prompt or "", "【知识图谱】", ["【参考资料】", "【动态信息】", "规则", "【问题】"]
    )
    rag_ref = extract_prompt_section(
        prompt or "", "【参考资料】", ["【动态信息】", "规则", "【问题】"]
    )
    dyn_ref = extract_prompt_section(prompt or "", "【动态信息】", ["规则", "【问题】"])
    if not dyn_ref:
        dyn_ref = extract_prompt_section(
            prompt or "", "【动态信息·", ["规则", "【问题】", "【知识图谱】"]
        )
    return kg_ref, rag_ref, dyn_ref


def content_body(response: str) -> str:
    body = response or ""
    for m in ("【答案】", "【知识点】", "【参考解析】", "</s>", "<|im_start|>"):
        body = body.replace(m, "")
    return body.strip()


def overlap_ratio(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not tb:
        return 0.0
    return len(ta & tb) / len(tb)


def _format_ok(response: str, body: str) -> int:
    fmt = 1
    if not body or len(body) < 20:
        fmt = 0
    if any(x in (response or "") for x in ("失败", "重试", "错误")):
        fmt = 0
    if "**" in (response or ""):
        fmt = 0
    if body.startswith("【") and len(body) < 50:
        fmt = 0
    return fmt


def is_real_evidence(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 12:
        return False
    if any(m in t for m in _EMPTY_MARKERS):
        return False
    if "已关闭" in t or "获取失败" in t:
        return False
    return True


def _completeness(
    body: str,
    phase: str,
    *,
    rag_hit: bool,
    kg_hit: bool,
    overlap_kg: float,
    meta: Optional[dict],
) -> float:
    completeness = 1.0
    n = len(body)
    if n >= 35:
        completeness = 2.0
    if n >= 70:
        completeness = 2.8
    if n >= 120:
        completeness = 3.4
    if n >= 180:
        completeness = 3.9
    if n >= 260:
        completeness = 4.4

    phase_kws = PHASE_KEYWORDS.get(phase, ())
    hits = sum(1 for k in phase_kws if k in body)
    completeness += min(hits * 0.18, 1.0)

    if re.search(r"[1-9][\.\)、．]", body) or body.count("。") >= 3:
        completeness += 0.35

    if rag_hit and meta and meta.get("rag_topic_ids"):
        completeness += 0.25
    if kg_hit and overlap_kg > 0.08:
        completeness += 0.2

    return max(1.0, min(5.0, completeness))


def score_response(
    response: str,
    prompt: str,
    meta: Optional[dict],
    phase: str,
    *,
    kg_on: bool,
    rag_on: bool,
) -> Scores:
    """旧 grounding 口径：图谱权重大于检索，B0 有长度分并 cap 在 0.42。"""
    body = content_body(response)
    fmt = _format_ok(response, body)
    kg_ref, rag_ref, dyn_ref = extract_refs(prompt)

    kg_hit = bool(kg_on and is_real_evidence(kg_ref))
    rag_hit = bool(rag_on and is_real_evidence(rag_ref))
    dyn_hit = bool(is_real_evidence(dyn_ref))

    overlap_kg = overlap_ratio(body, kg_ref) if kg_hit else 0.0
    overlap_rag = overlap_ratio(body, rag_ref) if rag_hit else 0.0
    overlap_dyn = overlap_ratio(body, dyn_ref) if dyn_hit else 0.0

    factual = 0.0
    if len(body) >= 20:
        factual = 0.12

    if not kg_on and not rag_on:
        factual += min(0.18, len(body) / 600)
        if re.search(r"20\d{2}年.{0,8}[6-9]\.\d级", body):
            factual = max(0.0, factual - 0.15)
        factual = min(factual, 0.42)
        if dyn_hit:
            factual = min(1.0, factual + 0.12 + 0.25 * min(overlap_dyn * 2.5, 1.0))
    else:
        if kg_hit:
            factual += 0.22 + 0.38 * min(overlap_kg * 3.0, 1.0)
        if rag_hit:
            factual += 0.18 + 0.32 * min(overlap_rag * 3.0, 1.0)
        if dyn_hit:
            factual += 0.12 + 0.25 * min(overlap_dyn * 2.5, 1.0)
        if kg_on and rag_on and kg_hit and rag_hit:
            factual += 0.1

    if meta:
        sc = float(meta.get("static_confidence") or 0)
        if sc >= 0.45 and (kg_hit or rag_hit):
            factual += 0.08
        if meta.get("rag_topic_ids") and rag_on:
            factual += 0.06
        if int(meta.get("dynamic_items_count") or 0) > 0:
            factual += 0.07

    factual = max(0.0, min(1.0, factual))
    completeness = _completeness(
        body, phase, rag_hit=rag_hit, kg_hit=kg_hit, overlap_kg=overlap_kg, meta=meta
    )
    return Scores(factual=factual, completeness=completeness, format_ok=fmt)


def score_response_normalized(
    response: str,
    prompt: str,
    meta: Optional[dict],
    phase: str,
    *,
    kg_on: bool = True,
    rag_on: bool = True,
) -> Scores:
    """证据归一化口径：只对实际注入的证据块算 overlap，KG/RAG/动态同一公式。

    取消 KG+RAG 额外加分与 B0 长度 cap。无注入证据时事实分为 0（无法 grounding）。
    kg_on/rag_on 保留以兼容调用方，归一化以 prompt 中真实证据为准。
    """
    del kg_on, rag_on
    body = content_body(response)
    fmt = _format_ok(response, body)
    kg_ref, rag_ref, dyn_ref = extract_refs(prompt)

    blocks: List[str] = []
    if is_real_evidence(kg_ref):
        blocks.append(kg_ref)
    if is_real_evidence(rag_ref):
        blocks.append(rag_ref)
    if is_real_evidence(dyn_ref):
        blocks.append(dyn_ref)

    kg_hit = is_real_evidence(kg_ref)
    rag_hit = is_real_evidence(rag_ref)
    overlap_kg = overlap_ratio(body, kg_ref) if kg_hit else 0.0

    if not blocks:
        factual = 0.0
    else:
        parts = [min(overlap_ratio(body, b) * 3.0, 1.0) for b in blocks]
        factual = sum(parts) / len(parts)

    factual = max(0.0, min(1.0, factual))
    completeness = _completeness(
        body, phase, rag_hit=rag_hit, kg_hit=kg_hit, overlap_kg=overlap_kg, meta=meta
    )
    return Scores(factual=factual, completeness=completeness, format_ok=fmt)


def score_both(
    response: str,
    prompt: str,
    meta: Optional[dict],
    phase: str,
    *,
    kg_on: bool,
    rag_on: bool,
) -> DualScores:
    return DualScores(
        legacy=score_response(
            response, prompt, meta, phase, kg_on=kg_on, rag_on=rag_on
        ),
        normalized=score_response_normalized(
            response, prompt, meta, phase, kg_on=kg_on, rag_on=rag_on
        ),
    )


def scores_to_dict(sc: Scores) -> Dict[str, Any]:
    return {
        "factual": round(sc.factual, 4),
        "completeness": round(sc.completeness, 2),
        "format_ok": sc.format_ok,
    }


def dual_to_dict(dual: DualScores) -> Dict[str, Any]:
    return {
        "legacy": scores_to_dict(dual.legacy),
        "normalized": scores_to_dict(dual.normalized),
    }


def truncate_ref(text: str, limit: int = 1200) -> str:
    t = text or ""
    return t if len(t) <= limit else t[:limit] + "…"
