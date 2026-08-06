#!/usr/bin/env python3
"""模型回答质量检测与 RAG 降级。"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_EQ_KEYWORDS = (
    "地震", "震", "避险", "应急", "撤离", "安全", "余震", "建筑", "受伤",
    "救援", "演练", "防灾", "震感", "避难", "自救", "互救", "滑坡", "燃气",
)

# 震级/震中事实问句
_QUAKE_FACT_QUERY = re.compile(
    r"(刚才|刚刚|这次|此次|最新|实时).*(地震|震)|"
    r"(地震|震).*(多大|几级|震级|震中|位置|哪里|哪儿|震源)|"
    r"(震级|震中).*(多少|哪|什么)|"
    r"发生(了)?(什么|哪).*(地震|震)"
)

_HEDGE_MARKERS = (
    "无法准确确定",
    "无法确定",
    "尚未得到权威",
    "尚未确认",
    "尚未有准确",
    "需要进一步核实",
    "具体震级需要参考",
    "具体震级需",
    "等待正式报告",
    "正式报告结果",
    "仅供参考",
    "暂无法给出",
)

_GARBAGE_MARKERS = (
    "UITableView", "Assistant,", "参考消息网", "蹲牙", "丹心图强", "严责历史",
    "本人企业", "震区任避", "绿豆西湖", "教学日志事", "一玉山", "一玉带",
    "Napište", "Zlepšení", "Human:", "User:",
)

_EVAL_PREFIX = re.compile(r"^参考答案[:：]?\s*")
# 小模型偶发在句末退化成一长串感叹号/问号等
_DEGENERATE_PUNCT = re.compile(r"([!！?？.。…~～])\1{7,}")
_COLLAPSE_PUNCT = re.compile(r"([!！?？~～])\1{2,}")
_TRAILING_BANGS = re.compile(r"[!！]{2,}\s*$")


def strip_eval_prefix(text: str) -> str:
    return _EVAL_PREFIX.sub("", (text or "").strip()).strip()


def has_degenerate_punctuation(text: str) -> bool:
    return bool(_DEGENERATE_PUNCT.search(text or ""))


def sanitize_repeated_punctuation(text: str) -> str:
    """折叠连续标点，并去掉句末退化的感叹号串。"""
    cleaned = text or ""
    cleaned = _COLLAPSE_PUNCT.sub(r"\1", cleaned)
    cleaned = _TRAILING_BANGS.sub("", cleaned)
    return cleaned.strip()


def build_rag_fallback_text(hits: List[Dict[str, Any]], max_topics: int = 1) -> str:
    if not hits:
        return ""
    parts: List[str] = []
    for hit in hits[:max_topics]:
        title = (hit.get("title") or "").strip()
        body = (hit.get("text") or "").strip()
        if not body:
            continue
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        if title and lines and lines[0] == title:
            lines = lines[1:]
        step_lines: List[str] = []
        for i, line in enumerate(lines[:6], 1):
            if re.match(r"^\d+[\.、．)]", line):
                step_lines.append(line)
            else:
                step_lines.append(f"{i}. {line}")
        if not step_lines:
            continue
        header = f"根据本地应急知识库「{title}」：" if title else "根据本地应急知识库："
        parts.append(header + "\n" + "\n".join(step_lines))
    return "\n\n".join(parts).strip()


def is_garbled_response(text: str) -> bool:
    cleaned = strip_eval_prefix(text)
    if not cleaned or len(cleaned) < 8:
        return True

    if any(marker in cleaned for marker in _GARBAGE_MARKERS):
        return True

    if has_degenerate_punctuation(cleaned):
        return True

    latin = sum(1 for c in cleaned if c.isascii() and c.isalpha())
    if latin / max(len(cleaned), 1) > 0.12:
        return True

    weird = sum(1 for c in cleaned if c in "[]{}<>|\\/@#$%^&*")
    if weird > 2:
        return True

    has_keyword = any(kw in cleaned for kw in _EQ_KEYWORDS)
    has_steps = bool(re.search(r"(?:^|\n)\s*[1-9一二三四五][\.、．)]", cleaned))
    if not has_keyword and not has_steps:
        return True

    # 连续非常用汉字片段，常见于小模型幻觉
    if re.search(r"[\u4e00-\u9fff]{1,2}[/／][\u4e00-\u9fff]", cleaned):
        return True

    return False


def is_quake_fact_query(text: str) -> bool:
    """是否在问当前/最新地震的震级或震中。"""
    q = (text or "").strip()
    if not q:
        return False
    if _QUAKE_FACT_QUERY.search(q):
        return True
    # 极短口语：多大 / 震中在哪
    compact = re.sub(r"\s+", "", q)
    return bool(
        re.search(r"(多大|几级|震级|震中|震源)", compact)
        and ("震" in compact or "地震" in compact or len(compact) <= 12)
    )


def has_hedging_language(text: str) -> bool:
    t = text or ""
    return any(m in t for m in _HEDGE_MARKERS)


def build_confident_dynamic_answer(
    items: List[Dict[str, Any]],
    *,
    source: str = "",
    fetched_at: str = "",
) -> str:
    """用动态速报字段拼肯定结论（不经过小模型）。"""
    if not items:
        return ""
    top = items[0] or {}
    mag = top.get("magnitude", "?")
    loc = top.get("location") or "未知区域"
    t = top.get("time") or ""
    depth = top.get("depth", "?")
    src = (top.get("source") or source or "速报").strip()
    src_label = src.upper() if len(src) <= 8 else src

    parts = [f"根据{src_label}最新速报，本次地震震级为{mag}级，震中位于{loc}"]
    if depth not in (None, "", "?"):
        parts[0] += f"，震源深度约{depth}公里"
    parts[0] += "。"
    if t:
        parts.append(f"发震时间：{t}。")
    if fetched_at:
        parts.append(f"（速报更新于 {fetched_at}）")
    return "".join(parts)


def maybe_confident_quake_answer(
    response_meta: Optional[Dict[str, Any]],
    *,
    user_query: str = "",
    force: bool = False,
) -> str:
    """有动态条目且问震级/震中时，返回肯定模板；否则空串。"""
    meta = response_meta or {}
    query = (user_query or meta.get("user_query") or "").strip()
    items = meta.get("dynamic_items") or []
    if not items:
        return ""
    if not force and not is_quake_fact_query(query):
        return ""
    return build_confident_dynamic_answer(
        items,
        source=str(meta.get("dynamic_source") or ""),
        fetched_at=str(meta.get("dynamic_fetched_at") or ""),
    )


def guard_response(
    text: str,
    response_meta: Optional[Dict[str, Any]] = None,
    *,
    static_confidence: Optional[float] = None,
) -> str:
    """检测低质量回答，必要时降级为 RAG 检索摘要。"""
    response_meta = response_meta or {}
    cleaned = strip_eval_prefix(text or "")
    degenerate = has_degenerate_punctuation(cleaned)
    if degenerate or _COLLAPSE_PUNCT.search(cleaned) or _TRAILING_BANGS.search(cleaned):
        cleaned = sanitize_repeated_punctuation(cleaned)
        response_meta["response_sanitized"] = True
        if degenerate:
            response_meta["response_quality"] = "degenerate_punctuation"

    # 已由动态速报短路生成的肯定回答，不再二次改写/降级
    if response_meta.get("response_fallback") == "dynamic_confident" and cleaned:
        return cleaned

    query = str(response_meta.get("user_query") or "")
    # 推诿/乱码的震情事实回答：有速报则改写为肯定结论
    if is_quake_fact_query(query) and (
        has_hedging_language(cleaned) or is_garbled_response(cleaned) or degenerate
    ):
        direct = maybe_confident_quake_answer(response_meta, force=True)
        if direct:
            response_meta["response_fallback"] = "dynamic_confident"
            response_meta["response_quality"] = "hedge_rewritten"
            return direct

    fallback = (response_meta.get("rag_fallback_text") or "").strip()
    sc = static_confidence
    if sc is None:
        sc = response_meta.get("static_confidence")

    low_conf = sc is not None and float(sc) < 0.7
    garbled = is_garbled_response(cleaned) or degenerate

    if garbled or (low_conf and len(cleaned) < 40):
        if fallback:
            response_meta["response_fallback"] = "rag"
            if garbled:
                response_meta["response_quality"] = "garbled"
            elif low_conf:
                response_meta["response_quality"] = "low_confidence"
            return fallback

    if garbled and not fallback:
        response_meta["response_quality"] = "garbled"
        return "抱歉，本次生成内容异常。请换种问法重试，或参考下方相关示意图与知识条目。"

    if cleaned != (text or "").strip():
        response_meta["response_sanitized"] = True

    return cleaned if cleaned else text
