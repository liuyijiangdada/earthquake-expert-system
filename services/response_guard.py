#!/usr/bin/env python3
"""模型回答质量检测与 RAG 降级。"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_EQ_KEYWORDS = (
    "地震", "震", "避险", "应急", "撤离", "安全", "余震", "建筑", "受伤",
    "救援", "演练", "防灾", "震感", "避难", "自救", "互救", "滑坡", "燃气",
)

_GARBAGE_MARKERS = (
    "UITableView", "Assistant,", "参考消息网", "蹲牙", "丹心图强", "严责历史",
    "本人企业", "震区任避", "绿豆西湖", "教学日志事", "一玉山", "一玉带",
    "Napište", "Zlepšení", "Human:", "User:",
)

_EVAL_PREFIX = re.compile(r"^参考答案[:：]?\s*")


def strip_eval_prefix(text: str) -> str:
    return _EVAL_PREFIX.sub("", (text or "").strip()).strip()


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


def guard_response(
    text: str,
    response_meta: Optional[Dict[str, Any]] = None,
    *,
    static_confidence: Optional[float] = None,
) -> str:
    """检测低质量回答，必要时降级为 RAG 检索摘要。"""
    response_meta = response_meta or {}
    cleaned = strip_eval_prefix(text or "")
    fallback = (response_meta.get("rag_fallback_text") or "").strip()
    sc = static_confidence
    if sc is None:
        sc = response_meta.get("static_confidence")

    low_conf = sc is not None and float(sc) < 0.7
    garbled = is_garbled_response(cleaned)

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
