#!/usr/bin/env python3
"""文本问答 LangGraph 编排：分类 → 信号 → 调度 → 检索 → 拼 prompt → 生成 → 校验。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from services.context_builder import ContextSections, QueryContextBuilder

logger = logging.getLogger(__name__)


class QueryWorkflowState(TypedDict, total=False):
    input_text: str
    history: Optional[List]
    for_vision: bool
    normalized_history: List
    response_meta: dict
    phase_result: Any
    phase_tag: str
    knowledge_signals: Any
    schedule_decision: Any
    sections: ContextSections
    prompt: str
    raw_response: str
    response: str
    error: Optional[str]
    skip_generate: bool


@dataclass
class QueryWorkflowDeps:
    context_builder: QueryContextBuilder
    apply_layer3: Callable[[str, str, str, dict], str]
    run_llm: Callable[[str], str]
    sanitize: Callable[[str, str], str]
    guard: Callable[[str, dict], str]
    finalize_media: Callable[[dict, str, str], None]


def _normalize_node(state: QueryWorkflowState, builder: QueryContextBuilder) -> dict:
    history = state.get("history")
    normalized = builder.normalize_history(history)
    for_vision = bool(state.get("for_vision", False))
    return {
        "normalized_history": normalized,
        "response_meta": builder.init_response_meta(
            for_vision=for_vision,
            normalized_history=normalized,
            user_query=state.get("input_text") or "",
        ),
    }


def _classify_node(state: QueryWorkflowState, builder: QueryContextBuilder) -> dict:
    response_meta = dict(state["response_meta"])
    phase_result, phase_tag = builder.step_classify(
        state["input_text"],
        response_meta,
        history=state.get("normalized_history"),
    )
    return {
        "response_meta": response_meta,
        "phase_result": phase_result,
        "phase_tag": phase_tag,
    }


def _signals_node(state: QueryWorkflowState, builder: QueryContextBuilder) -> dict:
    response_meta = dict(state["response_meta"])
    knowledge_signals = builder.step_compute_signals(
        state["input_text"],
        state.get("phase_tag", ""),
        state.get("phase_result"),
        response_meta,
    )
    return {"response_meta": response_meta, "knowledge_signals": knowledge_signals}


def _schedule_node(state: QueryWorkflowState, builder: QueryContextBuilder) -> dict:
    response_meta = dict(state["response_meta"])
    schedule_decision = builder.step_schedule(
        state.get("phase_result"),
        state.get("knowledge_signals"),
        response_meta,
    )
    return {"response_meta": response_meta, "schedule_decision": schedule_decision}


def _retrieve_node(state: QueryWorkflowState, builder: QueryContextBuilder) -> dict:
    response_meta = dict(state["response_meta"])
    sections = builder.step_retrieve_sections(
        state["input_text"],
        state.get("phase_tag", ""),
        state.get("schedule_decision"),
        state.get("knowledge_signals"),
        response_meta,
    )
    return {"response_meta": response_meta, "sections": sections}


def _build_prompt_node(state: QueryWorkflowState, builder: QueryContextBuilder) -> dict:
    prompt = builder.step_build_prompt(
        state["input_text"],
        for_vision=bool(state.get("for_vision", False)),
        normalized_history=state.get("normalized_history") or [],
        phase_tag=state.get("phase_tag", ""),
        sections=state["sections"],
    )
    return {"prompt": prompt}


def _apply_layer3_node(state: QueryWorkflowState, deps: QueryWorkflowDeps) -> dict:
    response_meta = dict(state["response_meta"])
    prompt = deps.apply_layer3(
        state["prompt"],
        state["input_text"],
        state.get("phase_tag", ""),
        response_meta,
    )
    return {"prompt": prompt, "response_meta": response_meta}


def _generate_node(state: QueryWorkflowState, deps: QueryWorkflowDeps) -> dict:
    if state.get("skip_generate"):
        return {}
    # 震级/震中事实问句：有动态速报则直接肯定作答，跳过小模型推诿
    try:
        from services.response_guard import maybe_confident_quake_answer

        direct = maybe_confident_quake_answer(
            state.get("response_meta") or {},
            user_query=state.get("input_text") or "",
        )
        if direct:
            meta = dict(state.get("response_meta") or {})
            meta["response_fallback"] = "dynamic_confident"
            meta["response_quality"] = "direct_dynamic"
            return {"raw_response": direct, "error": None, "response_meta": meta}
    except Exception:
        logger.exception("动态肯定回答短路失败，回退 LLM")

    try:
        raw_response = deps.run_llm(state["prompt"])
        return {"raw_response": raw_response, "error": None}
    except Exception as exc:
        logger.exception("LangGraph 生成节点失败")
        return {"error": str(exc), "raw_response": ""}


def _postprocess_node(state: QueryWorkflowState, deps: QueryWorkflowDeps) -> dict:
    if state.get("skip_generate"):
        return {"response": state.get("prompt", "")}

    response_meta = dict(state["response_meta"])
    text = state.get("raw_response") or ""
    text = deps.sanitize(text, state["input_text"])
    text = deps.guard(text, response_meta)
    return {"response": text, "response_meta": response_meta}


def _enrich_node(state: QueryWorkflowState, deps: QueryWorkflowDeps) -> dict:
    response_meta = dict(state["response_meta"])
    deps.finalize_media(response_meta, state["input_text"], state.get("phase_tag", ""))
    return {"response_meta": response_meta}


def build_query_workflow(deps: QueryWorkflowDeps, *, include_generate: bool = True):
    """编译 LangGraph；include_generate=False 时仅跑上下文编排（供多模态复用）。"""
    builder = deps.context_builder
    graph = StateGraph(QueryWorkflowState)

    graph.add_node("normalize", lambda s: _normalize_node(s, builder))
    graph.add_node("classify", lambda s: _classify_node(s, builder))
    graph.add_node("signals", lambda s: _signals_node(s, builder))
    graph.add_node("schedule", lambda s: _schedule_node(s, builder))
    graph.add_node("retrieve", lambda s: _retrieve_node(s, builder))
    graph.add_node("build_prompt", lambda s: _build_prompt_node(s, builder))
    graph.add_node("apply_layer3", lambda s: _apply_layer3_node(s, deps))

    graph.add_edge(START, "normalize")
    graph.add_edge("normalize", "classify")
    graph.add_edge("classify", "signals")
    graph.add_edge("signals", "schedule")
    graph.add_edge("schedule", "retrieve")
    graph.add_edge("retrieve", "build_prompt")
    graph.add_edge("build_prompt", "apply_layer3")

    if include_generate:
        graph.add_node("generate", lambda s: _generate_node(s, deps))
        graph.add_node("postprocess", lambda s: _postprocess_node(s, deps))
        graph.add_node("enrich", lambda s: _enrich_node(s, deps))
        graph.add_edge("apply_layer3", "generate")
        graph.add_edge("generate", "postprocess")
        graph.add_edge("postprocess", "enrich")
        graph.add_edge("enrich", END)
    else:
        graph.add_edge("apply_layer3", END)

    return graph.compile()


def run_context_workflow(
    deps: QueryWorkflowDeps,
    input_text: str,
    *,
    history: Optional[List] = None,
    for_vision: bool = False,
) -> tuple[str, dict, str]:
    """仅上下文编排：返回 (prompt, response_meta, phase_tag)。"""
    workflow = build_query_workflow(deps, include_generate=False)
    result = workflow.invoke(
        {
            "input_text": input_text,
            "history": history,
            "for_vision": for_vision,
            "skip_generate": True,
        }
    )
    return result["prompt"], result["response_meta"], result.get("phase_tag", "")


def run_text_query_workflow(
    deps: QueryWorkflowDeps,
    input_text: str,
    *,
    history: Optional[List] = None,
) -> tuple[str, dict]:
    """完整文本问答编排：返回 (response, response_meta)。"""
    workflow = build_query_workflow(deps, include_generate=True)
    result = workflow.invoke(
        {
            "input_text": input_text,
            "history": history,
            "for_vision": False,
            "skip_generate": False,
        }
    )
    if result.get("error") and not (result.get("response") or "").strip():
        raise RuntimeError(result["error"])
    return result.get("response", ""), result.get("response_meta") or {}
