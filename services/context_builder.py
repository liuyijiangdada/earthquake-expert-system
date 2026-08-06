#!/usr/bin/env python3
"""组装 KG / RAG / 动态检索与阶段调度上下文，供文本 LLM 与 Qwen-VL 共用。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from config.constants import match_region_in_text
from core.knowledge_signals import compute_knowledge_signals
from services.response_guard import build_rag_fallback_text


@dataclass
class QueryContextDeps:
    config: Any
    kg: Any
    emergency_rag: Any = None
    phase_classifier: Any = None
    scheduler: Any = None
    dynamic_retriever: Any = None
    multimodal_output: Any = None


@dataclass
class ContextSections:
    kg_section: str
    rag_section: str
    rag_hits: list
    dynamic_section: str
    phase_instruction: str
    validity_hint: str


class QueryContextBuilder:
    def __init__(self, deps: QueryContextDeps):
        self._deps = deps

    @property
    def config(self):
        return self._deps.config

    def normalize_history(self, raw, max_rounds: int = None) -> list:
        """解析 [{role, content}, ...]，保留最近 max_rounds 轮。"""
        cfg = self.config
        if max_rounds is None:
            max_rounds = int(getattr(cfg, "CHAT_HISTORY_MAX_ROUNDS", 3))
        max_chars = int(getattr(cfg, "CHAT_HISTORY_MAX_CHARS_PER_MSG", 500))
        if not raw or not isinstance(raw, list):
            return []

        out = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = (item.get("content") or "").strip()
            if role not in ("user", "assistant") or not content:
                continue
            out.append({"role": role, "content": content[:max_chars]})

        cap = max(0, max_rounds) * 2
        return out[-cap:] if cap else []

    def build_rag_section(
        self,
        input_text: str,
        *,
        precomputed_hits: Optional[list] = None,
    ) -> Tuple[str, list]:
        cfg = self.config
        if not getattr(cfg, "RAG_ENABLED", True):
            return "（本路径已关闭）", []
        hits = precomputed_hits
        if hits is None:
            rag = self._deps.emergency_rag
            if rag is None:
                return "（无相关条目）", []
            top_k = getattr(cfg, "RAG_TOP_K", 5)
            hits = rag.search(input_text, top_k=top_k)
        if not hits:
            return "（无相关条目）", []
        max_chars = getattr(cfg, "RAG_MAX_CHUNK_CHARS", 800)
        lines = []
        for i, h in enumerate(hits, 1):
            body = (h.get("text") or "")[:max_chars]
            lines.append(f"{i}. [{h.get('topic_id', '')}] {h.get('title', '')}\n{body}")
        return "\n".join(lines), hits

    def _format_history_section(self, history: list) -> str:
        if not history:
            return ""
        lines = ["【最近对话（供延续上下文，仅供参考）】"]
        for i, h in enumerate(history, 1):
            label = "用户" if h["role"] == "user" else "助手"
            lines.append(f"{i}. {label}：{h['content']}")
        lines.append("")
        return "\n".join(lines)

    def _build_kg_context(
        self,
        input_text: str,
        phase_tag: str,
        use_kg: bool,
        knowledge_signals=None,
    ) -> str:
        if not use_kg or not getattr(self.config, "KG_CONTEXT_ENABLED", True):
            return ""

        kg = self._deps.kg
        kg_context = ""

        if knowledge_signals and knowledge_signals.kg_region_snippet:
            kg_context += knowledge_signals.kg_region_snippet
        else:
            matched_region = match_region_in_text(input_text)
            if matched_region:
                region_results = kg.query_earthquakes_by_region(matched_region)
                if region_results:
                    kg_context += "【知识图谱信息】\n"
                    for i, eq in enumerate(region_results[:3]):
                        kg_context += f"{i+1}. {eq['location']}地震：\n"
                        kg_context += f"   时间：{eq['time']}\n"
                        kg_context += f"   震级：{eq['magnitude']}级\n"
                        kg_context += f"   深度：{eq['depth']}公里\n"
                        kg_context += f"   烈度：{eq['intensity']}\n"
                        kg_context += f"   描述：{eq['description']}\n\n"

        if "震级" in input_text and ("大于" in input_text or "高于" in input_text):
            match = re.search(r"(大于|高于)(\d+\.?\d*)", input_text)
            if match:
                min_mag = float(match.group(2))
                mag_results = kg.query_earthquakes_by_magnitude(min_mag)
                if mag_results:
                    kg_context += f"【震级大于{min_mag}级的地震】\n"
                    for i, eq in enumerate(mag_results[:3]):
                        kg_context += (
                            f"{i+1}. {eq['location']}：{eq['magnitude']}级 ({eq['time']})\n"
                        )
                    kg_context += "\n"

        if knowledge_signals and knowledge_signals.kg_emergency_snippet:
            kg_context += knowledge_signals.kg_emergency_snippet
        else:
            emg = kg.query_emergency_context(input_text, phase_tag=phase_tag)
            if emg:
                kg_context += emg

        return kg_context

    def init_response_meta(
        self,
        *,
        for_vision: bool,
        normalized_history: list,
        user_query: str = "",
    ) -> dict:
        cfg = self.config
        return {
            "kg_enabled": bool(getattr(cfg, "KG_CONTEXT_ENABLED", True)),
            "rag_enabled": bool(getattr(cfg, "RAG_ENABLED", True)),
            "rag_topic_ids": [],
            "phase": "通用",
            "phase_confidence": 0.0,
            "urgency": 0.0,
            "need_dynamic": False,
            "schedule_reasoning": "",
            "multimodal_backend": "qwen_vl" if for_vision else "text_llm",
            "history_rounds": len(normalized_history) // 2,
            "history_messages": len(normalized_history),
            "user_query": user_query or "",
        }

    def step_classify(self, input_text: str, response_meta: dict):
        """阶段分类，更新 response_meta，返回 (phase_result, phase_tag)。"""
        deps = self._deps
        phase_result = None
        if deps.phase_classifier:
            phase_result = deps.phase_classifier.classify(input_text)
            response_meta["phase"] = phase_result.phase.value
            response_meta["phase_confidence"] = round(phase_result.confidence, 2)
            response_meta["urgency"] = round(phase_result.urgency, 2)
            response_meta["need_dynamic"] = phase_result.need_dynamic
        phase_tag = phase_result.phase.value if phase_result else ""
        return phase_result, phase_tag

    def step_compute_signals(
        self,
        input_text: str,
        phase_tag: str,
        phase_result,
        response_meta: dict,
    ):
        if not phase_result:
            return None
        deps = self._deps
        knowledge_signals = compute_knowledge_signals(
            input_text,
            phase_tag,
            phase_result,
            config=self.config,
            kg=deps.kg,
            emergency_rag=deps.emergency_rag,
            dynamic_retriever=deps.dynamic_retriever,
        )
        response_meta["static_confidence"] = round(knowledge_signals.static_confidence, 2)
        response_meta["dynamic_availability"] = round(
            knowledge_signals.dynamic_availability, 2
        )
        if knowledge_signals.temporal_validity:
            response_meta["temporal_validity"] = knowledge_signals.temporal_validity
        return knowledge_signals

    def step_schedule(self, phase_result, knowledge_signals, response_meta: dict):
        deps = self._deps
        if not deps.scheduler or not phase_result:
            return None
        schedule_decision = deps.scheduler.decide(phase_result, knowledge_signals)
        response_meta["schedule_reasoning"] = schedule_decision.reasoning
        if schedule_decision.reliability_hint:
            response_meta["reliability_hint"] = schedule_decision.reliability_hint
        return schedule_decision

    def step_retrieve_sections(
        self,
        input_text: str,
        phase_tag: str,
        schedule_decision,
        knowledge_signals,
        response_meta: dict,
    ) -> ContextSections:
        cfg = self.config
        deps = self._deps

        use_kg = schedule_decision.use_kg if schedule_decision else True
        kg_context = self._build_kg_context(
            input_text, phase_tag, use_kg, knowledge_signals
        )
        if getattr(cfg, "KG_CONTEXT_ENABLED", True):
            kg_section = kg_context.strip() if kg_context.strip() else "（无）"
        else:
            kg_section = "（本路径已关闭）"

        use_rag = schedule_decision.use_rag if schedule_decision else True
        pre_rag_hits = knowledge_signals.rag_hits if knowledge_signals else None
        if use_rag:
            rag_section, rag_hits = self.build_rag_section(
                input_text, precomputed_hits=pre_rag_hits
            )
        else:
            rag_section = "（调度器判定本路径无需启用）"
            rag_hits = []
        response_meta["rag_topic_ids"] = [h.get("topic_id", "") for h in rag_hits]
        if rag_hits:
            response_meta["rag_fallback_text"] = build_rag_fallback_text(rag_hits)

        dynamic_section = ""
        use_dynamic = schedule_decision.use_dynamic if schedule_decision else False
        dynamic_retriever = deps.dynamic_retriever
        if use_dynamic and dynamic_retriever and dynamic_retriever.enabled:
            dynamic_result = None
            if knowledge_signals and knowledge_signals.dynamic_result is not None:
                dynamic_result = knowledge_signals.dynamic_result
            else:
                dynamic_result = dynamic_retriever.fetch_for_phase(phase_tag, input_text)
            dynamic_section = dynamic_result.to_context_text()
            response_meta["dynamic_source"] = dynamic_result.source
            response_meta["dynamic_items_count"] = len(dynamic_result.items)
            response_meta["dynamic_items"] = list(dynamic_result.items[:5])
            response_meta["dynamic_fetched_at"] = dynamic_result.fetched_at or ""

        phase_instruction = ""
        if schedule_decision and schedule_decision.prompt_suffix:
            phase_instruction = schedule_decision.prompt_suffix + "\n"

        validity_hint = ""
        if (
            getattr(cfg, "VALIDITY_HINT_ENABLED", True)
            and schedule_decision
            and schedule_decision.validity_hint
        ):
            fetched_at = ""
            if dynamic_retriever and dynamic_retriever._cache:
                fetched_at = dynamic_retriever._cache.fetched_at
            validity_hint = schedule_decision.validity_hint.format(
                fetched_at=fetched_at or "刚刚",
                refresh_minutes=10,
            )

        return ContextSections(
            kg_section=kg_section,
            rag_section=rag_section,
            rag_hits=rag_hits,
            dynamic_section=dynamic_section,
            phase_instruction=phase_instruction,
            validity_hint=validity_hint,
        )

    def step_build_prompt(
        self,
        input_text: str,
        *,
        for_vision: bool,
        normalized_history: list,
        phase_tag: str,
        sections: ContextSections,
    ) -> str:
        deps = self._deps
        cfg = self.config
        prompt = (
            "你是地震应急问答专家。请结合【知识图谱】【参考资料】与【动态信息】作答；"
            "语气肯定、结论明确，直接回答用户问题。\n\n"
        )

        if phase_tag and phase_tag != "通用":
            prompt += f"当前判定为【{phase_tag}阶段】的问题。\n\n"

        prompt += f"【知识图谱】\n{sections.kg_section}\n\n"
        prompt += f"【参考资料】\n{sections.rag_section}\n\n"

        if sections.dynamic_section:
            prompt += f"{sections.dynamic_section}\n\n"

        if getattr(cfg, "MULTIMODAL_INJECT_PROMPT", False) and deps.multimodal_output:
            media_section = deps.multimodal_output.build_media_section(input_text, phase_tag)
            if media_section:
                prompt += media_section + "\n"

        prompt += (
            "规则：震级、时间、地点等数值事实优先采信【动态信息】速报字段，其次采信【知识图谱】；"
            "【参考资料】仅补充避险步骤与表述。"
            "若【动态信息】已给出震级/地点/时间，必须直接、肯定地写出这些参数（可注明来源为台网或国际目录速报），"
            "禁止使用“无法准确确定”“尚未得到权威确认”“具体震级需等待正式报告”等犹豫句式。"
            "仅当动态与图谱均无相关条目时，才说明暂无匹配速报，并给出如何查询官方渠道的明确指引。\n"
        )

        if sections.phase_instruction:
            prompt += f"{sections.phase_instruction}\n"

        if for_vision:
            prompt += (
                "回答要求：\n"
                "1. 请直接观察用户上传的图片，结合上述知识上下文作答\n"
                "2. 优先采用知识图谱中的可验证事实，合理利用参考资料与动态信息\n"
                "3. 回答简洁明了，不要使用强调符号如***\n"
                "4. 若图片与地震应急无关，请说明并引导用户上传相关图片\n"
            )
        else:
            prompt += (
                "回答要求：\n"
                "1. 直接回答问题，不要有任何引言或开场白\n"
                "2. 有动态速报或图谱事实时，先给出明确结论（如震级、震中），再补一句来源或避险要点\n"
                "3. 语气肯定、表述干脆，避免冗长与推诿\n"
                "4. 不要使用任何强调符号如***\n"
                "5. 禁止输出“无法确定”“尚未确认”“仅供参考”等削弱结论的措辞\n"
            )

        if sections.validity_hint:
            prompt += f"6. 在回答末尾附上时效提示：{sections.validity_hint}\n"

        history_section = self._format_history_section(normalized_history)
        if history_section:
            prompt += f"\n{history_section}"

        if for_vision:
            prompt += (
                "【图片问答】请结合图片内容回答；若与上文对话相关请保持连贯。\n"
                f"【用户问题】\n{input_text}\n"
            )
        else:
            prompt += f"【问题】\n{input_text}\n"

        return prompt

    def prepare_context_pipeline(
        self,
        input_text: str,
        *,
        for_vision: bool = False,
        history: Optional[List] = None,
    ) -> Tuple[str, dict, str]:
        """分步组装上下文，供 LangGraph 与 prepare 共用。返回 (prompt, response_meta, phase_tag)。"""
        normalized_history = self.normalize_history(history)
        response_meta = self.init_response_meta(
            for_vision=for_vision,
            normalized_history=normalized_history,
            user_query=input_text,
        )
        phase_result, phase_tag = self.step_classify(input_text, response_meta)
        knowledge_signals = self.step_compute_signals(
            input_text, phase_tag, phase_result, response_meta
        )
        schedule_decision = self.step_schedule(
            phase_result, knowledge_signals, response_meta
        )
        sections = self.step_retrieve_sections(
            input_text, phase_tag, schedule_decision, knowledge_signals, response_meta
        )
        prompt = self.step_build_prompt(
            input_text,
            for_vision=for_vision,
            normalized_history=normalized_history,
            phase_tag=phase_tag,
            sections=sections,
        )
        return prompt, response_meta, phase_tag

    def prepare(
        self,
        input_text: str,
        *,
        for_vision: bool = False,
        history: Optional[List] = None,
    ) -> Tuple[str, dict, str]:
        """返回 (prompt, response_meta, phase_tag)。"""
        return self.prepare_context_pipeline(
            input_text, for_vision=for_vision, history=history
        )
