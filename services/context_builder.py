#!/usr/bin/env python3
"""组装 KG / RAG / 动态检索与阶段调度上下文，供文本 LLM 与 Qwen-VL 共用。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from config.constants import match_region_in_text
from core.knowledge_signals import compute_knowledge_signals


@dataclass
class QueryContextDeps:
    config: Any
    kg: Any
    emergency_rag: Any = None
    phase_classifier: Any = None
    scheduler: Any = None
    dynamic_retriever: Any = None
    multimodal_output: Any = None


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

    def prepare(
        self,
        input_text: str,
        *,
        for_vision: bool = False,
        history: Optional[List] = None,
    ) -> Tuple[str, dict, str]:
        """返回 (prompt, debug_meta, phase_tag)。"""
        cfg = self.config
        deps = self._deps
        kg_context = ""
        normalized_history = self.normalize_history(history)
        debug_meta = {
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
        }

        phase_result = None
        schedule_decision = None
        knowledge_signals = None
        if deps.phase_classifier:
            phase_result = deps.phase_classifier.classify(input_text)
            debug_meta["phase"] = phase_result.phase.value
            debug_meta["phase_confidence"] = round(phase_result.confidence, 2)
            debug_meta["urgency"] = round(phase_result.urgency, 2)
            debug_meta["need_dynamic"] = phase_result.need_dynamic

        phase_tag = phase_result.phase.value if phase_result else ""

        if phase_result:
            knowledge_signals = compute_knowledge_signals(
                input_text,
                phase_tag,
                phase_result,
                config=cfg,
                kg=deps.kg,
                emergency_rag=deps.emergency_rag,
                dynamic_retriever=deps.dynamic_retriever,
            )
            debug_meta["static_confidence"] = round(
                knowledge_signals.static_confidence, 2
            )
            debug_meta["dynamic_availability"] = round(
                knowledge_signals.dynamic_availability, 2
            )
            if knowledge_signals.temporal_validity:
                debug_meta["temporal_validity"] = knowledge_signals.temporal_validity

        if deps.scheduler and phase_result:
            schedule_decision = deps.scheduler.decide(phase_result, knowledge_signals)
            debug_meta["schedule_reasoning"] = schedule_decision.reasoning
            if schedule_decision.reliability_hint:
                debug_meta["reliability_hint"] = schedule_decision.reliability_hint

        use_kg = schedule_decision.use_kg if schedule_decision else True
        kg_context = self._build_kg_context(
            input_text, phase_tag, use_kg, knowledge_signals
        )

        if getattr(cfg, "KG_CONTEXT_ENABLED", True):
            kg_section = kg_context.strip() if kg_context.strip() else "（无）"
        else:
            kg_section = "（本路径已关闭）"

        use_rag = schedule_decision.use_rag if schedule_decision else True
        pre_rag_hits = (
            knowledge_signals.rag_hits if knowledge_signals else None
        )
        if use_rag:
            rag_section, rag_hits = self.build_rag_section(
                input_text, precomputed_hits=pre_rag_hits
            )
        else:
            rag_section = "（调度器判定本路径无需启用）"
            rag_hits = []
        debug_meta["rag_topic_ids"] = [h.get("topic_id", "") for h in rag_hits]

        dynamic_section = ""
        use_dynamic = schedule_decision.use_dynamic if schedule_decision else False
        dynamic_retriever = deps.dynamic_retriever
        if use_dynamic and dynamic_retriever and dynamic_retriever.enabled:
            dynamic_result = None
            if knowledge_signals and knowledge_signals.dynamic_result is not None:
                dynamic_result = knowledge_signals.dynamic_result
            else:
                dynamic_result = dynamic_retriever.fetch_for_phase(
                    phase_tag, input_text
                )
            dynamic_section = dynamic_result.to_context_text()
            debug_meta["dynamic_source"] = dynamic_result.source
            debug_meta["dynamic_items_count"] = len(dynamic_result.items)

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

        prompt = "你是一个地震知识专家。请结合【知识图谱】与【参考资料】回答问题。\n\n"

        if phase_tag and phase_tag != "通用":
            prompt += f"当前判定为【{phase_tag}阶段】的问题。\n\n"

        prompt += f"【知识图谱】\n{kg_section}\n\n"
        prompt += f"【参考资料】\n{rag_section}\n\n"

        if dynamic_section:
            prompt += f"{dynamic_section}\n\n"

        if getattr(cfg, "MULTIMODAL_INJECT_PROMPT", False) and deps.multimodal_output:
            media_section = deps.multimodal_output.build_media_section(input_text, phase_tag)
            if media_section:
                prompt += media_section + "\n"

        prompt += (
            "规则：数值、时间、震级、地点等可验证事实以知识图谱为准；参考资料仅作步骤与表述补充；"
            "动态信息为实时数据，可能随时更新。"
            "若三者均未提供有效条目，可基于常识回答，并简要说明未命中本地知识库。\n"
        )

        if phase_instruction:
            prompt += f"{phase_instruction}\n"

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
                "2. 优先采用知识图谱中的可验证事实，合理利用参考资料与动态信息\n"
                "3. 回答要简洁明了，避免冗长\n"
                "4. 不要使用任何强调符号如***\n"
                "5. 如果知识图谱与参考资料均未提供相关信息，请基于你的知识提供合理回答，并说明未命中本地知识库\n"
            )

        if validity_hint:
            prompt += f"6. 在回答末尾附上时效提示：{validity_hint}\n"

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

        return prompt, debug_meta, phase_tag
