#!/usr/bin/env python3
"""离线评测运行时：不加载大模型，组装 QueryContextBuilder（JSON 图谱回退）。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from core.dynamic_retriever import DynamicRetriever
from core.dynamic_snapshot import apply_snapshot_to_retriever
from core.phase_classifier import PhaseClassifier
from core.scheduler import Scheduler
from services.context_builder import QueryContextBuilder, QueryContextDeps

ROOT = Path(__file__).resolve().parent.parent


class JsonEvalKG:
    """无 Neo4j 时用目录 JSON + 应急主题 JSON 提供与生产查询同形的接口。"""

    def __init__(
        self,
        catalog_path: Optional[Path] = None,
        knowledge_path: Optional[Path] = None,
    ):
        cat = catalog_path or ROOT / "data/real_earthquakes_catalog.json"
        kn = knowledge_path or ROOT / "data/emergency_knowledge.json"
        self.earthquakes = json.loads(cat.read_text(encoding="utf-8")).get(
            "earthquakes", []
        )
        self.topics = json.loads(kn.read_text(encoding="utf-8")).get("topics", [])

    def query_earthquakes_by_region(self, region: str) -> List[dict]:
        if not region:
            return []
        out = []
        for eq in self.earthquakes:
            loc = str(eq.get("location") or "") + str(eq.get("region") or "")
            if region in loc:
                out.append(eq)
        return out[:3]

    def query_earthquakes_by_magnitude(self, min_mag: float) -> List[dict]:
        out = [eq for eq in self.earthquakes if float(eq.get("magnitude") or 0) >= min_mag]
        return out[:3]

    def query_emergency_context(self, user_text: str, phase_tag: str = "") -> str:
        triggers = (
            "怎么办", "如何做", "怎样", "避险", "避震", "应急", "自救", "互救",
            "余震", "室内", "室外", "高楼", "学校", "准备", "演练", "逃生",
            "多大", "在哪", "哪些", "怎么", "什么", "准备", "检查", "安置", "补贴",
        )
        if not any(t in user_text for t in triggers):
            return ""

        def _grams(t: str) -> set:
            s = "".join(re.findall(r"[\u4e00-\u9fff]+", t or ""))
            if len(s) < 2:
                return set(s)
            return {s[i : i + 2] for i in range(len(s) - 1)}

        qg = _grams(user_text)
        if not qg:
            return ""
        scored = []
        for topic in self.topics:
            title = topic.get("title") or ""
            blob = title + " " + " ".join(
                s.get("text") or "" for s in topic.get("steps") or []
            )
            tg = _grams(blob)
            if not tg:
                continue
            score = len(qg & tg) / max(len(qg), 1)
            ptag = topic.get("phase_tag") or ""
            if phase_tag and phase_tag not in ("通用", ""):
                if ptag == phase_tag:
                    score += 0.4
                elif ptag in ("通用", ""):
                    score += 0.08
            title_g = _grams(title)
            if title_g:
                score += 0.4 * len(qg & title_g) / max(len(qg), 1)
            if score > 0.02:
                scored.append((score, topic))
        if not scored:
            return ""
        scored.sort(key=lambda x: -x[0])
        lines = ["【应急主题】\n"]
        for _, topic in scored[:2]:
            lines.append(f"{topic.get('title')}\n")
            for step in (topic.get("steps") or [])[:4]:
                lines.append(f"- {step.get('text')}\n")
        return "".join(lines)


def try_build_kg(config, *, timeout_sec: float = 3.0) -> Any:
    """优先 Neo4j；连不上时回退 JSON 图谱。

    旧实现只调用 ``kg._connect()``——neo4j Driver 是惰性建连，永远不抛异常，
    于是后端被误判为 neo4j，后续每次查询都失败返回空串，KG 证据块恒为「（无）」。
    这里补一次带超时的一次性探活，失败即回退，保证离线评测也能注入图谱证据。
    """
    # 先用 socket 探活，避免 neo4j 驱动内部的重试把每次查询拖到秒级
    try:
        import socket
        from urllib.parse import urlparse

        uri = getattr(config, "NEO4J_URI", "bolt://localhost:7687")
        host = urlparse(uri).hostname or "localhost"
        port = urlparse(uri).port or 7687
        with socket.create_connection((host, port), timeout=timeout_sec):
            pass
    except Exception:
        return JsonEvalKG()

    try:
        from kg.neo4j_kg import Neo4jKG

        kg = Neo4jKG()
        driver = kg._connect()
        with driver.session() as session:
            session.run("RETURN 1 AS ok").single()
        return kg
    except Exception:
        return JsonEvalKG()


class KeywordRAG:
    """无句向量模型时的词重叠检索，保证 B2 评测仍有【参考资料】可注入。"""

    def __init__(self, chunks: List[dict]):
        self._chunks = chunks

    @classmethod
    def from_config(cls, config) -> Optional["KeywordRAG"]:
        try:
            from rag.emergency_rag import load_emergency_chunks, resolve_emergency_knowledge_path

            path = resolve_emergency_knowledge_path(config)
            chunks = load_emergency_chunks(path)
            return cls(chunks) if chunks else None
        except Exception:
            return None

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        chars = re.findall(r"[\u4e00-\u9fff]", query or "")
        q = {"".join(chars[i : i + 2]) for i in range(max(0, len(chars) - 1))}
        scored = []
        for c in self._chunks:
            blob = (c.get("title") or "") + (c.get("text") or "")
            bchars = re.findall(r"[\u4e00-\u9fff]", blob)
            t = {"".join(bchars[i : i + 2]) for i in range(max(0, len(bchars) - 1))}
            score = (len(q & t) / max(len(q), 1)) if q else 0.0
            scored.append((score, c))
        scored.sort(key=lambda x: -x[0])
        hits = []
        for score, c in scored[:top_k]:
            if score <= 0:
                continue
            hits.append(
                {
                    "topic_id": c.get("topic_id", ""),
                    "title": c.get("title", ""),
                    "source": c.get("source", ""),
                    "text": c.get("text", ""),
                    "score": float(score),
                }
            )
        return hits


def try_build_rag(config):
    try:
        from rag.emergency_rag import build_emergency_rag_from_config

        rag = build_emergency_rag_from_config(config)
        if rag is not None:
            return rag
    except Exception:
        pass
    return KeywordRAG.from_config(config)


@dataclass
class EvalStack:
    config: Any
    builder: QueryContextBuilder
    classifier: PhaseClassifier
    scheduler: Scheduler
    retriever: DynamicRetriever
    kg_backend: str
    rag_backend: str


def build_eval_stack(
    config,
    *,
    snapshot_path: Optional[Path] = None,
    force_json_kg: bool = False,
    skip_rag: bool = False,
) -> EvalStack:
    kg = JsonEvalKG() if force_json_kg else try_build_kg(config)
    kg_backend = "json" if isinstance(kg, JsonEvalKG) else "neo4j"
    rag = None if skip_rag else try_build_rag(config)
    rag_backend = "none"
    if rag is not None:
        rag_backend = "keyword" if isinstance(rag, KeywordRAG) else "embedding"
    classifier = PhaseClassifier()
    scheduler = Scheduler(config)
    retriever = DynamicRetriever(config)
    if snapshot_path:
        apply_snapshot_to_retriever(retriever, snapshot_path)
    builder = QueryContextBuilder(
        QueryContextDeps(
            config=config,
            kg=kg,
            emergency_rag=rag,
            phase_classifier=classifier,
            scheduler=scheduler,
            dynamic_retriever=retriever,
            multimodal_output=None,
        )
    )
    return EvalStack(
        config=config,
        builder=builder,
        classifier=classifier,
        scheduler=scheduler,
        retriever=retriever,
        kg_backend=kg_backend,
        rag_backend=rag_backend,
    )


def apply_baseline_to_stack(stack: EvalStack, spec: "BaselineSpec") -> None:
    cfg = stack.config
    cfg.KG_CONTEXT_ENABLED = spec.kg
    cfg.RAG_ENABLED = spec.rag
    cfg.DYNAMIC_RETRIEVAL_ENABLED = spec.dynamic
    stack.retriever._enabled = spec.dynamic
    if spec.use_scheduler:
        stack.builder._deps.phase_classifier = stack.classifier
        stack.builder._deps.scheduler = stack.scheduler
    else:
        stack.builder._deps.phase_classifier = None
        stack.builder._deps.scheduler = None


@dataclass
class BaselineSpec:
    id: str
    kg: bool
    rag: bool
    dynamic: bool
    use_scheduler: bool


BASELINE_SPECS: List[BaselineSpec] = [
    BaselineSpec("B0", False, False, False, True),
    BaselineSpec("B1", True, False, False, True),
    BaselineSpec("B2", False, True, False, True),
    BaselineSpec("B3", True, True, False, True),
    BaselineSpec("B3-ns", True, True, False, False),
    BaselineSpec("B4", False, False, True, True),
    BaselineSpec("B5", True, True, True, True),
]


def spec_by_id(bid: str) -> BaselineSpec:
    for s in BASELINE_SPECS:
        if s.id == bid:
            return s
    raise KeyError(f"未知基线: {bid}")


def decision_from_meta(meta: dict) -> dict:
    return {
        "phase": meta.get("phase"),
        "phase_confidence": meta.get("phase_confidence"),
        "urgency": meta.get("urgency"),
        "need_dynamic": meta.get("need_dynamic"),
        "static_confidence": meta.get("static_confidence"),
        "dynamic_availability": meta.get("dynamic_availability"),
        "use_dynamic": int(meta.get("dynamic_items_count") or 0) > 0
        or bool(meta.get("dynamic_source")),
        "dynamic_source": meta.get("dynamic_source") or "",
        "dynamic_items_count": int(meta.get("dynamic_items_count") or 0),
        "reasoning": meta.get("schedule_reasoning") or "",
        "reliability_hint": meta.get("reliability_hint") or "",
        "rag_topic_ids": meta.get("rag_topic_ids") or [],
    }


def model_snapshot(config) -> dict:
    lora = getattr(config, "FINETUNED_MODEL_PATH", "")
    lora_path = Path(lora) if lora else None
    adapter = False
    if lora_path:
        p = lora_path if lora_path.is_absolute() else ROOT / lora_path
        adapter = (p / "adapter_config.json").is_file()
    return {
        "MODEL_NAME": getattr(config, "MODEL_NAME", ""),
        "FINETUNED_MODEL_PATH": lora,
        "lora_adapter_present": adapter,
        "LLM_MAX_NEW_TOKENS": getattr(config, "LLM_MAX_NEW_TOKENS", None),
        "LLM_DO_SAMPLE": getattr(config, "LLM_DO_SAMPLE", None),
        "LLM_TEMPERATURE": getattr(config, "LLM_TEMPERATURE", None),
    }
