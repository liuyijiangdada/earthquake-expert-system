#!/usr/bin/env python3
"""应急知识 JSON 分块与轻量向量检索（内存余弦 Top-K）。"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np

EmbedFn = Callable[[str], np.ndarray]


def load_emergency_chunks(path: str) -> List[Dict[str, Any]]:
    """从 emergency_knowledge.json 加载 topic 级 chunk（忽略 meta、topic_relations）。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks: List[Dict[str, Any]] = []
    for topic in data.get("topics", []):
        tid = topic.get("id") or ""
        title = topic.get("title") or ""
        category = topic.get("category") or ""
        source = topic.get("source") or ""
        def _step_order(s: Dict[str, Any]) -> int:
            o = s.get("order", 0)
            try:
                return int(float(o))
            except (TypeError, ValueError):
                return 0

        steps = sorted(topic.get("steps", []), key=_step_order)
        step_lines = [s.get("text") or "" for s in steps]
        text = f"{title}\n{category}\n" + "\n".join(step_lines)
        chunks.append(
            {
                "topic_id": tid,
                "title": title,
                "source": source,
                "text": text.strip(),
            }
        )
    return chunks


def _cosine_scores(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = np.asarray(query_vec, dtype=np.float64).ravel()
    m = np.asarray(matrix, dtype=np.float64)
    qn = np.linalg.norm(q) + 1e-12
    mn = np.linalg.norm(m, axis=1, keepdims=True) + 1e-12
    return (m @ q) / (mn.ravel() * qn)


class EmergencyRAG:
    """对固定 chunk 列表预计算向量，查询时余弦相似度 Top-K。"""

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        embed: EmbedFn,
        embed_query: Optional[EmbedFn] = None,
    ):
        self._chunks = chunks
        self._embed = embed
        self._embed_query = embed_query or embed
        self._chunk_matrix = np.stack([np.asarray(embed(c["text"]), dtype=np.float64) for c in chunks])

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        qv = np.asarray(self._embed_query(query), dtype=np.float64).ravel()
        scores = _cosine_scores(qv, self._chunk_matrix)
        k = min(top_k, len(self._chunks))
        idx = np.argsort(-scores)[:k]
        out: List[Dict[str, Any]] = []
        for i in idx:
            c = self._chunks[int(i)]
            out.append(
                {
                    "topic_id": c["topic_id"],
                    "title": c["title"],
                    "source": c["source"],
                    "text": c["text"],
                    "score": float(scores[int(i)]),
                }
            )
        return out


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_emergency_knowledge_path(config: Any) -> str:
    p = getattr(config, "EMERGENCY_KNOWLEDGE_FILE", "data/emergency_knowledge.json")
    if os.path.isabs(p) and os.path.isfile(p):
        return p
    cand = os.path.join(os.getcwd(), p)
    if os.path.isfile(cand):
        return cand
    return os.path.join(_project_root(), p)


def build_emergency_rag_from_config(config: Any) -> Optional[EmergencyRAG]:
    """加载 SentenceTransformer 与语料；失败返回 None 并记录日志。"""
    try:
        from sentence_transformers import SentenceTransformer

        path = resolve_emergency_knowledge_path(config)
        if not os.path.isfile(path):
            logging.error("Emergency knowledge file not found: %s", path)
            return None
        chunks = load_emergency_chunks(path)
        if not chunks:
            logging.error("No topics loaded from %s", path)
            return None
        model_name = getattr(config, "RAG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
        local_only = bool(getattr(config, "RAG_EMBEDDING_LOCAL_FILES_ONLY", True))
        model = SentenceTransformer(model_name, local_files_only=local_only)

        def embed(text: str) -> np.ndarray:
            v = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            return np.asarray(v, dtype=np.float32)

        return EmergencyRAG(chunks=chunks, embed=embed, embed_query=embed)
    except Exception:
        logging.exception("Failed to build emergency RAG index")
        return None
