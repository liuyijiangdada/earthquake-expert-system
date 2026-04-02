"""应急知识 RAG：向量存 Milvus，检索 IP（与 L2 归一化后的余弦等价）。"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

EmbedFn = Callable[[str], np.ndarray]

# 默认与 BAAI/bge-small-zh-v1.5 一致；实际以首条向量维度为准
_DEFAULT_EMBED_DIM = 512

# Milvus VARCHAR 上限内留余量
_MAX_TITLE = 1024
_MAX_SOURCE = 512
_MAX_TEXT = 16384


def _connect(host: str, port: Any) -> None:
    port_int = int(port) if not isinstance(port, int) else port
    alias = "default"
    if connections.has_connection(alias):
        return
    connections.connect(alias=alias, host=host, port=port_int, timeout=30)


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _entity_str(ent: Any, key: str) -> str:
    if ent is None:
        return ""
    v = ent.get(key) if hasattr(ent, "get") else getattr(ent, key, "")
    if v is None:
        return ""
    return str(v)


class EmergencyRAGMilvus:
    """与 EmergencyRAG 相同的 search 接口，底层为 Milvus。"""

    def __init__(
        self,
        collection: Collection,
        embed_query: EmbedFn,
        embed_dim: int,
    ):
        self._collection = collection
        self._embed_query = embed_query
        self._embed_dim = embed_dim

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        qv = np.asarray(self._embed_query(query), dtype=np.float32).ravel()
        if qv.shape[0] != self._embed_dim:
            logging.error(
                "Query embedding dim %s != collection dim %s (check RAG_EMBEDDING_MODEL)",
                qv.shape[0],
                self._embed_dim,
            )
            return []
        k = min(top_k, 16384)
        results = self._collection.search(
            data=[qv.tolist()],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {}},
            limit=k,
            output_fields=["topic_id", "title", "source", "text"],
        )
        out: List[Dict[str, Any]] = []
        for hit in results[0]:
            ent = hit.entity
            out.append(
                {
                    "topic_id": _entity_str(ent, "topic_id"),
                    "title": _entity_str(ent, "title"),
                    "source": _entity_str(ent, "source"),
                    "text": _entity_str(ent, "text"),
                    "score": float(hit.distance),
                }
            )
        return out


def _schema(embed_dim: int) -> CollectionSchema:
    fields = [
        FieldSchema(
            name="topic_id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            max_length=256,
            auto_id=False,
        ),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embed_dim),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=_MAX_TITLE),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=_MAX_SOURCE),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=_MAX_TEXT),
    ]
    return CollectionSchema(fields=fields, description="Earthquake emergency RAG chunks")


def _infer_dim_from_collection(col: Collection) -> int:
    for f in col.schema.fields:
        if f.name == "embedding" and f.dtype == DataType.FLOAT_VECTOR:
            params = getattr(f, "params", None) or {}
            d = params.get("dim")
            return int(d) if d is not None else 0
    return 0


def build_milvus_emergency_rag(
    chunks: List[Dict[str, Any]],
    embed: EmbedFn,
    embed_query: Optional[EmbedFn],
    config: Any,
) -> Optional[EmergencyRAGMilvus]:
    """创建/重建集合并写入向量；失败返回 None。"""
    host = getattr(config, "MILVUS_HOST", "localhost")
    port = getattr(config, "MILVUS_PORT", 19530)
    name = getattr(config, "RAG_MILVUS_COLLECTION", "emergency_rag")
    rebuild = bool(getattr(config, "RAG_MILVUS_REBUILD_ON_START", True))

    eq = embed_query or embed

    try:
        _connect(host, port)
    except Exception:
        logging.exception("Milvus connect failed (%s:%s)", host, port)
        return None

    try:
        if utility.has_collection(name):
            if rebuild:
                utility.drop_collection(name)
            else:
                col = Collection(name)
                col.load()
                dim = _infer_dim_from_collection(col)
                if dim <= 0:
                    dim = _DEFAULT_EMBED_DIM
                return EmergencyRAGMilvus(col, eq, dim)

        sample = np.asarray(embed((chunks[0].get("text") or "")), dtype=np.float32).ravel()
        embed_dim = int(sample.shape[0])
        if embed_dim <= 0:
            logging.error("Invalid embedding dimension")
            return None

        col = Collection(name=name, schema=_schema(embed_dim))

        ids: List[str] = []
        vectors: List[List[float]] = []
        titles: List[str] = []
        sources: List[str] = []
        texts: List[str] = []

        for i, c in enumerate(chunks):
            tid = (c.get("topic_id") or "").strip() or f"_row_{i}"
            text = c.get("text") or ""
            vec = np.asarray(embed(text), dtype=np.float32).ravel()
            if vec.shape[0] != embed_dim:
                logging.error("Chunk embedding dim mismatch for topic_id=%s", tid)
                utility.drop_collection(name)
                return None
            ids.append(tid)
            vectors.append(vec.tolist())
            titles.append(_truncate(c.get("title") or "", _MAX_TITLE))
            sources.append(_truncate(c.get("source") or "", _MAX_SOURCE))
            texts.append(_truncate(text, _MAX_TEXT))

        if not ids:
            logging.error("No chunks to insert into Milvus")
            utility.drop_collection(name)
            return None

        col.insert([ids, vectors, titles, sources, texts])
        col.flush()

        col.create_index(
            field_name="embedding",
            index_params={"metric_type": "IP", "index_type": "FLAT"},
        )
        col.load()
        return EmergencyRAGMilvus(col, eq, embed_dim)
    except Exception:
        logging.exception("Milvus RAG build failed")
        return None
