#!/usr/bin/env python3
"""导出 Milvus 中应急 RAG 集合（与 rag/milvus_rag.py 一致）为 JSON，便于查看。

用法（项目根目录）:
  PYTHONPATH=. python scripts/export_milvus_rag.py
  PYTHONPATH=. python scripts/export_milvus_rag.py -o data/milvus_dump.json --with-vectors

依赖: pymilvus；需已启动 Milvus（如 docker compose）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

# 项目根
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config.config import Config  # noqa: E402
from pymilvus import Collection, connections, utility  # noqa: E402

_ALIAS = "milvus_export_script"


def _vector_to_json(v: Any) -> List[float]:
    if v is None:
        return []
    if hasattr(v, "tolist"):
        return [float(x) for x in v.tolist()]
    return [float(x) for x in list(v)]


def fetch_all_rows(
    collection: Collection,
    with_vectors: bool,
    limit: int,
) -> List[Dict[str, Any]]:
    """全表拉取。应急 topic 数量远小于 Milvus 单次 query 上限；更大库可改 query_iterator。"""
    fields = ["topic_id", "title", "source", "text"]
    if with_vectors:
        fields.append("embedding")

    # topic_id 为 VARCHAR 主键；建库逻辑保证非空（空则 _row_i）
    expr = 'topic_id != ""'
    cap = min(limit, 16384)
    res = collection.query(expr=expr, output_fields=fields, limit=cap)
    return res[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 Milvus emergency_rag 集合为 JSON")
    parser.add_argument(
        "-o",
        "--output",
        default="milvus_emergency_rag_export.json",
        help="输出 JSON 路径（相对当前工作目录或绝对路径）",
    )
    parser.add_argument(
        "--with-vectors",
        action="store_true",
        help="包含 embedding 全量浮点列表（文件会明显变大）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100000,
        help="最多导出条数（默认 100000）",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="集合名（默认取 config.RAG_MILVUS_COLLECTION）",
    )
    args = parser.parse_args()

    cfg = Config()
    host = getattr(cfg, "MILVUS_HOST", "localhost")
    port = int(getattr(cfg, "MILVUS_PORT", 19530))
    name = args.collection or getattr(cfg, "RAG_MILVUS_COLLECTION", "emergency_rag")

    if connections.has_connection(_ALIAS):
        connections.disconnect(_ALIAS)
    connections.connect(alias=_ALIAS, host=host, port=port, timeout=30)

    try:
        if not utility.has_collection(name, using=_ALIAS):
            print(f"集合不存在: {name}（请先启动 app 或 Milvus 并完成建表）", file=sys.stderr)
            sys.exit(1)

        col = Collection(name, using=_ALIAS)
        col.load()
        n = col.num_entities
        rows = fetch_all_rows(col, args.with_vectors, args.limit)
        payload: Dict[str, Any] = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "milvus_host": host,
            "milvus_port": port,
            "collection": name,
            "num_entities_reported": int(n),
            "rows_exported": len(rows),
            "with_vectors": args.with_vectors,
            "rows": [],
        }
        for r in rows:
            item: Dict[str, Any] = {
                "topic_id": r.get("topic_id", ""),
                "title": r.get("title", ""),
                "source": r.get("source", ""),
                "text": r.get("text", ""),
            }
            if args.with_vectors and "embedding" in r:
                item["embedding"] = _vector_to_json(r["embedding"])
                item["embedding_dim"] = len(item["embedding"])
            payload["rows"].append(item)

        out_path = args.output
        if not os.path.isabs(out_path):
            out_path = os.path.join(os.getcwd(), out_path)
        _dir = os.path.dirname(out_path)
        if _dir:
            os.makedirs(_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"已导出 {len(rows)} 条到: {out_path}")
        print(f"集合 {name} 上报实体数: {n}")
    finally:
        if connections.has_connection(_ALIAS):
            connections.disconnect(_ALIAS)


if __name__ == "__main__":
    main()
