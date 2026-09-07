#!/usr/bin/env python3
"""冻结动态速报快照，供离线消融复现（避免 CEIC/USGS 直播抖动）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from core.dynamic_retriever import DynamicResult, DynamicRetriever

DEFAULT_SNAPSHOT_PATH = Path("data/eval/dynamic_feed_snapshot.json")


def snapshot_to_result(data: Dict[str, Any]) -> DynamicResult:
    return DynamicResult(
        items=list(data.get("items") or []),
        source=str(data.get("source") or "frozen-snapshot"),
        fetched_at=str(data.get("fetched_at") or ""),
        is_fresh=True,
        error=data.get("error"),
        feed_errors=list(data.get("feed_errors") or []),
    )


def result_to_snapshot(result: DynamicResult) -> Dict[str, Any]:
    return {
        "fetched_at": result.fetched_at,
        "source": result.source,
        "items": list(result.items),
        "error": result.error,
        "feed_errors": list(result.feed_errors),
        "note": "离线消融冻结快照；评测不得再请求直播目录。",
    }


def load_snapshot(path: Optional[Path] = None) -> DynamicResult:
    p = Path(path) if path else DEFAULT_SNAPSHOT_PATH
    data = json.loads(p.read_text(encoding="utf-8"))
    return snapshot_to_result(data)


def save_snapshot(result: DynamicResult, path: Optional[Path] = None) -> Path:
    p = Path(path) if path else DEFAULT_SNAPSHOT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(result_to_snapshot(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return p


def apply_snapshot_to_retriever(
    retriever: DynamicRetriever,
    path: Optional[Path] = None,
) -> DynamicResult:
    result = load_snapshot(path)
    retriever.apply_frozen_snapshot(result)
    return result
