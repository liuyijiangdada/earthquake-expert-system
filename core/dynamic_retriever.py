#!/usr/bin/env python3
"""动态知识检索模块：从 CEIC / USGS 获取实时地震数据，转为可注入提示的文本段落。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from core.earthquake_feed import fetch_recent_earthquakes

logger = logging.getLogger(__name__)

TZ_CN = timezone(timedelta(hours=8))


@dataclass
class DynamicResult:
    items: List[Dict[str, Any]] = field(default_factory=list)
    source: str = ""
    fetched_at: str = ""
    is_fresh: bool = False
    error: Optional[str] = None
    feed_errors: List[str] = field(default_factory=list)

    def to_context_text(self, max_items: int = 5, max_age_minutes: int = 30) -> str:
        if self.error:
            return f"【动态信息·获取失败】{self.error}"
        if not self.items:
            return "【动态信息】当前无最新实时数据。"

        lines = [f"【动态信息·{self.source}（更新于 {self.fetched_at}）】"]
        for i, item in enumerate(self.items[:max_items], 1):
            mag = item.get("magnitude", "?")
            loc = item.get("location", "未知")
            t = item.get("time", "未知")
            depth = item.get("depth", "?")
            src = item.get("source", "")
            src_tag = f"[{src.upper()}]" if src else ""
            lines.append(f"  {i}. {src_tag}{loc}：{mag}级，深度{depth}km，时间{t}")

        if not self.is_fresh:
            lines.append(f"  （数据可能超过{max_age_minutes}分钟未更新，建议刷新）")
        if self.feed_errors:
            lines.append(f"  （部分数据源不可用：{'；'.join(self.feed_errors[:2])}）")

        return "\n".join(lines)


class DynamicRetriever:
    def __init__(self, config=None):
        self._config = config
        self._cache: Optional[DynamicResult] = None
        self._cache_ts: float = 0
        self._cache_ttl: int = 300
        self._enabled = True
        self._max_items = 10

        self._snapshot_only = False
        if config:
            self._enabled = getattr(config, "DYNAMIC_RETRIEVAL_ENABLED", True)
            self._cache_ttl = getattr(config, "DYNAMIC_CACHE_TTL_SECONDS", 300)
            self._max_items = getattr(config, "DYNAMIC_MAX_ITEMS", 10)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def apply_frozen_snapshot(self, result: DynamicResult) -> None:
        """评测用：后续 fetch 只返回该快照，不再请求直播目录。"""
        self._cache = result
        self._cache_ts = time.time()
        self._cache_ttl = 10**12
        self._snapshot_only = True

    def fetch_recent_earthquakes(self, force: bool = False) -> DynamicResult:
        if not self._enabled:
            return DynamicResult(error="动态检索已关闭")

        if self._snapshot_only:
            return self._cache or DynamicResult(error="动态快照为空")

        now = time.time()
        if not force and self._cache and (now - self._cache_ts) < self._cache_ttl:
            self._cache.is_fresh = True
            return self._cache

        try:
            feed = fetch_recent_earthquakes(self._config)
            if not feed.items:
                err = "；".join(feed.errors) if feed.errors else "未获取到符合条件的地震事件"
                if self._cache:
                    self._cache.is_fresh = False
                    self._cache.feed_errors = feed.errors
                    return self._cache
                return DynamicResult(error=err, feed_errors=feed.errors)

            result = DynamicResult(
                items=feed.items,
                source=feed.source_label,
                fetched_at=feed.fetched_at,
                is_fresh=True,
                feed_errors=feed.errors,
            )
            self._cache = result
            self._cache_ts = now
            return result
        except Exception as e:
            logger.warning("动态数据获取失败: %s", e)
            if self._cache:
                self._cache.is_fresh = False
                return self._cache
            return DynamicResult(error=str(e))

    def fetch_for_phase(self, phase_str: str, user_text: str = "",
                        force: bool = False) -> DynamicResult:
        if phase_str == "震前":
            return DynamicResult(
                items=[],
                source="震前无需动态数据",
                fetched_at="",
                is_fresh=True,
            )

        return self.fetch_recent_earthquakes(force=force)

    def invalidate_cache(self):
        self._cache = None
        self._cache_ts = 0
