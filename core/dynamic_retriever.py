#!/usr/bin/env python3
"""动态知识检索模块：从外部 API / 爬虫获取实时地震数据，转为可注入提示的文本段落。"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

TZ_CN = timezone(timedelta(hours=8))


@dataclass
class DynamicResult:
    items: List[Dict[str, Any]] = field(default_factory=list)
    source: str = ""
    fetched_at: str = ""
    is_fresh: bool = False
    error: Optional[str] = None

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
            lines.append(f"  {i}. {loc}：{mag}级，深度{depth}km，时间{t}")

        if not self.is_fresh:
            lines.append(f"  （数据可能超过{max_age_minutes}分钟未更新，建议刷新）")

        return "\n".join(lines)


class DynamicRetriever:
    def __init__(self, config=None):
        self._cache: Optional[DynamicResult] = None
        self._cache_ts: float = 0
        self._cache_ttl: int = 300
        self._usgs_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        self._enabled = True
        self._timeout = 15
        self._min_magnitude = 4.5
        self._max_items = 10
        self._china_filter = True
        self._china_min_lat = 18.0
        self._china_max_lat = 54.0
        self._china_min_lon = 73.0
        self._china_max_lon = 135.0

        if config:
            self._enabled = getattr(config, "DYNAMIC_RETRIEVAL_ENABLED", True)
            self._cache_ttl = getattr(config, "DYNAMIC_CACHE_TTL_SECONDS", 300)
            self._timeout = getattr(config, "DYNAMIC_API_TIMEOUT", 15)
            self._min_magnitude = getattr(config, "DYNAMIC_MIN_MAGNITUDE", 4.5)
            self._max_items = getattr(config, "DYNAMIC_MAX_ITEMS", 10)
            self._china_filter = getattr(config, "DYNAMIC_CHINA_FILTER_ENABLED", True)
            self._china_min_lat = float(getattr(config, "DYNAMIC_CHINA_MIN_LAT", 18.0))
            self._china_max_lat = float(getattr(config, "DYNAMIC_CHINA_MAX_LAT", 54.0))
            self._china_min_lon = float(getattr(config, "DYNAMIC_CHINA_MIN_LON", 73.0))
            self._china_max_lon = float(getattr(config, "DYNAMIC_CHINA_MAX_LON", 135.0))

    @property
    def enabled(self) -> bool:
        return self._enabled

    def fetch_recent_earthquakes(self, force: bool = False) -> DynamicResult:
        if not self._enabled:
            return DynamicResult(error="动态检索已关闭")

        now = time.time()
        if not force and self._cache and (now - self._cache_ts) < self._cache_ttl:
            self._cache.is_fresh = True
            return self._cache

        try:
            result = self._fetch_usgs()
            self._cache = result
            self._cache_ts = now
            result.is_fresh = True
            return result
        except Exception as e:
            logger.warning("动态数据获取失败: %s", e)
            if self._cache:
                self._cache.is_fresh = False
                return self._cache
            return DynamicResult(error=str(e))

    def _in_china_bbox(self, latitude: float, longitude: float) -> bool:
        return (
            self._china_min_lat <= latitude <= self._china_max_lat
            and self._china_min_lon <= longitude <= self._china_max_lon
        )

    def _fetch_usgs(self) -> DynamicResult:
        params = {
            "format": "geojson",
            "starttime": (datetime.utcnow() - timedelta(hours=48)).isoformat(),
            "endtime": datetime.utcnow().isoformat(),
            "minmagnitude": self._min_magnitude,
            "limit": self._max_items * 3 if self._china_filter else self._max_items,
            "orderby": "time",
        }
        if self._china_filter:
            params.update({
                "minlatitude": self._china_min_lat,
                "maxlatitude": self._china_max_lat,
                "minlongitude": self._china_min_lon,
                "maxlongitude": self._china_max_lon,
            })

        try:
            resp = requests.get(self._usgs_url, params=params, timeout=self._timeout)
            resp.raise_for_status()
        except requests.ConnectionError:
            raise ConnectionError("无法连接 USGS API，请检查网络连接")
        except requests.Timeout:
            raise TimeoutError(f"USGS API 请求超时（{self._timeout}秒），请稍后重试")
        except requests.HTTPError as e:
            raise RuntimeError(f"USGS API 返回错误：{e.response.status_code}")
        except requests.RequestException as e:
            raise RuntimeError(f"USGS API 请求失败：{type(e).__name__}")

        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError):
            raise RuntimeError("USGS API 返回数据格式异常，无法解析")

        items = []
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            geom = feature.get("geometry", {})
            coords = geom.get("coordinates", [0, 0, 0])

            ts_ms = props.get("time", 0)
            try:
                t_str = datetime.fromtimestamp(ts_ms / 1000, tz=TZ_CN).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            except Exception:
                t_str = "未知"

            lat = coords[1] if len(coords) > 1 else 0.0
            lon = coords[0] if len(coords) > 0 else 0.0
            if self._china_filter and not self._in_china_bbox(lat, lon):
                continue

            items.append({
                "id": feature.get("id", ""),
                "magnitude": props.get("mag", 0),
                "location": props.get("place", "未知"),
                "time": t_str,
                "depth": abs(coords[2]) if len(coords) > 2 else 0,
                "latitude": lat,
                "longitude": lon,
                "url": props.get("url", ""),
            })
            if len(items) >= self._max_items:
                break

        fetched_at = datetime.now(TZ_CN).strftime("%Y-%m-%d %H:%M:%S")
        source = "USGS实时地震目录（中国区）" if self._china_filter else "USGS实时地震目录"
        return DynamicResult(
            items=items,
            source=source,
            fetched_at=fetched_at,
        )

    def fetch_for_phase(self, phase_str: str, user_text: str = "",
                        force: bool = False) -> DynamicResult:
        if phase_str == "震前":
            return DynamicResult(
                items=[],
                source="震前无需动态数据",
                fetched_at="",
                is_fresh=True,
            )

        if phase_str == "震中":
            result = self.fetch_recent_earthquakes(force=force)
            return result

        if phase_str == "震后":
            result = self.fetch_recent_earthquakes(force=force)
            return result

        return self.fetch_recent_earthquakes(force=force)

    def invalidate_cache(self):
        self._cache = None
        self._cache_ts = 0
