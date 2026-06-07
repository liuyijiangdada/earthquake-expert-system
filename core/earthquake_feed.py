#!/usr/bin/env python3
"""统一地震动态数据源：中国地震台网（CEIC）+ USGS，供 DynamicRetriever 与 Neo4j 更新共用。"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

TZ_CN = timezone(timedelta(hours=8))


@dataclass
class EarthquakeFeedConfig:
    hours_window: int = 48
    min_magnitude: float = 4.5
    max_items: int = 10
    timeout: int = 15
    china_filter: bool = True
    china_min_lat: float = 18.0
    china_max_lat: float = 54.0
    china_min_lon: float = 73.0
    china_max_lon: float = 135.0
    providers: Tuple[str, ...] = ("ceic", "usgs")
    ceic_enabled: bool = True
    ceic_base_url: str = "http://www.ceic.ac.cn"
    ceic_speedsearch_num: int = 2
    usgs_enabled: bool = True
    usgs_url: str = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    prefer_ceic_for_china: bool = True

    @classmethod
    def from_config(cls, config=None) -> "EarthquakeFeedConfig":
        if config is None:
            return cls()
        hours = int(getattr(config, "DYNAMIC_HOURS_WINDOW", 48))
        ceic_num = int(getattr(config, "CEIC_SPEEDSEARCH_NUM", 0))
        if ceic_num <= 0:
            ceic_num = 1 if hours <= 24 else 2
        raw_providers = getattr(config, "DYNAMIC_FEED_PROVIDERS", ("ceic", "usgs"))
        if isinstance(raw_providers, str):
            providers = tuple(p.strip() for p in raw_providers.split(",") if p.strip())
        else:
            providers = tuple(raw_providers)
        return cls(
            hours_window=hours,
            min_magnitude=float(getattr(config, "DYNAMIC_MIN_MAGNITUDE", 4.5)),
            max_items=int(getattr(config, "DYNAMIC_MAX_ITEMS", 10)),
            timeout=int(getattr(config, "DYNAMIC_API_TIMEOUT", 15)),
            china_filter=bool(getattr(config, "DYNAMIC_CHINA_FILTER_ENABLED", True)),
            china_min_lat=float(getattr(config, "DYNAMIC_CHINA_MIN_LAT", 18.0)),
            china_max_lat=float(getattr(config, "DYNAMIC_CHINA_MAX_LAT", 54.0)),
            china_min_lon=float(getattr(config, "DYNAMIC_CHINA_MIN_LON", 73.0)),
            china_max_lon=float(getattr(config, "DYNAMIC_CHINA_MAX_LON", 135.0)),
            providers=providers or ("ceic", "usgs"),
            ceic_enabled=bool(getattr(config, "CEIC_ENABLED", True)),
            ceic_base_url=str(getattr(config, "CEIC_BASE_URL", "http://www.ceic.ac.cn")).rstrip("/"),
            ceic_speedsearch_num=ceic_num,
            usgs_enabled=bool(getattr(config, "USGS_ENABLED", True)),
            usgs_url=str(getattr(config, "USGS_EVENT_API_URL", "https://earthquake.usgs.gov/fdsnws/event/1/query")),
            prefer_ceic_for_china=bool(getattr(config, "DYNAMIC_PREFER_CEIC_FOR_CHINA", True)),
        )


@dataclass
class FeedFetchResult:
    items: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    fetched_at: str = ""

    @property
    def source_label(self) -> str:
        if not self.sources:
            return "无可用动态源"
        if len(self.sources) == 1:
            return self.sources[0]
        return " + ".join(self.sources)


def _in_china_bbox(cfg: EarthquakeFeedConfig, lat: float, lon: float) -> bool:
    return (
        cfg.china_min_lat <= lat <= cfg.china_max_lat
        and cfg.china_min_lon <= lon <= cfg.china_max_lon
    )


def _parse_time_cn(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "未知"
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            dt = datetime.strptime(text, fmt).replace(tzinfo=TZ_CN)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    return text


def _item_key(item: Dict[str, Any]) -> Tuple:
    return (
        round(float(item.get("latitude") or 0), 1),
        round(float(item.get("longitude") or 0), 1),
        round(float(item.get("magnitude") or 0), 1),
        (item.get("time") or "")[:16],
    )


def _normalize_item(
    *,
    source: str,
    source_id: str,
    magnitude: float,
    location: str,
    time_str: str,
    depth: float,
    latitude: float,
    longitude: float,
    url: str = "",
    name: str = "",
    description: str = "",
    intensity: str = "未知",
) -> Dict[str, Any]:
    loc = (location or "未知").strip()
    mag = float(magnitude or 0)
    eq_id = f"{source}_{source_id}" if source_id else f"{source}_{_item_key({'latitude': latitude, 'longitude': longitude, 'magnitude': mag, 'time': time_str})}"
    return {
        "id": eq_id,
        "name": name or f"{loc}地震",
        "location": loc,
        "time": time_str,
        "magnitude": mag,
        "depth": float(depth or 0),
        "latitude": float(latitude or 0),
        "longitude": float(longitude or 0),
        "intensity": intensity,
        "description": description or f"震级{mag}级地震，{loc}，{time_str}",
        "url": url,
        "source": source,
        "source_id": source_id,
    }


def _parse_ceic_payload(raw: str) -> List[Dict[str, Any]]:
    """解析 CEIC speedsearch 返回（常为 ({...}) 或 {...}）。"""
    text = (raw or "").strip()
    if not text or text.startswith("<"):
        raise ValueError("CEIC 返回非 JSON 内容")
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    text = re.sub(r',"page":"(.*?)","num":', ',"num":', text)
    payload = json.loads(text)
    rows = payload.get("shuju") or []
    if not isinstance(rows, list):
        raise ValueError("CEIC shuju 字段格式异常")
    return rows


def fetch_ceic(cfg: EarthquakeFeedConfig) -> List[Dict[str, Any]]:
    url = f"{cfg.ceic_base_url}/ajax/speedsearch"
    resp = requests.get(
        url,
        params={"num": cfg.ceic_speedsearch_num, "page": 1},
        timeout=cfg.timeout,
        headers={"User-Agent": "EarthquakeQA/1.0 (research; +https://www.ceic.ac.cn)"},
    )
    resp.raise_for_status()
    rows = _parse_ceic_payload(resp.text)
    items: List[Dict[str, Any]] = []
    for row in rows:
        try:
            mag = float(row.get("M") or 0)
            if mag < cfg.min_magnitude:
                continue
            lat = float(row.get("EPI_LAT") or 0)
            lon = float(row.get("EPI_LON") or 0)
            if cfg.china_filter and not _in_china_bbox(cfg, lat, lon):
                continue
            loc = str(row.get("LOCATION_C") or "未知位置")
            t_str = _parse_time_cn(str(row.get("O_TIME") or ""))
            depth = float(row.get("EPI_DEPTH") or 0)
            sid = str(row.get("NEW_DID") or row.get("id") or f"{t_str}_{lat}_{lon}")
            detail_url = f"{cfg.ceic_base_url}/{sid}.html" if sid else cfg.ceic_base_url
            items.append(
                _normalize_item(
                    source="ceic",
                    source_id=sid,
                    magnitude=mag,
                    location=loc,
                    time_str=t_str,
                    depth=depth,
                    latitude=lat,
                    longitude=lon,
                    url=detail_url,
                    description=f"据中国地震台网测定，{t_str}在{loc}发生{mag}级地震，震源深度{depth}公里。",
                )
            )
        except (TypeError, ValueError) as e:
            logger.debug("跳过 CEIC 条目: %s", e)
    items.sort(key=lambda x: x.get("time", ""), reverse=True)
    return items[: cfg.max_items * 3]


def fetch_usgs(cfg: EarthquakeFeedConfig) -> List[Dict[str, Any]]:
    end = datetime.utcnow()
    start = end - timedelta(hours=cfg.hours_window)
    params: Dict[str, Any] = {
        "format": "geojson",
        "starttime": start.isoformat(),
        "endtime": end.isoformat(),
        "minmagnitude": cfg.min_magnitude,
        "limit": cfg.max_items * 3 if cfg.china_filter else cfg.max_items,
        "orderby": "time",
    }
    if cfg.china_filter:
        params.update({
            "minlatitude": cfg.china_min_lat,
            "maxlatitude": cfg.china_max_lat,
            "minlongitude": cfg.china_min_lon,
            "maxlongitude": cfg.china_max_lon,
        })
    resp = requests.get(cfg.usgs_url, params=params, timeout=cfg.timeout)
    resp.raise_for_status()
    data = resp.json()
    items: List[Dict[str, Any]] = []
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})
        coords = geom.get("coordinates", [0, 0, 0])
        lat = float(coords[1] if len(coords) > 1 else 0)
        lon = float(coords[0] if len(coords) > 0 else 0)
        if cfg.china_filter and not _in_china_bbox(cfg, lat, lon):
            continue
        ts_ms = props.get("time", 0)
        try:
            t_str = datetime.fromtimestamp(ts_ms / 1000, tz=TZ_CN).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            t_str = "未知"
        sid = str(feature.get("id") or props.get("ids") or "")
        mag = float(props.get("mag") or 0)
        place = str(props.get("place") or "未知")
        depth = abs(float(coords[2])) if len(coords) > 2 else 0.0
        items.append(
            _normalize_item(
                source="usgs",
                source_id=sid,
                magnitude=mag,
                location=place,
                time_str=t_str,
                depth=depth,
                latitude=lat,
                longitude=lon,
                url=str(props.get("url") or ""),
                description=f"USGS公开目录：{place}，Mw/M约{mag}，深度约{depth}km。",
            )
        )
        if len(items) >= cfg.max_items * 3:
            break
    return items


def merge_feed_items(
    batches: List[Tuple[str, List[Dict[str, Any]]]],
    *,
    max_items: int,
    prefer_ceic: bool = True,
) -> List[Dict[str, Any]]:
    """合并多源结果，近似事件去重；默认 CEIC 条目优先。"""
    ordered_batches = batches
    if prefer_ceic:
        ordered_batches = sorted(batches, key=lambda x: 0 if x[0] == "ceic" else 1)

    merged: List[Dict[str, Any]] = []
    seen = set()
    for _src, batch in ordered_batches:
        for item in batch:
            key = _item_key(item)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= max_items:
                return merged
    merged.sort(key=lambda x: x.get("time", ""), reverse=True)
    return merged[:max_items]


def fetch_recent_earthquakes(config=None) -> FeedFetchResult:
    """拉取近期地震目录（统一参数，多源合并）。"""
    cfg = EarthquakeFeedConfig.from_config(config)
    result = FeedFetchResult(fetched_at=datetime.now(TZ_CN).strftime("%Y-%m-%d %H:%M:%S"))
    batches: List[Tuple[str, List[Dict[str, Any]]]] = []

    for provider in cfg.providers:
        if provider == "ceic" and cfg.ceic_enabled:
            try:
                items = fetch_ceic(cfg)
                if items:
                    batches.append(("ceic", items))
                    result.sources.append("中国地震台网（CEIC）")
                else:
                    result.errors.append("CEIC 返回空列表")
            except Exception as e:
                logger.warning("CEIC 获取失败: %s", e)
                result.errors.append(f"CEIC: {e}")
        elif provider == "usgs" and cfg.usgs_enabled:
            try:
                items = fetch_usgs(cfg)
                if items:
                    batches.append(("usgs", items))
                    result.sources.append("USGS实时地震目录（中国区）" if cfg.china_filter else "USGS实时地震目录")
                else:
                    result.errors.append("USGS 返回空列表")
            except Exception as e:
                logger.warning("USGS 获取失败: %s", e)
                result.errors.append(f"USGS: {e}")

    result.items = merge_feed_items(
        batches,
        max_items=cfg.max_items,
        prefer_ceic=cfg.prefer_ceic_for_china,
    )
    return result


def to_dynamic_result_items(feed: FeedFetchResult) -> Tuple[List[Dict[str, Any]], str]:
    """转为 DynamicRetriever 使用的 items + source 标签。"""
    return feed.items, feed.source_label
