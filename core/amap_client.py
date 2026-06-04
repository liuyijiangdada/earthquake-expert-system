#!/usr/bin/env python3
"""高德地图 Web 服务 API 封装（静态图、地理编码、步行路径规划）。"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests

from core.coord_convert import wgs84_to_gcj02

logger = logging.getLogger(__name__)

_BASE = "https://restapi.amap.com/v3"


class AmapClient:
    def __init__(
        self,
        api_key: str = "",
        *,
        enabled: bool = True,
        timeout: int = 12,
        static_size: str = "480*280",
        coord_system: str = "gcj02",
        geocode_cache_ttl: int = 86400,
    ):
        self._key = (api_key or "").strip()
        self._enabled = enabled and bool(self._key)
        self._timeout = timeout
        self._static_size = static_size
        self._coord_system = coord_system
        self._geocode_cache: Dict[str, Tuple[float, float, float]] = {}
        self._geocode_cache_ttl = geocode_cache_ttl

    @classmethod
    def from_config(cls, config=None) -> "AmapClient":
        key = ""
        if config:
            key = getattr(config, "AMAP_WEB_SERVICE_KEY", "") or ""
        if not key:
            key = os.environ.get("AMAP_WEB_SERVICE_KEY", "")
        enabled = bool(getattr(config, "AMAP_WEB_SERVICE_ENABLED", True)) if config else True
        return cls(
            key,
            enabled=enabled,
            timeout=int(getattr(config, "AMAP_API_TIMEOUT", 12)) if config else 12,
            static_size=getattr(config, "AMAP_STATIC_MAP_SIZE", "480*280") if config else "480*280",
            coord_system=getattr(config, "AMAP_COORD_SYSTEM", "gcj02") if config else "gcj02",
        )

    @property
    def available(self) -> bool:
        return self._enabled

    def normalize_coords(
        self,
        lat: float,
        lon: float,
        *,
        input_crs: str = "wgs84",
    ) -> Tuple[float, float]:
        """统一为 (经度, 纬度) GCJ-02。"""
        if input_crs.lower() in ("wgs84", "gps"):
            lon, lat = wgs84_to_gcj02(lon, lat)
        return lon, lat

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.available:
            raise RuntimeError("高德 Web 服务未配置 Key")
        q = dict(params)
        q["key"] = self._key
        url = f"{_BASE}/{path.lstrip('/')}"
        resp = requests.get(url, params=q, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()
        if str(data.get("status")) != "1":
            info = data.get("info") or data.get("infocode") or "unknown"
            raise RuntimeError(f"高德 API 错误：{info}")
        return data

    def geocode(self, address: str, city: str = "") -> Optional[Tuple[float, float]]:
        """地址 → (经度, 纬度) GCJ-02。"""
        if not address or not self.available:
            return None
        cache_key = f"{city}|{address}"
        now = time.time()
        cached = self._geocode_cache.get(cache_key)
        if cached and (now - cached[2]) < self._geocode_cache_ttl:
            return cached[0], cached[1]

        params = {"address": address}
        if city:
            params["city"] = city
        try:
            data = self._get("geocode/geo", params)
            geocodes = data.get("geocodes") or []
            if not geocodes:
                return None
            loc = geocodes[0].get("location", "")
            parts = loc.split(",")
            if len(parts) != 2:
                return None
            lon, lat = float(parts[0]), float(parts[1])
            self._geocode_cache[cache_key] = (lon, lat, now)
            return lon, lat
        except Exception as e:
            logger.warning("高德地理编码失败: %s", e)
            return None

    def regeocode(self, lon: float, lat: float) -> str:
        if not self.available:
            return ""
        try:
            data = self._get(
                "geocode/regeo",
                {"location": f"{lon},{lat}", "extensions": "base"},
            )
            regeo = data.get("regeocode") or {}
            return (regeo.get("formatted_address") or "").strip()
        except Exception as e:
            logger.warning("高德逆地理编码失败: %s", e)
            return ""

    def walking_route(
        self,
        origin_lon: float,
        origin_lat: float,
        dest_lon: float,
        dest_lat: float,
    ) -> Optional[Dict[str, Any]]:
        """步行路径：distance(米)、duration(秒)。"""
        if not self.available:
            return None
        try:
            data = self._get(
                "direction/walking",
                {
                    "origin": f"{origin_lon},{origin_lat}",
                    "destination": f"{dest_lon},{dest_lat}",
                },
            )
            route = data.get("route") or {}
            paths = route.get("paths") or []
            if not paths:
                return None
            p0 = paths[0]
            return {
                "distance_m": int(p0.get("distance", 0)),
                "duration_s": int(p0.get("duration", 0)),
            }
        except Exception as e:
            logger.warning("高德步行规划失败: %s", e)
            return None

    def build_static_map_url(
        self,
        lon: float,
        lat: float,
        *,
        zoom: int = 10,
        markers: Optional[List[Tuple[float, float, str]]] = None,
    ) -> str:
        """返回带 key 的静态图 URL（仅供服务端拉取）。"""
        if not self.available:
            raise RuntimeError("高德未启用")
        parts: List[str] = []
        labels = "ABCDEFGHIJK"
        all_pts = [(lon, lat, "center")] + (markers or [])
        for i, (mlon, mlat, _label) in enumerate(all_pts[:9]):
            tag = labels[i % len(labels)]
            parts.append(f"mid,,{tag}:{mlon},{mlat}")
        params = {
            "location": f"{lon},{lat}",
            "zoom": max(3, min(18, int(zoom))),
            "size": self._static_size,
            "markers": "|".join(parts),
            "key": self._key,
        }
        return f"{_BASE}/staticmap?{urlencode(params)}"

    def fetch_static_map(
        self,
        lon: float,
        lat: float,
        *,
        zoom: int = 10,
        markers: Optional[List[Tuple[float, float, str]]] = None,
    ) -> bytes:
        url = self.build_static_map_url(lon, lat, zoom=zoom, markers=markers)
        resp = requests.get(url, timeout=self._timeout)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if "image" not in ctype and len(resp.content) < 200:
            raise RuntimeError("高德静态图返回非图片内容")
        return resp.content

    def public_static_map_path(
        self,
        lon: float,
        lat: float,
        *,
        zoom: int = 10,
        label: str = "",
        marker_lon: Optional[float] = None,
        marker_lat: Optional[float] = None,
    ) -> str:
        """前端可用的代理路径（不暴露 Key）。"""
        q = {
            "lon": f"{lon:.6f}",
            "lat": f"{lat:.6f}",
            "zoom": str(zoom),
        }
        if label:
            q["label"] = label[:40]
        if marker_lon is not None and marker_lat is not None:
            q["mlon"] = f"{marker_lon:.6f}"
            q["mlat"] = f"{marker_lat:.6f}"
        return f"/api/amap/static-map?{urlencode(q)}"

    def web_navigation_url(self, lon: float, lat: float, name: str) -> str:
        from urllib.parse import quote

        dest = quote(f"{lon},{lat},{name[:30]}")
        return f"https://uri.amap.com/navigation?to={dest}&mode=walk&coordinate=gaode"
