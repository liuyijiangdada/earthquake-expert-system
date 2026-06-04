#!/usr/bin/env python3
"""避难所静态库：最近点检索与导航链接。"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from core.geo_utils import amap_navigation_url, haversine_km, match_cities_in_text

if TYPE_CHECKING:
    from core.amap_client import AmapClient

logger = logging.getLogger(__name__)


class ShelterService:
    def __init__(self, config=None, amap_client: Optional["AmapClient"] = None):
        self._shelters: List[Dict[str, Any]] = []
        self._city_aliases: Dict[str, List[str]] = {}
        self._enabled = True
        self._amap = amap_client
        if config:
            self._enabled = getattr(config, "LAYER3_SHELTER_ENABLED", True)
        self._load(config)

    def _load(self, config):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        rel = getattr(config, "SHELTERS_DATA_FILE", "data/shelters.json") if config else "data/shelters.json"
        path = rel if os.path.isabs(rel) else os.path.join(root, rel)
        if not os.path.isfile(path):
            logger.warning("避难所数据不存在: %s", path)
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._shelters = data.get("shelters", [])
            self._city_aliases = data.get("city_aliases", {})
            logger.info("已加载 %d 条避难所记录", len(self._shelters))
        except Exception as e:
            logger.error("加载避难所数据失败: %s", e)

    @property
    def enabled(self) -> bool:
        return self._enabled and bool(self._shelters)

    def should_handle(self, text: str) -> bool:
        if not text:
            return False
        if any(k in text for k in ("避难", "避难所", "安置点", "疏散点")):
            return True
        if "最近" in text and any(k in text for k in ("在哪", "哪里", "何处", "哪")):
            return True
        return False

    def _resolve_user_location(
        self,
        user_text: str,
        user_lat: Optional[float],
        user_lon: Optional[float],
        cities: List[str],
    ) -> tuple:
        if user_lat is not None and user_lon is not None:
            if self._amap and self._amap.available:
                lon, lat = self._amap.normalize_coords(user_lat, user_lon, input_crs="wgs84")
                return lat, lon
            return user_lat, user_lon

        if self._amap and self._amap.available and cities:
            city = cities[0]
            geo = self._amap.geocode(f"{city}市", city=city)
            if geo:
                lon, lat = geo
                return lat, lon
        return None, None

    def find_nearest(
        self,
        user_text: str,
        *,
        user_lat: Optional[float] = None,
        user_lon: Optional[float] = None,
        top_n: int = 1,
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []

        candidates = list(self._shelters)
        cities = match_cities_in_text(user_text, self._city_aliases)
        if cities:
            city_set = set(cities)
            filtered = [s for s in candidates if s.get("city") in city_set]
            if filtered:
                candidates = filtered

        ulat, ulon = self._resolve_user_location(user_text, user_lat, user_lon, cities)

        if ulat is not None and ulon is not None:
            for s in candidates:
                s["_distance_km"] = haversine_km(
                    ulat, ulon, s["latitude"], s["longitude"]
                )
            candidates.sort(key=lambda x: x.get("_distance_km", 1e9))
        elif cities:
            pass
        else:
            return []

        out = []
        for s in candidates[:top_n]:
            dist = s.get("_distance_km")
            slat, slon = s["latitude"], s["longitude"]
            if self._amap and self._amap.available:
                slon, slat = self._amap.normalize_coords(slat, slon, input_crs="gcj02")
                nav_url = self._amap.web_navigation_url(slon, slat, s.get("name", "避难所"))
            else:
                nav_url = amap_navigation_url(slat, slon, s.get("name", "避难所"))

            item = {
                "id": s.get("id", ""),
                "name": s.get("name", ""),
                "address": s.get("address", ""),
                "city": s.get("city", ""),
                "latitude": slat,
                "longitude": slon,
                "distance_km": round(dist, 2) if dist is not None else None,
                "navigation_url": nav_url,
            }

            if self._amap and self._amap.available and ulon is not None and ulat is not None:
                route = self._amap.walking_route(ulon, ulat, slon, slat)
                if route:
                    item["walking_distance_m"] = route["distance_m"]
                    item["walking_duration_s"] = route["duration_s"]
                    item["walking_duration_min"] = max(1, route["duration_s"] // 60)

            out.append(item)
        return out

    def to_context_text(self, shelters: List[Dict[str, Any]]) -> str:
        if not shelters:
            return ""
        lines = ["【附近避难所（示范数据，请以当地政府发布为准）】"]
        for i, s in enumerate(shelters, 1):
            dist = s.get("distance_km")
            dist_s = f"，直线约{dist}公里" if dist is not None else ""
            walk = ""
            if s.get("walking_duration_min"):
                walk = f"，步行约{s['walking_duration_min']}分钟（高德路径规划）"
            lines.append(
                f"  {i}. {s['name']}（{s.get('address', '')}{dist_s}{walk}）"
            )
        return "\n".join(lines)
