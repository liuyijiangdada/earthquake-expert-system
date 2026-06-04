#!/usr/bin/env python3
"""第三层增强输出：地图/导航、避难所、余震图、政策 RSS。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from core.aftershock_chart import AftershockChartGenerator
from core.geo_utils import (
    amap_marker_url,
    baidu_marker_url,
    extract_coords_from_text,
    osm_static_map_url,
)
from core.policy_rss import PolicyFeedFetcher
from core.shelter_service import ShelterService

if TYPE_CHECKING:
    from core.amap_client import AmapClient

logger = logging.getLogger(__name__)

_EPICENTER_KW = ("震中", "刚才", "多大", "哪里地震", "实时", "最新震情")


@dataclass
class EnrichmentResult:
    media_resources: List[Dict[str, Any]] = field(default_factory=list)
    prompt_section: str = ""
    layer3_meta: Dict[str, Any] = field(default_factory=dict)


class OutputEnricher:
    def __init__(self, config=None, amap_client: Optional["AmapClient"] = None):
        self._enabled = bool(getattr(config, "LAYER3_ENABLED", True)) if config else True
        self._map_enabled = bool(getattr(config, "LAYER3_MAP_ENABLED", True)) if config else True
        self._inject_prompt = bool(getattr(config, "LAYER3_INJECT_PROMPT", False)) if config else False
        self._amap = amap_client
        self._shelters = ShelterService(config, amap_client=amap_client)
        self._charts = AftershockChartGenerator(config)
        self._policy = PolicyFeedFetcher(config)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _static_map_resource(
        self,
        res_id: str,
        lat: float,
        lon: float,
        caption: str,
        *,
        zoom: int = 10,
        phase: str = "震中",
        input_crs: str = "wgs84",
        marker_lat: Optional[float] = None,
        marker_lon: Optional[float] = None,
    ) -> Dict[str, Any]:
        if self._amap and self._amap.available:
            glon, glat = self._amap.normalize_coords(lat, lon, input_crs=input_crs)
            url = self._amap.public_static_map_path(
                glon,
                glat,
                zoom=zoom,
                marker_lon=marker_lon,
                marker_lat=marker_lat,
            )
            source = "高德静态地图"
        else:
            url = osm_static_map_url(lat, lon, zoom=zoom)
            source = "OpenStreetMap"
        return {
            "id": res_id,
            "type": "image",
            "url": url,
            "caption": caption,
            "source": source,
            "phase": phase,
        }

    def enrich(
        self,
        user_text: str,
        phase_tag: str = "",
        *,
        dynamic_items: Optional[List[Dict[str, Any]]] = None,
    ) -> EnrichmentResult:
        result = EnrichmentResult()
        if not self.enabled or not user_text:
            return result

        prompt_parts = []
        media: List[Dict[str, Any]] = []

        if self._shelters.enabled and self._shelters.should_handle(user_text):
            coords = extract_coords_from_text(user_text)
            ulat, ulon = (coords if coords else (None, None))
            nearest = self._shelters.find_nearest(
                user_text, user_lat=ulat, user_lon=ulon, top_n=2
            )
            if nearest:
                result.layer3_meta["shelters"] = nearest
                ctx = self._shelters.to_context_text(nearest)
                if ctx:
                    prompt_parts.append(ctx)
                for s in nearest:
                    media.append({
                        "id": f"shelter_nav_{s['id']}",
                        "type": "link",
                        "url": s["navigation_url"],
                        "caption": f"导航至{s['name']}"
                        + (
                            f"（步行约{s['walking_duration_min']}分钟）"
                            if s.get("walking_duration_min")
                            else ""
                        ),
                        "source": "高德" if self._amap and self._amap.available else "高德 URI",
                        "phase": phase_tag or "震中",
                    })
                    media.append(
                        self._static_map_resource(
                            f"shelter_map_{s['id']}",
                            s["latitude"],
                            s["longitude"],
                            f"{s['name']}位置图",
                            zoom=14,
                            phase=phase_tag or "震中",
                            input_crs="gcj02",
                        )
                    )

        if self._map_enabled and dynamic_items and any(k in user_text for k in _EPICENTER_KW):
            epicenter = dynamic_items[0]
            lat = epicenter.get("latitude")
            lon = epicenter.get("longitude")
            if lat is not None and lon is not None:
                loc = epicenter.get("location", "震中")
                mag = epicenter.get("magnitude", "?")
                display_addr = loc
                if self._amap and self._amap.available:
                    glon, glat = self._amap.normalize_coords(lat, lon, input_crs="wgs84")
                    rev = self._amap.regeocode(glon, glat)
                    if rev:
                        display_addr = rev
                    result.layer3_meta["epicenter"] = {
                        "latitude_wgs84": lat,
                        "longitude_wgs84": lon,
                        "latitude_gcj02": glat,
                        "longitude_gcj02": glon,
                        "location": loc,
                        "formatted_address": display_addr,
                        "magnitude": mag,
                    }
                else:
                    result.layer3_meta["epicenter"] = {
                        "latitude": lat,
                        "longitude": lon,
                        "location": loc,
                        "magnitude": mag,
                    }

                prompt_parts.append(
                    f"【最新震中参考】{display_addr}，约{mag}级"
                    f"（WGS84: {lat:.2f}, {lon:.2f}）。"
                )
                media.append(
                    self._static_map_resource(
                        "layer3_epicenter_map",
                        lat,
                        lon,
                        f"震中位置图（{display_addr}）",
                        zoom=6,
                        phase="震中",
                        input_crs="wgs84",
                    )
                )
                if self._amap and self._amap.available:
                    glon, glat = self._amap.normalize_coords(lat, lon, input_crs="wgs84")
                    media.append({
                        "id": "layer3_amap_epicenter",
                        "type": "link",
                        "url": self._amap.web_navigation_url(glon, glat, display_addr),
                        "caption": "高德地图导航至震中参考位置",
                        "source": "高德 Web",
                        "phase": "震中",
                    })
                else:
                    media.append({
                        "id": "layer3_amap_epicenter",
                        "type": "link",
                        "url": amap_marker_url(lat, lon, loc),
                        "caption": "在高德地图中查看震中",
                        "source": "高德 URI",
                        "phase": "震中",
                    })
                media.append({
                    "id": "layer3_baidu_epicenter",
                    "type": "link",
                    "url": baidu_marker_url(lat, lon, loc),
                    "caption": "在百度地图中查看震中",
                    "source": "百度地图",
                    "phase": "震中",
                })

        if self._charts.enabled and self._charts.should_handle(user_text) and dynamic_items:
            chart = self._charts.generate(dynamic_items)
            if chart:
                media.append(chart)
                result.layer3_meta["aftershock_chart"] = chart.get("url")

        if self._policy.enabled:
            for item in self._policy.match_for_query(user_text, max_items=2):
                media.append(item)

        result.media_resources = _dedupe_media(media)
        if self._inject_prompt and prompt_parts:
            result.prompt_section = "\n".join(prompt_parts) + "\n"
        return result


def merge_media_resources(
    base: List[Dict[str, Any]],
    extra: List[Dict[str, Any]],
    *,
    max_total: int = 8,
) -> List[Dict[str, Any]]:
    merged = _dedupe_media((base or []) + (extra or []))
    return merged[:max_total]


def _dedupe_media(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for it in items:
        key = it.get("id") or it.get("url")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out
