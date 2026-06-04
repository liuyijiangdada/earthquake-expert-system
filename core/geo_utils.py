#!/usr/bin/env python3
"""地理计算与地名抽取。"""

from __future__ import annotations

import math
import re
from typing import List, Optional, Tuple
from urllib.parse import quote

from config.constants import CHINA_REGION_NAMES


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(min(1.0, a)))


def extract_coords_from_text(text: str) -> Optional[Tuple[float, float]]:
    """尝试解析「纬度/经度」或「lon,lat」形式坐标。"""
    if not text:
        return None
    m = re.search(
        r"(?:纬度|lat)[^\d-]*(-?\d+\.?\d*)[^\d-]*(?:经度|lon|lng)[^\d-]*(-?\d+\.?\d*)",
        text,
        re.I,
    )
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"(-?\d{1,2}\.\d+)\s*[,，]\s*(-?\d{1,3}\.\d+)", text)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        if abs(a) <= 90 and abs(b) <= 180:
            return a, b
        if abs(b) <= 90 and abs(a) <= 180:
            return b, a
    return None


def match_cities_in_text(text: str, city_aliases: dict) -> List[str]:
    if not text:
        return []
    found = []
    for city, aliases in city_aliases.items():
        for alias in aliases:
            if alias in text:
                found.append(city)
                break
    for region in CHINA_REGION_NAMES:
        if region in text and region not in found:
            found.append(region)
    return found


def amap_marker_url(lat: float, lon: float, title: str = "震中") -> str:
    name = quote(title[:40])
    return f"https://uri.amap.com/marker?position={lon},{lat}&name={name}"


def amap_navigation_url(lat: float, lon: float, name: str) -> str:
    dest = quote(f"{lon},{lat},{name[:30]}")
    return f"https://uri.amap.com/navigation?to={dest}&mode=walk&coordinate=gaode"


def baidu_marker_url(lat: float, lon: float, title: str = "震中") -> str:
    title_q = quote(title[:40])
    return (
        "https://api.map.baidu.com/marker?"
        f"location={lat},{lon}&title={title_q}&content={title_q}&output=html"
    )


def osm_static_map_url(lat: float, lon: float, zoom: int = 6) -> str:
    return (
        "https://staticmap.openstreetmap.de/staticmap.php?"
        f"center={lat},{lon}&zoom={zoom}&size=480x280&markers={lat},{lon},red-pushpin"
    )
