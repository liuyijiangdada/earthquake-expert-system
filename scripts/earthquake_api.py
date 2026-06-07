#!/usr/bin/env python3
"""实时地震数据获取模块（统一 CEIC + USGS，写入本地 JSON 缓存）。"""

import json
import os
import sys
from typing import List

_APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

from config.config import Config
from core.earthquake_feed import fetch_recent_earthquakes

_CACHE_JSON = os.path.join(_APP_ROOT, "data", "earthquake_data.json")


def get_recent_earthquakes() -> List[dict]:
    """获取近期地震数据（与 DynamicRetriever 共用参数与数据源）。"""
    feed = fetch_recent_earthquakes(Config())
    return feed.items


def update_earthquake_data() -> List[dict]:
    """拉取最新地震并合并写入 data/earthquake_data.json。"""
    recent = get_recent_earthquakes()
    if not recent:
        return []

    try:
        with open(_CACHE_JSON, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if not isinstance(existing, list):
            existing = []
    except (FileNotFoundError, json.JSONDecodeError):
        existing = []

    existing_ids = {str(eq.get("id")) for eq in existing if eq.get("id")}
    new_items = [eq for eq in recent if str(eq.get("id")) not in existing_ids]

    merged = new_items + existing
    os.makedirs(os.path.dirname(_CACHE_JSON), exist_ok=True)
    with open(_CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(merged[:1000], f, ensure_ascii=False, indent=2)

    print(f"更新了 {len(new_items)} 条地震数据（来源：{', '.join({eq.get('source', '?') for eq in recent}) or '无'}）")
    return new_items


if __name__ == "__main__":
    print("获取实时地震数据（CEIC + USGS 统一接口）...")
    new_earthquakes = update_earthquake_data()
    print(f"成功获取 {len(new_earthquakes)} 条新地震数据")
