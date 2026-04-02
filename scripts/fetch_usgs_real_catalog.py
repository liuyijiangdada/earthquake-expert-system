#!/usr/bin/env python3
"""
从 USGS FDSN API 拉取近期真实地震事件，写入 data/usgs_recent_earthquakes.json。
需联网。可将条目手工合并进 data/real_earthquakes_catalog.json，或自行写导入脚本。

示例:
  python scripts/fetch_usgs_real_catalog.py
  python scripts/fetch_usgs_real_catalog.py --days 7 --minmag 4.5
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import requests

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "usgs_recent_earthquakes.json",
)


def fetch_geojson(days: float, min_mag: float):
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    params = {
        "format": "geojson",
        "starttime": start.isoformat(),
        "endtime": end.isoformat(),
        "minmagnitude": min_mag,
        "orderby": "time",
        "limit": 200,
    }
    r = requests.get(url, params=params, timeout=45)
    r.raise_for_status()
    return r.json()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=float, default=30.0, help="回溯天数")
    p.add_argument("--minmag", type=float, default=4.0, help="最小震级")
    args = p.parse_args()

    data = fetch_geojson(args.days, args.minmag)
    feats = data.get("features", [])
    out_list = []
    for i, f in enumerate(feats):
        pr = f.get("properties", {})
        geom = f.get("geometry", {})
        coords = geom.get("coordinates", [0, 0, 0])
        eid = pr.get("ids", "") or f"usgs_{i}_{int(pr.get('time', 0))}"
        mag = float(pr.get("mag") or 0)
        t_ms = pr.get("time")
        if t_ms:
            ts = datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        else:
            ts = ""
        place = pr.get("place") or "未知地区"
        lat, lon, dep = float(coords[1]), float(coords[0]), float(abs(coords[2] or 0))
        region = place.split(",")[-1].strip() if "," in place else place
        out_list.append(
            {
                "id": str(eid).replace(",", "_")[:120],
                "name": f"{place} M{mag}",
                "time": ts,
                "magnitude": mag,
                "depth": dep,
                "location": place,
                "region": region[:32],
                "latitude": lat,
                "longitude": lon,
                "intensity": "参考USGS",
                "description": f"USGS公开目录：{place}，Mw/M约{mag}，深度约{dep}km。",
                "links_to_topics": [],
            }
        )

    payload = {
        "meta": {
            "source": "USGS FDSN",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "count": len(out_list),
        },
        "earthquakes": out_list,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"已写入 {len(out_list)} 条 -> {OUT}")


if __name__ == "__main__":
    main()
