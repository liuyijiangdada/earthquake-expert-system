#!/usr/bin/env python3
"""
【可选】向 data/earthquake_data.csv 追加合成记录。当前项目优先使用
data/real_earthquakes_catalog.json + data/emergency_knowledge.json（真实简录 + 应急知识），
本脚本仅作压力测试或兼容旧 CSV 流程。

用法:
  python scripts/generate_initial_kg_data.py              # 默认追加 400 条
  python scripts/generate_initial_kg_data.py --extra 200  # 指定条数
  python scripts/generate_initial_kg_data.py --dry-run  # 只打印统计不写文件

说明: Neo4j 仅在「库中无 Earthquake 节点」时从 CSV 全量导入。
若图库已有数据，需先清空 Neo4j 数据后再启动应用以重新导入，例如:
  docker compose exec neo4j cypher-shell -u neo4j -p password \\
    \"MATCH (n) DETACH DELETE n\"
然后重启 app.py。
"""

import argparse
import os
import random
import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

# 区域名 + 大致经纬度范围（中国及周边活动带）
_REGION_BOXES = [
    ("四川", 27.0, 34.0, 97.0, 108.0),
    ("云南", 21.0, 29.0, 98.0, 106.0),
    ("青海", 31.0, 39.0, 89.0, 103.0),
    ("西藏", 27.0, 36.0, 78.0, 99.0),
    ("新疆", 34.0, 49.0, 73.0, 96.0),
    ("甘肃", 32.0, 43.0, 92.0, 108.0),
    ("台湾", 21.5, 25.5, 119.5, 122.5),
    ("河北", 36.0, 42.0, 113.0, 120.0),
    ("山西", 34.0, 41.0, 110.0, 115.0),
    ("陕西", 31.0, 40.0, 105.0, 111.0),
    ("宁夏", 35.0, 39.0, 104.0, 107.0),
    ("山东", 34.0, 38.5, 115.0, 122.0),
    ("河南", 31.0, 36.5, 110.0, 116.5),
    ("重庆", 28.0, 32.0, 105.0, 110.0),
    ("贵州", 24.0, 29.5, 103.0, 109.5),
    ("广东", 20.0, 25.5, 109.0, 117.5),
    ("广西", 21.0, 26.5, 104.0, 112.0),
    ("海南", 18.0, 20.2, 108.5, 111.0),
    ("辽宁", 38.5, 43.5, 118.0, 125.5),
    ("吉林", 40.5, 46.0, 121.0, 131.0),
    ("黑龙江", 43.0, 50.5, 121.0, 135.0),
    ("内蒙古", 37.0, 50.5, 97.0, 126.0),
    ("江苏", 30.5, 35.5, 116.0, 122.0),
    ("浙江", 27.0, 31.5, 118.0, 123.0),
    ("安徽", 29.0, 34.5, 114.5, 119.0),
    ("福建", 23.5, 28.5, 116.0, 120.5),
    ("江西", 24.5, 30.0, 113.0, 118.5),
    ("湖南", 24.5, 30.5, 108.0, 114.0),
    ("湖北", 29.0, 33.5, 108.0, 116.5),
]


def _intensity_for_mag(m: float) -> str:
    if m < 2.5:
        return "Ⅰ"
    if m < 3.5:
        return "Ⅱ"
    if m < 4.0:
        return "Ⅱ-Ⅲ"
    if m < 4.5:
        return "Ⅲ"
    if m < 5.0:
        return "Ⅲ-Ⅳ"
    if m < 6.0:
        return "Ⅳ"
    if m < 7.0:
        return "Ⅳ-Ⅴ"
    return "Ⅴ"


def _random_time(rng: random.Random) -> str:
    start = datetime(2018, 1, 1)
    end = datetime(2025, 12, 31)
    delta = end - start
    sec = rng.randint(0, int(delta.total_seconds()))
    t = start + timedelta(seconds=sec)
    return t.strftime("%Y-%m-%d %H:%M:%S")


def _generate_rows(count: int, start_id: int, rng: random.Random) -> list[dict]:
    rows = []
    for i in range(count):
        name, la0, la1, lo0, lo1 = rng.choice(_REGION_BOXES)
        mag = round(rng.uniform(2.0, 7.2), 1)
        depth = round(rng.uniform(5.0, 120.0), 1)
        lat = round(rng.uniform(la0, la1), 4)
        lon = round(rng.uniform(lo0, lo1), 4)
        intensity = _intensity_for_mag(mag)
        eid = f"eq_{start_id + i}"
        desc = f"{name}地区发生{mag}级地震，震源深度{depth}公里，周边震感因烈度而异。"
        rows.append(
            {
                "id": eid,
                "name": f"{name}{mag}级地震",
                "time": _random_time(rng),
                "magnitude": mag,
                "depth": depth,
                "location": name,
                "latitude": lat,
                "longitude": lon,
                "intensity": intensity,
                "description": desc,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="追加合成地震 CSV 数据")
    parser.add_argument("--extra", type=int, default=400, help="追加条数（默认 400）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，便于复现")
    parser.add_argument("--dry-run", action="store_true", help="不写文件")
    args = parser.parse_args()

    cfg = Config()
    path = cfg.EARTHQUAKE_DATA_FILE
    if not os.path.isfile(path):
        print(f"错误: 找不到 {path}")
        sys.exit(1)

    df_old = pd.read_csv(path)
    max_num = 0
    for s in df_old["id"].astype(str):
        if s.startswith("eq_"):
            try:
                max_num = max(max_num, int(s.split("_", 1)[1]))
            except ValueError:
                pass
    start_id = max_num + 1

    rng = random.Random(args.seed)
    new_rows = _generate_rows(args.extra, start_id, rng)
    df_new = pd.DataFrame(new_rows)
    df_out = pd.concat([df_old, df_new], ignore_index=True)

    print(f"原有 {len(df_old)} 条，追加 {len(df_new)} 条，合计 {len(df_out)} 条（新 id: eq_{start_id} … eq_{start_id + len(df_new) - 1}）")

    if args.dry_run:
        return

    df_out.to_csv(path, index=False, encoding="utf-8")
    print(f"已写入: {path}")


if __name__ == "__main__":
    main()
