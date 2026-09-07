#!/usr/bin/env python3
"""拉取一次直播目录并写成评测快照；失败则保留已有冻结文件。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from core.dynamic_retriever import DynamicRetriever
from core.dynamic_snapshot import DEFAULT_SNAPSHOT_PATH, save_snapshot


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT_SNAPSHOT_PATH)
    args = ap.parse_args()

    from config.config import Config

    retriever = DynamicRetriever(Config)
    result = retriever.fetch_recent_earthquakes(force=True)
    if result.error or not result.items:
        print(f"直播拉取失败或为空：{result.error or 'no items'}，不覆盖 {args.output}")
        sys.exit(1)
    path = save_snapshot(result, args.output)
    print(f"已冻结 {len(result.items)} 条 -> {path}  source={result.source}")


if __name__ == "__main__":
    main()
