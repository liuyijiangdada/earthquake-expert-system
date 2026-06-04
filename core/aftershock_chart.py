#!/usr/bin/env python3
"""基于动态地震目录生成余震震级时序图（Matplotlib）。"""

from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_AFTERSHOCK_KW = ("余震", "时序", "序列", "还会震吗", "还会震")


class AftershockChartGenerator:
    def __init__(self, config=None):
        self._enabled = True
        self._min_items = 2
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        rel = getattr(config, "GENERATED_MEDIA_DIR", "static/generated") if config else "static/generated"
        self._out_dir = rel if os.path.isabs(rel) else os.path.join(root, rel)
        self._url_prefix = getattr(config, "GENERATED_MEDIA_URL_PREFIX", "/generated-media") if config else "/generated-media"
        if config:
            self._enabled = getattr(config, "LAYER3_AFTERSHOCK_CHART_ENABLED", True)
            self._min_items = int(getattr(config, "LAYER3_AFTERSHOCK_MIN_ITEMS", 2))
        os.makedirs(self._out_dir, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def should_handle(self, text: str) -> bool:
        return bool(text) and any(k in text for k in _AFTERSHOCK_KW)

    def generate(self, items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not self.enabled or len(items) < self._min_items:
            return None
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib import rcParams

            rcParams["font.sans-serif"] = [
                "PingFang SC",
                "Heiti SC",
                "SimHei",
                "Arial Unicode MS",
                "DejaVu Sans",
            ]
            rcParams["axes.unicode_minus"] = False

            times = []
            mags = []
            for it in items:
                t = it.get("time", "")
                mag = it.get("magnitude")
                if mag is None:
                    continue
                try:
                    dt = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue
                times.append(dt)
                mags.append(float(mag))

            if len(times) < self._min_items:
                return None

            pairs = sorted(zip(times, mags), key=lambda x: x[0])
            times, mags = zip(*pairs)

            fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=100)
            ax.plot(times, mags, marker="o", linestyle="-", color="#c0392b", linewidth=1.5)
            ax.set_xlabel("时间")
            ax.set_ylabel("震级")
            ax.set_title("近期地震震级时序（动态目录）")
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate(rotation=25)
            fig.tight_layout()

            fname = f"aftershock_{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
            fpath = os.path.join(self._out_dir, fname)
            fig.savefig(fpath, format="png")
            plt.close(fig)

            self._prune_old_files(keep=30)

            return {
                "id": f"gen_aftershock_{fname}",
                "type": "image",
                "url": f"{self._url_prefix}/{fname}",
                "caption": "近期地震震级时序图（基于实时目录生成）",
                "source": "本地生成",
                "phase": "震中",
            }
        except Exception as e:
            logger.warning("余震时序图生成失败: %s", e)
            return None

    def _prune_old_files(self, keep: int = 30):
        try:
            files = [
                os.path.join(self._out_dir, f)
                for f in os.listdir(self._out_dir)
                if f.startswith("aftershock_") and f.endswith(".png")
            ]
            files.sort(key=os.path.getmtime, reverse=True)
            for old in files[keep:]:
                os.remove(old)
        except OSError:
            pass
