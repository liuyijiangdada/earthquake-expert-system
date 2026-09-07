#!/usr/bin/env python3
"""扫描训练数据中混入的化工/期货等无关领域样本（不修改文件，仅统计）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 与 app.py 的 _UNRELATED_TOPICS 对齐（化工/期货/能源）
OFFTOPIC_KEYWORDS = [
    "聚氯乙烯", "聚乙烯", "期货", "产能", "开工率", "供需", "下游需求",
    "库存水平", "弱势震荡", "PVC", "能源化工", "甲醇", "乙二醇", "纯碱",
    "现货", "盘面", "套利", "基差", "持仓", "主力合约", "沥青", "PP",
    "塑料", "橡胶", "PTA", "玻璃", "螺纹", "铁矿石",
]

ROOT = Path(__file__).resolve().parent.parent
FILES = [
    "data/sft_train.jsonl",
    "data/sft_val.jsonl",
    "data/ollama_finetune_data.jsonl",
    "data/train_data.json",
    "data/val_data.json",
]


def _iter_samples(path: Path):
    """统一迭代不同格式的样本，产出 (user_text, assistant_text)。"""
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "messages" in obj:
                    u = next((m["content"] for m in obj["messages"] if m["role"] == "user"), "")
                    a = next((m["content"] for m in obj["messages"] if m["role"] == "assistant"), "")
                    yield u, a
                elif "prompt" in obj:
                    yield obj.get("prompt", ""), obj.get("response", "")
    else:  # json 数组（alpaca 风格）
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return
        for item in data:
            if isinstance(item, dict):
                yield item.get("input", "") or item.get("instruction", ""), item.get("output", "")


def _hit_keywords(text: str) -> list:
    return [kw for kw in OFFTOPIC_KEYWORDS if kw in text]


def main():
    total_all = 0
    off_all = 0
    for rel in FILES:
        path = ROOT / rel
        if not path.is_file():
            print(f"[跳过] {rel} 不存在")
            continue
        total = 0
        off = 0
        examples = []
        for u, a in _iter_samples(path):
            total += 1
            hits = _hit_keywords(u) + _hit_keywords(a)
            if hits:
                off += 1
                if len(examples) < 3:
                    examples.append((u[:80], sorted(set(hits))[:5]))
        total_all += total
        off_all += off
        rate = (off / total * 100) if total else 0
        print(f"[{rel}] 总样本 {total}，疑似污染 {off}（{rate:.1f}%）")
        for u, kws in examples:
            print(f"    例: {u!r}  命中 {kws}")
    print("-" * 50)
    print(f"合计: {total_all} 样本，{off_all} 疑似污染（{off_all/total_all*100:.1f}%）")


if __name__ == "__main__":
    sys.exit(main() or 0)
