#!/usr/bin/env python3
"""汇总人工评测表：均值 + Cohen's Kappa（两人列均已填写时）。"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def _to_int(raw: str) -> Optional[int]:
    s = (raw or "").strip()
    if s == "":
        return None
    return int(float(s))


def cohen_kappa(y1: Sequence[int], y2: Sequence[int], labels: Sequence[int]) -> float:
    n = len(y1)
    if n == 0:
        return float("nan")
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    cm = [[0] * k for _ in range(k)]
    for a, b in zip(y1, y2):
        cm[idx[a]][idx[b]] += 1
    po = sum(cm[i][i] for i in range(k)) / n
    row = [sum(cm[i]) / n for i in range(k)]
    col = [sum(cm[r][c] for r in range(k)) / n for c in range(k)]
    pe = sum(row[i] * col[i] for i in range(k))
    if math.isclose(1.0 - pe, 0.0):
        return 1.0 if math.isclose(po, 1.0) else 0.0
    return (po - pe) / (1.0 - pe)


def _mean(vals: List[Optional[int]]) -> Optional[float]:
    xs = [v for v in vals if v is not None]
    if not xs:
        return None
    return round(sum(xs) / len(xs), 3)


def summarize(sheet: Path, output: Path) -> dict:
    with sheet.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    f1 = [_to_int(r.get("factual_r1", "")) for r in rows]
    f2 = [_to_int(r.get("factual_r2", "")) for r in rows]
    c1 = [_to_int(r.get("completeness_r1", "")) for r in rows]
    c2 = [_to_int(r.get("completeness_r2", "")) for r in rows]
    s1 = [_to_int(r.get("safety_r1", "")) for r in rows]
    s2 = [_to_int(r.get("safety_r2", "")) for r in rows]

    paired_f = [(a, b) for a, b in zip(f1, f2) if a is not None and b is not None]
    paired_c = [(a, b) for a, b in zip(c1, c2) if a is not None and b is not None]
    paired_s = [(a, b) for a, b in zip(s1, s2) if a is not None and b is not None]

    out: Dict = {
        "meta": {
            "sheet": str(sheet),
            "n_rows": len(rows),
            "summarized_at": datetime.now().isoformat(),
            "notes": "仅统计已填单元格；两人未齐时 Kappa 为 null。",
        },
        "rater1": {
            "n_factual": sum(v is not None for v in f1),
            "factual_mean": _mean(f1),
            "completeness_mean": _mean(c1),
            "safety_rate": _mean(s1),
        },
        "rater2": {
            "n_factual": sum(v is not None for v in f2),
            "factual_mean": _mean(f2),
            "completeness_mean": _mean(c2),
            "safety_rate": _mean(s2),
        },
        "agreement": {
            "n_paired_factual": len(paired_f),
            "kappa_factual": None,
            "kappa_completeness": None,
            "kappa_safety": None,
        },
        "by_phase": {},
    }
    if paired_f:
        out["agreement"]["kappa_factual"] = round(
            cohen_kappa([a for a, _ in paired_f], [b for _, b in paired_f], (0, 1, 2)),
            3,
        )
    if paired_c:
        out["agreement"]["kappa_completeness"] = round(
            cohen_kappa(
                [a for a, _ in paired_c],
                [b for _, b in paired_c],
                (1, 2, 3, 4, 5),
            ),
            3,
        )
    if paired_s:
        out["agreement"]["kappa_safety"] = round(
            cohen_kappa([a for a, _ in paired_s], [b for _, b in paired_s], (0, 1)),
            3,
        )

    by_phase: Dict[str, List[dict]] = {}
    for r in rows:
        by_phase.setdefault(r.get("phase") or "未知", []).append(r)
    for phase, rs in by_phase.items():
        out["by_phase"][phase] = {
            "n": len(rs),
            "factual_r1_mean": _mean([_to_int(x.get("factual_r1", "")) for x in rs]),
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["agreement"], ensure_ascii=False, indent=2))
    print(f"已写入 {output}")
    return out


def main():
    import argparse

    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", type=Path, default=root / "data/eval/human_eval_sheet.csv")
    ap.add_argument(
        "--output",
        type=Path,
        default=root / "data/eval/human_eval_summary.json",
    )
    args = ap.parse_args()
    summarize(args.sheet, args.output)


if __name__ == "__main__":
    main()
