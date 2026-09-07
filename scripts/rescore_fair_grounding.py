#!/usr/bin/env python3
"""对已有回答按当前提示证据重打分：旧 grounding vs 证据归一化口径并列。"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RAG_USE_MEMORY_RAG", "1")

from core.dynamic_snapshot import DEFAULT_SNAPSHOT_PATH
from core.eval_runtime import (
    apply_baseline_to_stack,
    build_eval_stack,
    spec_by_id,
)
from core.eval_scoring import dual_to_dict, extract_refs, score_both, truncate_ref


def _spec_for_item(baseline: str):
    try:
        return spec_by_id(baseline)
    except KeyError:
        return spec_by_id("B3")


def _agg(items: List[dict], scheme: str) -> Dict[str, Any]:
    by: Dict[str, List[dict]] = defaultdict(list)
    for it in items:
        by[it["baseline"]].append(it)
    out = {}
    for bid, rows in by.items():
        n = len(rows) or 1
        fact = [((r.get("scores") or {}).get(scheme) or {}).get("factual") or 0 for r in rows]
        comp = [
            ((r.get("scores") or {}).get(scheme) or {}).get("completeness") or 0 for r in rows
        ]
        fmt = [((r.get("scores") or {}).get(scheme) or {}).get("format_ok") or 0 for r in rows]
        out[bid] = {
            "n": len(rows),
            "factual_accuracy_pct": round(100 * sum(fact) / n, 1),
            "completeness_mean": round(sum(comp) / n, 2),
            "format_compliance_pct": round(100 * sum(fmt) / n, 1),
        }
    return out


def rescore(
    *,
    source: Path,
    output_dir: Path,
    snapshot_path: Path = DEFAULT_SNAPSHOT_PATH,
    force_json_kg: bool = False,
    skip_rag: bool = False,
) -> dict:
    data = json.loads(source.read_text(encoding="utf-8"))
    items_in = data.get("items") or []
    from config.config import Config

    Config.RAG_USE_MEMORY_RAG = True
    Config.LAYER3_ENABLED = False
    stack = build_eval_stack(
        Config,
        snapshot_path=snapshot_path,
        force_json_kg=force_json_kg,
        skip_rag=skip_rag,
    )

    cache: Dict[Tuple[str, str, str], Tuple[str, dict]] = {}
    out_items = []
    for it in items_in:
        bid = it.get("baseline") or "B3"
        phase = it.get("phase") or ""
        question = it.get("question") or ""
        response = it.get("response") or ""
        spec = _spec_for_item(bid)
        key = (bid, phase, question)
        if key not in cache:
            apply_baseline_to_stack(stack, spec)
            prompt, meta, _tag = stack.builder.prepare(
                question, for_vision=False, history=None
            )
            cache[key] = (prompt, meta)
        prompt, meta = cache[key]
        dual = score_both(
            response, prompt, meta, phase, kg_on=spec.kg, rag_on=spec.rag
        )
        kg_ref, rag_ref, dyn_ref = extract_refs(prompt)
        out_items.append(
            {
                "baseline": bid,
                "phase": phase,
                "question": question,
                "scores": dual_to_dict(dual),
                "refs": {
                    "kg": truncate_ref(kg_ref, 400),
                    "rag": truncate_ref(rag_ref, 400),
                    "dyn": truncate_ref(dyn_ref, 400),
                },
            }
        )

    payload = {
        "meta": {
            "source": str(source),
            "rescored_at": datetime.now().isoformat(),
            "kg_backend": stack.kg_backend,
            "rag_backend": stack.rag_backend,
            "n_items": len(out_items),
            "notes": (
                "normalized：注入证据等权 overlap，无 KG+RAG 额外加分、无 B0 长度 cap；"
                "与论文旧表并列，不替代、不上调。"
            ),
        },
        "summary_legacy": _agg(out_items, "legacy"),
        "summary_normalized": _agg(out_items, "normalized"),
        "items": out_items,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "table_fair_vs_legacy.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== legacy 事实一致性% ===")
    for bid, s in payload["summary_legacy"].items():
        print(f"  {bid}: {s['factual_accuracy_pct']}")
    print("=== normalized 事实一致性% ===")
    for bid, s in payload["summary_normalized"].items():
        print(f"  {bid}: {s['factual_accuracy_pct']}")
    print(f"已写入 {out_path}")
    return payload


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data/eval/ablation_results_60.json",
    )
    ap.add_argument("--output", type=Path, default=ROOT / "data/eval")
    ap.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT_PATH)
    ap.add_argument("--force-json-kg", action="store_true")
    ap.add_argument("--skip-rag", action="store_true")
    args = ap.parse_args()
    rescore(
        source=args.source,
        output_dir=args.output,
        snapshot_path=args.snapshot,
        force_json_kg=args.force_json_kg,
        skip_rag=args.skip_rag,
    )


if __name__ == "__main__":
    main()
