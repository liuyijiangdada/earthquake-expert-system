#!/usr/bin/env python3
"""调度阈值敏感性：60 题只跑分类 + 知识探测 + decide()，不加载 LLM。"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RAG_USE_MEMORY_RAG", "1")

from core.dynamic_snapshot import DEFAULT_SNAPSHOT_PATH
from core.eval_runtime import build_eval_stack
from core.knowledge_signals import compute_knowledge_signals
from core.scheduler import Scheduler

DEFAULT_GRID = {
    "SCHEDULER_DYNAMIC_CONFIDENCE_THRESHOLD": (0.2, 0.4, 0.6),
    "SCHEDULER_STATIC_CONFIDENCE_THRESHOLD": (0.7, 0.9, 0.95),
    "SCHEDULER_URGENCY_HIGH_THRESHOLD": (0.3, 0.5, 0.7),
    "SCHEDULER_URGENCY_CRITICAL_THRESHOLD": (0.5, 0.7, 0.9),
}
DEFAULT_KEYS = (
    0.4,
    0.9,
    0.5,
    0.7,
)


def load_questions(path: Path) -> List[Tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: List[Tuple[str, str]] = []
    for phase, qs in data.get("phases", {}).items():
        for q in qs:
            out.append((phase, q))
    return out


def cfg_tuple(dyn: float, sta: float, high: float, urg: float) -> Tuple[float, float, float, float]:
    return (float(dyn), float(sta), float(high), float(urg))


def run_sensitivity(
    *,
    questions_file: Path,
    output_dir: Path,
    snapshot_path: Path = DEFAULT_SNAPSHOT_PATH,
    force_json_kg: bool = False,
    skip_rag: bool = False,
) -> dict:
    from config.config import Config

    Config.RAG_USE_MEMORY_RAG = True
    Config.DYNAMIC_RETRIEVAL_ENABLED = True
    Config.KG_CONTEXT_ENABLED = True
    Config.RAG_ENABLED = True
    Config.LAYER3_ENABLED = False

    stack = build_eval_stack(
        Config,
        snapshot_path=snapshot_path,
        force_json_kg=force_json_kg,
        skip_rag=skip_rag,
    )
    questions = load_questions(questions_file)

    probes: List[dict] = []
    for phase, question in questions:
        phase_result = stack.classifier.classify(question)
        signals = compute_knowledge_signals(
            question,
            phase_result.phase.value,
            phase_result,
            config=Config,
            kg=stack.builder._deps.kg,
            emergency_rag=stack.builder._deps.emergency_rag,
            dynamic_retriever=stack.retriever,
        )
        probes.append(
            {
                "phase_gold": phase,
                "question": question,
                "phase_pred": phase_result.phase.value,
                "confidence": round(phase_result.confidence, 3),
                "urgency": round(phase_result.urgency, 3),
                "need_dynamic": phase_result.need_dynamic,
                "static_confidence": round(signals.static_confidence, 3),
                "dynamic_availability": round(signals.dynamic_availability, 3),
                "_phase_result": phase_result,
                "_signals": signals,
            }
        )

    combos = list(
        product(
            DEFAULT_GRID["SCHEDULER_DYNAMIC_CONFIDENCE_THRESHOLD"],
            DEFAULT_GRID["SCHEDULER_STATIC_CONFIDENCE_THRESHOLD"],
            DEFAULT_GRID["SCHEDULER_URGENCY_HIGH_THRESHOLD"],
            DEFAULT_GRID["SCHEDULER_URGENCY_CRITICAL_THRESHOLD"],
        )
    )

    per_combo: Dict[str, Any] = {}
    default_flags: List[bool] = []

    class _Thr:
        SCHEDULER_DYNAMIC_CONFIDENCE_THRESHOLD = 0.4
        SCHEDULER_STATIC_CONFIDENCE_THRESHOLD = 0.9
        SCHEDULER_URGENCY_HIGH_THRESHOLD = 0.5
        SCHEDULER_URGENCY_CRITICAL_THRESHOLD = 0.7

    def _eval(dyn_t, sta_t, high_t, urg_t):
        thr = _Thr()
        thr.SCHEDULER_DYNAMIC_CONFIDENCE_THRESHOLD = dyn_t
        thr.SCHEDULER_STATIC_CONFIDENCE_THRESHOLD = sta_t
        thr.SCHEDULER_URGENCY_HIGH_THRESHOLD = high_t
        thr.SCHEDULER_URGENCY_CRITICAL_THRESHOLD = urg_t
        sched = Scheduler(thr)
        flags: List[bool] = []
        by_phase: Dict[str, List[bool]] = {"震前": [], "震中": [], "震后": []}
        for row in probes:
            d = sched.decide(row["_phase_result"], row["_signals"])
            flags.append(bool(d.use_dynamic))
            gold = row["phase_gold"]
            if gold in by_phase:
                by_phase[gold].append(bool(d.use_dynamic))
        return flags, by_phase

    for dyn_t, sta_t, high_t, urg_t in combos:
        key = f"dyn{dyn_t}_sta{sta_t}_high{high_t}_urg{urg_t}"
        flags, by_phase = _eval(dyn_t, sta_t, high_t, urg_t)
        if cfg_tuple(dyn_t, sta_t, high_t, urg_t) == DEFAULT_KEYS:
            default_flags = list(flags)
        per_combo[key] = {
            "thresholds": {
                "dynamic_confidence": dyn_t,
                "static_confidence": sta_t,
                "urgency_high": high_t,
                "urgency_critical": urg_t,
            },
            "use_dynamic_rate": round(sum(flags) / len(flags), 3) if flags else 0.0,
            "use_dynamic_rate_by_phase": {
                p: round(sum(v) / len(v), 3) if v else 0.0 for p, v in by_phase.items()
            },
            "n_on": int(sum(flags)),
        }

    n = len(default_flags) or 1
    for rec in per_combo.values():
        t = rec["thresholds"]
        flags, _ = _eval(
            t["dynamic_confidence"],
            t["static_confidence"],
            t["urgency_high"],
            t["urgency_critical"],
        )
        flips = sum(int(a != b) for a, b in zip(flags, default_flags))
        rec["flip_vs_default"] = flips
        rec["agree_with_default"] = round(1.0 - flips / n, 3)

    payload_probes = []
    for row in probes:
        clean = {k: v for k, v in row.items() if not k.startswith("_")}
        payload_probes.append(clean)

    out = {
        "meta": {
            "started_at": datetime.now().isoformat(),
            "question_count": len(questions),
            "kg_backend": stack.kg_backend,
            "rag_backend": stack.rag_backend,
            "snapshot_path": str(snapshot_path),
            "default_thresholds": {
                "dynamic_confidence": DEFAULT_KEYS[0],
                "static_confidence": DEFAULT_KEYS[1],
                "urgency_high": DEFAULT_KEYS[2],
                "urgency_critical": DEFAULT_KEYS[3],
            },
            "grid": {k: list(v) for k, v in DEFAULT_GRID.items()},
            "notes": (
                "无 LLM。阶段先验主导震中开启；震后对 urgency_high 敏感。"
                "仅扫描 critical 时决策可零翻转。"
            ),
        },
        "probes": payload_probes,
        "combos": per_combo,
    }
    default_rec = next(
        rec
        for rec in per_combo.values()
        if cfg_tuple(
            rec["thresholds"]["dynamic_confidence"],
            rec["thresholds"]["static_confidence"],
            rec["thresholds"]["urgency_high"],
            rec["thresholds"]["urgency_critical"],
        )
        == DEFAULT_KEYS
    )
    out["default_summary"] = default_rec

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "scheduler_sensitivity.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("默认阈值 use_dynamic 比例:", default_rec["use_dynamic_rate_by_phase"])
    print(f"已写入 {path}")
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data/eval/phase_questions.json",
    )
    ap.add_argument("--output", type=Path, default=ROOT / "data/eval")
    ap.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT_PATH)
    ap.add_argument("--force-json-kg", action="store_true")
    ap.add_argument("--skip-rag", action="store_true")
    args = ap.parse_args()
    run_sensitivity(
        questions_file=args.questions,
        output_dir=args.output,
        snapshot_path=args.snapshot,
        force_json_kg=args.force_json_kg,
        skip_rag=args.skip_rag,
    )


if __name__ == "__main__":
    main()
