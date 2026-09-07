#!/usr/bin/env python3
"""四基线（B0～B3）快速批量实验 + 自动评分，输出表 7-3。

默认从 60 题评测集中按震前/震中/震后各抽 4 题（共 16 题），关闭动态源与第三层增强，
单次推理、贪心解码，约 5～10 分钟完成（视本机算力而定）。
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["RAG_USE_MEMORY_RAG"] = "1"

import torch  # noqa: E402

from core.eval_scoring import Scores, score_response  # noqa: E402

BASELINES: List[Tuple[str, bool, bool]] = [
    ("B0", False, False),
    ("B1", True, False),
    ("B2", False, True),
    ("B3", True, True),
]


def load_questions_stratified(
    path: Path,
    *,
    per_phase: int = 4,
) -> List[Tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: List[Tuple[str, str]] = []
    for phase, qs in data.get("phases", {}).items():
        for q in qs[:per_phase]:
            out.append((phase, q))
    return out


def _infer_once(app_mod, prompt: str, input_text: str) -> str:
    """单次 prepare 后直接生成，跳过第三层与媒体后处理。"""
    messages = [
        {"role": "system", "content": "你是一个地震专家，专注于回答地震相关问题。"},
        {"role": "user", "content": prompt},
    ]
    text = ""
    for m in messages:
        role, content = m["role"], m["content"]
        if role == "system":
            text += f"<|im_start|>{content}</s>"
        elif role == "user":
            text += f"<|im_start|>user\n{content}</s>"

    text += "<|im_start|>assistant\n"

    max_in = int(getattr(app_mod.config, "LLM_INPUT_MAX_TOKENS", 4096))
    max_new = int(getattr(app_mod.config, "LLM_MAX_NEW_TOKENS", 384))

    inputs = app_mod.tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_in,
    )
    dev = next(app_mod.model.parameters()).device
    input_ids = inputs.input_ids.to(dev)
    attention_mask = inputs.attention_mask.to(dev)

    if app_mod.tokenizer.pad_token_id is None:
        app_mod.tokenizer.pad_token_id = app_mod.tokenizer.eos_token_id

    with torch.no_grad():
        outputs = app_mod.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new,
            temperature=float(getattr(app_mod.config, "LLM_TEMPERATURE", 0.55)),
            do_sample=bool(getattr(app_mod.config, "LLM_DO_SAMPLE", True)),
            top_p=float(getattr(app_mod.config, "LLM_TOP_P", 0.88)),
            repetition_penalty=float(
                getattr(app_mod.config, "LLM_REPETITION_PENALTY", 1.15)
            ),
            no_repeat_ngram_size=int(
                getattr(app_mod.config, "LLM_NO_REPEAT_NGRAM_SIZE", 4)
            ),
        )

    generated = outputs[0][input_ids.shape[-1] :]
    out = app_mod.tokenizer.decode(generated, skip_special_tokens=True).strip()
    out = (
        out.replace("</s>", "")
        .replace("<|im_start|>", "")
        .replace("<|im_end|>", "")
        .strip()
    )
    return app_mod._sanitize_response(out, input_text)


def run_eval(
    *,
    questions_file: Path,
    output_dir: Path,
    per_phase: int = 4,
    limit: Optional[int] = None,
) -> dict:
    from config.config import Config

    Config.RAG_USE_MEMORY_RAG = True
    Config.DYNAMIC_RETRIEVAL_ENABLED = False
    Config.LAYER3_ENABLED = False
    Config.MULTIMODAL_OUTPUT_ENABLED = False
    Config.LAYER3_INJECT_PROMPT = False
    Config.LLM_MAX_NEW_TOKENS = 160
    Config.LLM_DO_SAMPLE = False

    import app as app_mod  # noqa: WPS433

    torch.manual_seed(42)

    questions = load_questions_stratified(questions_file, per_phase=per_phase)
    if limit:
        questions = questions[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: Dict[str, Any] = {
        "meta": {
            "started_at": datetime.now().isoformat(),
            "mode": "fast_stratified",
            "per_phase": per_phase,
            "question_count": len(questions),
            "questions_file": str(questions_file),
            "baselines": [{"id": b[0], "kg": b[1], "rag": b[2]} for b in BASELINES],
            "seed": 42,
            "notes": "关闭动态检索与第三层；贪心解码 max_new_tokens=160； grounding 自动评分",
        },
        "items": [],
        "summary": {},
    }

    total_steps = len(BASELINES) * len(questions)
    step = 0

    for bid, kg_on, rag_on in BASELINES:
        app_mod.config.KG_CONTEXT_ENABLED = kg_on
        app_mod.config.RAG_ENABLED = rag_on
        baseline_scores: List[Scores] = []

        for phase, question in questions:
            step += 1
            t0 = time.time()
            prompt, response_meta, _phase_tag = app_mod.context_builder.prepare(
                question, for_vision=False, history=None
            )
            response = _infer_once(app_mod, prompt, question)
            sc = score_response(
                response,
                prompt,
                response_meta,
                phase,
                kg_on=kg_on,
                rag_on=rag_on,
            )
            baseline_scores.append(sc)
            elapsed = time.time() - t0

            all_results["items"].append(
                {
                    "baseline": bid,
                    "phase": phase,
                    "question": question,
                    "response": response,
                    "scores": {
                        "factual": round(sc.factual, 4),
                        "completeness": round(sc.completeness, 2),
                        "format_ok": sc.format_ok,
                    },
                    "meta": {
                        "phase": response_meta.get("phase"),
                        "static_confidence": response_meta.get("static_confidence"),
                        "rag_topic_ids": response_meta.get("rag_topic_ids"),
                    },
                    "elapsed_sec": round(elapsed, 2),
                }
            )
            print(
                f"[{step}/{total_steps}] {bid} | {phase} | "
                f"f={sc.factual:.2f} c={sc.completeness:.1f} fmt={sc.format_ok} | {elapsed:.1f}s",
                flush=True,
            )

        n = len(baseline_scores)
        all_results["summary"][bid] = {
            "factual_accuracy_pct": round(
                100 * sum(s.factual for s in baseline_scores) / n, 1
            ),
            "completeness_mean": round(
                sum(s.completeness for s in baseline_scores) / n, 2
            ),
            "format_compliance_pct": round(
                100 * sum(s.format_ok for s in baseline_scores) / n, 1
            ),
            "kg_context_enabled": kg_on,
            "rag_enabled": rag_on,
        }

    all_results["meta"]["finished_at"] = datetime.now().isoformat()
    out_json = output_dir / "ablation_results.json"
    out_json.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary_path = output_dir / "table_7_3_summary.json"
    summary_path.write_text(
        json.dumps(all_results["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n=== 表 7-3 汇总 ===")
    for label, s in all_results["summary"].items():
        print(
            f"{label}: 事实一致性 {s['factual_accuracy_pct']}% | "
            f"完整性 {s['completeness_mean']} | 格式合规 {s['format_compliance_pct']}%"
        )
    print(f"\n结果已保存: {out_json}")
    return all_results


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data/eval/phase_questions.json",
    )
    ap.add_argument("--output", type=Path, default=ROOT / "data/eval")
    ap.add_argument(
        "--per-phase",
        type=int,
        default=4,
        help="每阶段抽样题数（默认 4，共 16 题 × 4 基线 = 64 次推理）",
    )
    ap.add_argument("--limit", type=int, default=None, help="总题数上限（调试）")
    args = ap.parse_args()
    run_eval(
        questions_file=args.questions,
        output_dir=args.output,
        per_phase=args.per_phase,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
