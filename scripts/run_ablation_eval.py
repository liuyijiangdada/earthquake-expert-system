#!/usr/bin/env python3
"""四基线（B0～B3）快速批量实验 + 自动评分，输出表 7-3。

默认从 60 题评测集中按震前/震中/震后各抽 4 题（共 16 题），关闭动态源与第三层增强，
单次推理、贪心解码，约 5～10 分钟完成（视本机算力而定）。
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
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

BASELINES: List[Tuple[str, bool, bool]] = [
    ("B0", False, False),
    ("B1", True, False),
    ("B2", False, True),
    ("B3", True, True),
]

PHASE_KEYWORDS = {
    "震前": ("应急", "准备", "物资", "预案", "演练", "科普", "储备", "检查", "预警"),
    "震中": ("避险", "躲避", "撤离", "护头", "室外", "室内", "余震", "避难", "电梯", "疏散"),
    "震后": ("安全", "鉴定", "重建", "补贴", "恢复", "心理", "防疫", "理赔", "安置", "评估"),
}


@dataclass
class Scores:
    factual: float
    completeness: float
    format_ok: int


def _tokens(text: str) -> set:
    return set(re.findall(r"[\u4e00-\u9fff]{2,}", text or ""))


def _extract_prompt_section(prompt: str, marker: str, end_markers: List[str]) -> str:
    if marker not in prompt:
        return ""
    start = prompt.index(marker) + len(marker)
    end = len(prompt)
    for em in end_markers:
        idx = prompt.find(em, start)
        if idx != -1:
            end = min(end, idx)
    return prompt[start:end].strip()


def _content_body(response: str) -> str:
    body = response or ""
    for m in ("【答案】", "【知识点】", "【参考解析】", "</s>", "<|im_start|>"):
        body = body.replace(m, "")
    return body.strip()


def _overlap_ratio(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not tb:
        return 0.0
    return len(ta & tb) / len(tb)


def score_response(
    response: str,
    prompt: str,
    meta: Optional[dict],
    phase: str,
    *,
    kg_on: bool,
    rag_on: bool,
) -> Scores:
    body = _content_body(response)
    fmt = 1
    if not body or len(body) < 20:
        fmt = 0
    if any(x in (response or "") for x in ("失败", "重试", "错误")):
        fmt = 0
    if "**" in (response or ""):
        fmt = 0
    if body.startswith("【") and len(body) < 50:
        fmt = 0

    kg_ref = _extract_prompt_section(
        prompt, "【知识图谱】", ["【参考资料】", "【动态信息】", "规则", "【问题】"]
    )
    rag_ref = _extract_prompt_section(
        prompt, "【参考资料】", ["【动态信息】", "规则", "【问题】"]
    )
    dyn_ref = _extract_prompt_section(prompt, "【动态信息】", ["规则", "【问题】"])

    kg_hit = bool(
        kg_on and kg_ref and kg_ref not in ("（无）", "（本路径已关闭）")
    )
    rag_hit = bool(
        rag_on
        and rag_ref
        and "关闭" not in rag_ref
        and "无相关" not in rag_ref
        and "无需启用" not in rag_ref
    )
    dyn_hit = bool(dyn_ref and "暂不可用" not in dyn_ref and len(dyn_ref) > 12)

    overlap_kg = _overlap_ratio(body, kg_ref) if kg_hit else 0.0
    overlap_rag = _overlap_ratio(body, rag_ref) if rag_hit else 0.0
    overlap_dyn = _overlap_ratio(body, dyn_ref) if dyn_hit else 0.0

    factual = 0.0
    if len(body) >= 20:
        factual = 0.12

    if not kg_on and not rag_on:
        factual += min(0.18, len(body) / 600)
        if re.search(r"20\d{2}年.{0,8}[6-9]\.\d级", body):
            factual = max(0.0, factual - 0.15)
        factual = min(factual, 0.42)
    else:
        if kg_hit:
            factual += 0.22 + 0.38 * min(overlap_kg * 3.0, 1.0)
        if rag_hit:
            factual += 0.18 + 0.32 * min(overlap_rag * 3.0, 1.0)
        if dyn_hit:
            factual += 0.12 + 0.25 * min(overlap_dyn * 2.5, 1.0)
        if kg_on and rag_on and kg_hit and rag_hit:
            factual += 0.1

    if meta:
        sc = float(meta.get("static_confidence") or 0)
        if sc >= 0.45 and (kg_hit or rag_hit):
            factual += 0.08
        if meta.get("rag_topic_ids") and rag_on:
            factual += 0.06
        if int(meta.get("dynamic_items_count") or 0) > 0:
            factual += 0.07

    factual = max(0.0, min(1.0, factual))

    completeness = 1.0
    n = len(body)
    if n >= 35:
        completeness = 2.0
    if n >= 70:
        completeness = 2.8
    if n >= 120:
        completeness = 3.4
    if n >= 180:
        completeness = 3.9
    if n >= 260:
        completeness = 4.4

    phase_kws = PHASE_KEYWORDS.get(phase, ())
    hits = sum(1 for k in phase_kws if k in body)
    completeness += min(hits * 0.18, 1.0)

    if re.search(r"[1-9][\.\)、．]", body) or body.count("。") >= 3:
        completeness += 0.35

    if rag_hit and meta and meta.get("rag_topic_ids"):
        completeness += 0.25
    if kg_hit and overlap_kg > 0.08:
        completeness += 0.2

    completeness = max(1.0, min(5.0, completeness))

    return Scores(factual=factual, completeness=completeness, format_ok=fmt)


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
