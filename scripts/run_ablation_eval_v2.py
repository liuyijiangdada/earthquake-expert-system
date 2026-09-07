#!/usr/bin/env python3
"""扩展消融：B0–B3 + B3-ns（无阶段调度）+ B4（仅动态）+ B5（全源+动态）。

默认走冻结动态快照、归档 prompt 证据块与调度决策。
  --skip-llm     只组装上下文，不加载大模型
  --reuse-responses PATH  复用已有回答（按 baseline+question 匹配）
  --during-only  仅震中子集（适合先跑 B4/B5）
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("RAG_USE_MEMORY_RAG", "1")

from core.dynamic_snapshot import DEFAULT_SNAPSHOT_PATH
from core.eval_runtime import (
    BASELINE_SPECS,
    apply_baseline_to_stack,
    build_eval_stack,
    decision_from_meta,
    model_snapshot,
    spec_by_id,
)
from core.eval_scoring import (
    dual_to_dict,
    extract_refs,
    score_both,
    truncate_ref,
)

DEFAULT_QUESTIONS = ROOT / "data/eval/phase_questions.json"


def load_questions(path: Path, *, per_phase: int, during_only: bool) -> List[Tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: List[Tuple[str, str]] = []
    for phase, qs in data.get("phases", {}).items():
        if during_only and phase != "震中":
            continue
        for q in qs[:per_phase]:
            out.append((phase, q))
    return out


def load_reuse_map(path: Optional[Path]) -> Dict[Tuple[str, str], str]:
    if not path or not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping: Dict[Tuple[str, str], str] = {}
    for item in data.get("items") or []:
        bid = item.get("baseline")
        q = item.get("question")
        resp = item.get("response")
        if bid and q and resp:
            mapping[(bid, q)] = resp
    return mapping


def summarize(items: List[dict], spec_id: str, kg: bool, rag: bool, dynamic: bool) -> dict:
    rows = [it for it in items if it.get("baseline") == spec_id]

    def _mean(scheme: str, key: str, scale: float = 1.0) -> float:
        vals = []
        for it in rows:
            sc = (it.get("scores") or {}).get(scheme) or {}
            if key not in sc:
                continue
            vals.append(sc[key] * scale)
        return round(sum(vals) / len(vals), 1 if scale == 100 else 2) if vals else 0.0

    dyn_flags = [bool((it.get("decision") or {}).get("use_dynamic")) for it in rows]
    by_phase: Dict[str, List[bool]] = {}
    phase_tagged = 0
    for it in rows:
        ph = it.get("phase") or ""
        by_phase.setdefault(ph, []).append(bool((it.get("decision") or {}).get("use_dynamic")))
        if it.get("phase_tag") in ("震前", "震中", "震后"):
            phase_tagged += 1
    return {
        "n": len(rows),
        "n_scored": sum(1 for it in rows if it.get("scores")),
        "kg_context_enabled": kg,
        "rag_enabled": rag,
        "dynamic_enabled": dynamic,
        "use_dynamic_rate": round(sum(dyn_flags) / len(dyn_flags), 3) if dyn_flags else 0.0,
        "use_dynamic_rate_by_phase": {
            p: round(sum(v) / len(v), 3) if v else 0.0 for p, v in by_phase.items()
        },
        "phase_tag_rate": round(phase_tagged / len(rows), 3) if rows else 0.0,
        "legacy": {
            "factual_accuracy_pct": _mean("legacy", "factual", 100),
            "completeness_mean": _mean("legacy", "completeness"),
            "format_compliance_pct": _mean("legacy", "format_ok", 100),
        },
        "normalized": {
            "factual_accuracy_pct": _mean("normalized", "factual", 100),
            "completeness_mean": _mean("normalized", "completeness"),
            "format_compliance_pct": _mean("normalized", "format_ok", 100),
        },
    }


def _lite_sanitize(text: str, _input_text: str) -> str:
    if not text or not re.search(r"[\u4e00-\u9fff]", text):
        return "抱歉，模型生成了不相关的内容。请重新提问您关心的地震相关问题。"
    return (
        text.replace("</s>", "")
        .replace("<|im_start|>", "")
        .replace("<|im_end|>", "")
        .strip()
    )


def _load_eval_llm(config, *, no_lora: bool = False, base_model_path: str | None = None):
    """评测专用加载，不导入 app.py（避免强制连接 Neo4j）。

    no_lora=True 时跳过 LoRA 加载，用于纯基座外部对照基线（如原生 Qwen2.5-7B）。
    base_model_path 可显式指定本地基座目录（环境变量 EVAL_BASE_MODEL_PATH 亦可）。
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    model_name = getattr(config, "MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    # 显式本地路径优先：命令行 > 环境变量 > HF 缓存自动探测
    local_model = base_model_path or os.environ.get("EVAL_BASE_MODEL_PATH")
    if not local_model:
        hub = Path.home() / ".cache/huggingface/hub" / f"models--{model_name.replace('/', '--')}" / "snapshots"
        local_model = model_name
        if hub.is_dir():
            snaps = sorted(hub.iterdir())
            if snaps:
                local_model = str(snaps[-1])
    lora_rel = getattr(config, "FINETUNED_MODEL_PATH", "llm/earthquake_expert_qwen25_7b")
    lora_dir = lora_rel if os.path.isabs(lora_rel) else str(ROOT / lora_rel)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    print(f"评测加载基座 {local_model} device={device}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        local_model,
        dtype=dtype,
        device_map=None,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        local_model, trust_remote_code=False, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if no_lora:
        print("评测：--no-lora 已启用，跳过 LoRA 加载（纯基座模式）", flush=True)
        model = base
    else:
        print(f"评测加载 LoRA {lora_dir}", flush=True)
        model = PeftModel.from_pretrained(base, lora_dir, local_files_only=True)
    model.to(device)
    model.eval()
    print("评测模型加载成功", flush=True)
    return model, tokenizer, device


def _infer_with_model(model, tokenizer, device, config, prompt: str, input_text: str) -> str:
    import torch

    # 与训练端 SFTDataset 使用同一套 Qwen2.5 chat 模板，避免格式错位
    messages = [
        {"role": "system", "content": "你是一个地震专家，专注于回答地震相关问题。"},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    max_in = int(getattr(config, "LLM_INPUT_MAX_TOKENS", 4096))
    max_new = int(getattr(config, "LLM_MAX_NEW_TOKENS", 160))
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_in,
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new,
            do_sample=False,
        )
    generated = outputs[0][input_ids.shape[-1] :]
    out = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return _lite_sanitize(out, input_text)


def run_eval(
    *,
    questions_file: Path,
    output_dir: Path,
    per_phase: int = 20,
    baselines: Optional[List[str]] = None,
    skip_llm: bool = True,
    reuse_path: Optional[Path] = None,
    snapshot_path: Path = DEFAULT_SNAPSHOT_PATH,
    during_only: bool = False,
    force_json_kg: bool = False,
    skip_rag: bool = False,
    no_lora: bool = False,
    base_model_path: Optional[str] = None,
) -> dict:
    from config.config import Config

    Config.RAG_USE_MEMORY_RAG = True
    Config.LAYER3_ENABLED = False
    Config.MULTIMODAL_OUTPUT_ENABLED = False
    Config.LAYER3_INJECT_PROMPT = False
    Config.LLM_MAX_NEW_TOKENS = 160
    Config.LLM_DO_SAMPLE = False
    Config.DYNAMIC_RETRIEVAL_ENABLED = True

    specs = [spec_by_id(b) for b in (baselines or [s.id for s in BASELINE_SPECS])]
    questions = load_questions(questions_file, per_phase=per_phase, during_only=during_only)
    reuse = load_reuse_map(reuse_path)
    stack = build_eval_stack(
        Config,
        snapshot_path=snapshot_path,
        force_json_kg=force_json_kg,
        skip_rag=skip_rag,
    )

    llm_bundle = None
    if not skip_llm:
        llm_bundle = _load_eval_llm(Config, no_lora=no_lora, base_model_path=base_model_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: Dict[str, Any] = {
        "meta": {
            "started_at": datetime.now().isoformat(),
            "mode": "extended_ablation",
            "per_phase": per_phase,
            "during_only": during_only,
            "question_count": len(questions),
            "questions_file": str(questions_file),
            "baselines": [s.__dict__ for s in specs],
            "seed": 42,
            "skip_llm": skip_llm,
            "kg_backend": stack.kg_backend,
            "rag_backend": stack.rag_backend,
            "snapshot_path": str(snapshot_path),
            "reuse_responses": str(reuse_path) if reuse_path else None,
            "model": model_snapshot(Config),
            "notes": (
                "动态源走冻结快照；B3-ns 关闭阶段分类与调度；"
                "同时归档 legacy / normalized 两套自动分；禁止事后整体校准上调。"
            ),
        },
        "items": [],
        "summary": {},
    }

    total = len(specs) * len(questions)
    step = 0
    for spec in specs:
        apply_baseline_to_stack(stack, spec)
        for phase, question in questions:
            step += 1
            t0 = time.time()
            prompt, response_meta, phase_tag = stack.builder.prepare(
                question, for_vision=False, history=None
            )
            kg_ref, rag_ref, dyn_ref = extract_refs(prompt)
            response = reuse.get((spec.id, question), "")
            source = "reuse" if response else ("skip_llm" if skip_llm else "generate")
            if not response and not skip_llm and llm_bundle is not None:
                model, tokenizer, device = llm_bundle
                response = _infer_with_model(
                    model, tokenizer, device, Config, prompt, question
                )
                source = "generate"
            dual = None
            if response:
                dual = score_both(
                    response,
                    prompt,
                    response_meta,
                    phase,
                    kg_on=spec.kg,
                    rag_on=spec.rag,
                )
            elapsed = time.time() - t0
            item = {
                "baseline": spec.id,
                "phase": phase,
                "question": question,
                "response": response,
                "response_source": source,
                "scores": dual_to_dict(dual) if dual else None,
                "refs": {
                    "kg": truncate_ref(kg_ref),
                    "rag": truncate_ref(rag_ref),
                    "dyn": truncate_ref(dyn_ref),
                },
                "decision": decision_from_meta(response_meta),
                "phase_tag": phase_tag,
                "elapsed_sec": round(elapsed, 2),
            }
            all_results["items"].append(item)
            f_legacy = (item["scores"] or {}).get("legacy", {}).get("factual")
            print(
                f"[{step}/{total}] {spec.id} | {phase} | src={source} | "
                f"dyn={item['decision']['use_dynamic']} | "
                f"f_legacy={f_legacy} | {elapsed:.1f}s",
                flush=True,
            )

        all_results["summary"][spec.id] = summarize(
            all_results["items"], spec.id, spec.kg, spec.rag, spec.dynamic
        )

    all_results["meta"]["finished_at"] = datetime.now().isoformat()
    out_json = output_dir / "ablation_results_extended.json"
    out_json.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary_path = output_dir / "table_extended_summary.json"
    summary_path.write_text(
        json.dumps(all_results["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n=== 扩展消融汇总（legacy / normalized 事实一致性%） ===")
    for bid, s in all_results["summary"].items():
        print(
            f"{bid}: legacy {s['legacy']['factual_accuracy_pct']}% | "
            f"normalized {s['normalized']['factual_accuracy_pct']}% | n={s['n']}"
        )
    print(f"\n结果已保存: {out_json}")
    return all_results


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    ap.add_argument("--output", type=Path, default=ROOT / "data/eval")
    ap.add_argument("--per-phase", type=int, default=20)
    ap.add_argument(
        "--baselines",
        default=",".join(s.id for s in BASELINE_SPECS),
        help="逗号分隔，如 B3,B3-ns,B4,B5",
    )
    ap.add_argument(
        "--run-llm",
        action="store_true",
        help="真正调用本地 LoRA 生成；默认只组装上下文（可配合 --reuse-responses）",
    )
    ap.add_argument("--reuse-responses", type=Path, default=None)
    ap.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT_PATH)
    ap.add_argument("--during-only", action="store_true")
    ap.add_argument("--force-json-kg", action="store_true")
    ap.add_argument("--skip-rag", action="store_true")
    ap.add_argument(
        "--no-lora",
        action="store_true",
        help="跳过 LoRA 加载，仅用纯基座（外部对照基线，如原生 Qwen2.5-7B）",
    )
    ap.add_argument(
        "--base-model-path",
        type=str,
        default=None,
        help="显式指定本地基座模型目录（覆盖 HF 缓存自动探测；也可用环境变量 EVAL_BASE_MODEL_PATH）",
    )
    args = ap.parse_args()
    run_eval(
        questions_file=args.questions,
        output_dir=args.output,
        per_phase=args.per_phase,
        baselines=[b.strip() for b in args.baselines.split(",") if b.strip()],
        skip_llm=not args.run_llm,
        reuse_path=args.reuse_responses,
        snapshot_path=args.snapshot,
        during_only=args.during_only,
        force_json_kg=args.force_json_kg,
        skip_rag=args.skip_rag,
        no_lora=args.no_lora,
        base_model_path=args.base_model_path,
    )


if __name__ == "__main__":
    main()
