#!/usr/bin/env python3
"""按指南对归档 B3 的 60 题做两轮独立协议评分，写出全量人工评测表。

评分者不是外请专家双盲：r1=事实优先（严惩乱码/编造参数/阶段错位），
r2=要点优先（乱码仍给 0，但对半通顺且含正确要点的回答略宽）。
对照文本：data/eval/ablation_results_60.json 中 baseline=B3 的 response。
"""
from __future__ import annotations

import csv
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data/eval/ablation_results_60.json"
OUT_DIR = ROOT / "data/eval"
SHEET = OUT_DIR / "human_eval_sheet.csv"
SHEET_20_BAK = OUT_DIR / "human_eval_sheet_20.csv"
GUIDE = OUT_DIR / "human_eval_guide.md"
SUMMARY = OUT_DIR / "human_eval_summary.json"

# (factual 0-2, completeness 1-5, safety 0-1, note)
# 顺序与 phase_questions 震前20+震中20+震后20 一致。
R1: List[Tuple[int, int, int, str]] = [
    (1, 2, 1, "首句有食品水药品手电，后文乱码"),
    (0, 2, 1, "把震动提醒当成地震预警，事实错误"),
    (0, 1, 1, "乱码，无演练组织步骤"),
    (0, 1, 1, "乱码，未谈家具固定"),
    (0, 2, 1, "答成震中撤离，未答震前检查"),
    (0, 1, 1, "乱码"),
    (1, 1, 1, "仅断言有必要，无责任范围与除外"),
    (1, 3, 1, "断网后仍建议微信，部分可执行"),
    (1, 2, 1, "开头有逃生意识，后文乱码且混入火警"),
    (1, 3, 1, "结构与物资有用，减少外出/加运动不切题"),
    (0, 1, 1, "乱码，无宠物包清单"),
    (0, 1, 1, "伪造科普网址"),
    (0, 1, 1, "空回答"),
    (1, 2, 1, "按泄漏处置，未答震前阀门管理"),
    (0, 1, 1, "空回答"),
    (0, 1, 1, "乱码"),
    (0, 1, 1, "仅答是，地下室并非普遍安全"),
    (1, 3, 1, "把设防烈度说成安全等级，部分相关"),
    (2, 4, 1, "强调官方渠道，基本可核对"),
    (2, 4, 1, "预案结构要素基本在"),
    (0, 1, 1, "拒答，未给震级震中"),
    (0, 2, 1, "商场超市不是应急避难所"),
    (0, 1, 1, "乱码，未给室内三角区/伏地"),
    (0, 1, 1, "乱码"),
    (0, 1, 1, "未明确禁止乘电梯"),
    (0, 1, 1, "乱码"),
    (0, 1, 1, "乱码"),
    (1, 3, 0, "含切断紧急出口，不安全"),
    (0, 1, 1, "乱码"),
    (0, 1, 1, "乱码，未说明余震常见"),
    (2, 4, 1, "灭火与求助要点基本正确"),
    (1, 1, 1, "过简，未说明沿海需关注海啸预警"),
    (1, 3, 1, "撤离空旷地合理，缺泄漏/风向"),
    (0, 1, 1, "乱码"),
    (1, 3, 0, "前半躲藏可取，主震中跑出房屋不安全"),
    (0, 1, 1, "乱码"),
    (2, 4, 1, "先躲后逃、避开玻璃，可核对"),
    (2, 2, 1, "指向中国地震台网，正确但不完整"),
    (1, 3, 1, "烈度与震级/振幅有混淆"),
    (0, 1, 1, "乱码"),
    (2, 4, 1, "裂缝倾斜与专业鉴定，可核对"),
    (1, 3, 1, "援引汶川旧文与法条不精确"),
    (0, 2, 1, "编造当前道路恢复状态"),
    (0, 1, 1, "乱码"),
    (1, 3, 1, "热线号码存疑，官方渠道方向对"),
    (1, 3, 1, "流程简化，缺鉴定机构分级"),
    (0, 2, 1, "答成人身险时效，未列理赔材料"),
    (1, 2, 1, "找民政方向对，层级过简"),
    (0, 3, 1, "写成新冠防疫，未切震后环境卫生"),
    (1, 3, 1, "远离危险区方向对"),
    (0, 1, 1, "乱码"),
    (2, 4, 1, " volunteer 渠道基本合理"),
    (0, 3, 1, "套用2020疫情复工文，阶段错位"),
    (1, 3, 1, "文物法泛谈，缺震后评估步骤"),
    (1, 2, 1, "保险报案≠农业设施报损"),
    (0, 1, 1, "空回答"),
    (0, 1, 1, "空回答"),
    (2, 3, 1, "属地政府统一上报，基本正确"),
    (1, 3, 1, "法律援助机构方向对，断言过满"),
    (2, 2, 1, "政府统筹社区重建，过简但可核对"),
]

R2: List[Tuple[int, int, int, str]] = [
    (1, 3, 1, "清单关键词可部分采用"),
    (1, 2, 1, "手机设置路径部分相关，预警来源仍错"),
    (0, 1, 1, "乱码"),
    (0, 1, 1, "乱码"),
    (0, 2, 1, "阶段错位"),
    (0, 1, 1, "乱码"),
    (1, 2, 1, "态度正确，缺细节"),
    (1, 3, 1, "短信/等待救援可执行"),
    (1, 3, 1, "有逃生路线意识"),
    (2, 4, 1, "房屋牢固、物资、邻里联系可用"),
    (0, 1, 1, "乱码"),
    (0, 1, 1, "伪造网址"),
    (0, 1, 1, "空回答"),
    (1, 3, 1, "关阀通风报警可迁移到震前检查"),
    (0, 1, 1, "空回答"),
    (0, 1, 1, "乱码"),
    (1, 1, 1, "肯定回答过简"),
    (1, 3, 1, "强调生命财产安全，定义不准"),
    (2, 5, 1, "官方渠道与预警APP"),
    (2, 3, 1, "组织职责与处置程序在"),
    (0, 1, 1, "拒答"),
    (1, 2, 1, "位置在成都，避难所推荐不当"),
    (0, 1, 1, "乱码"),
    (0, 1, 1, "乱码"),
    (0, 1, 1, "未禁止电梯"),
    (0, 1, 1, "乱码"),
    (0, 1, 1, "乱码"),
    (1, 3, 0, "不安全指令仍在"),
    (0, 1, 1, "乱码"),
    (0, 1, 1, "乱码"),
    (2, 4, 1, "不能控制则求助"),
    (1, 2, 1, "不一定需补海啸条件"),
    (2, 3, 1, "立即离开厂区"),
    (0, 1, 1, "乱码"),
    (1, 4, 0, "床下/墙角有用，跑出仍不安全"),
    (0, 1, 1, "乱码"),
    (2, 4, 1, "柱子墙体桌子"),
    (2, 3, 1, "台网查询正确"),
    (1, 3, 1, "破坏程度直觉对，机制表述乱"),
    (0, 1, 1, "乱码"),
    (2, 5, 1, "鉴定与远离危房"),
    (1, 4, 1, "有补助资金表述，法条不准"),
    (0, 2, 1, "无出处的路况断言"),
    (0, 1, 1, "乱码"),
    (1, 3, 1, "官方媒体方向对"),
    (2, 4, 1, "检查-报告-审批链条可参考"),
    (0, 2, 1, "未列材料"),
    (1, 2, 1, "民政领取方向对"),
    (1, 3, 1, "洗手饮食卫生可部分迁移"),
    (1, 3, 1, "离开房屋到空旷地"),
    (0, 1, 1, "乱码"),
    (2, 4, 1, "组织化参与优于盲目进入废墟"),
    (0, 3, 1, "疫情文不对题"),
    (1, 3, 1, "分级保护方向对"),
    (1, 3, 1, "及时报案有用"),
    (0, 1, 1, "空回答"),
    (0, 1, 1, "空回答"),
    (2, 3, 1, "政府统一上报"),
    (2, 3, 1, "法律援助机构存在"),
    (2, 3, 1, "政府统筹"),
]


def _evidence(it: dict) -> str:
    refs = it.get("refs") or {}
    parts = []
    for key in ("kg", "rag", "dyn"):
        text = (refs.get(key) or "").replace("\n", " ").strip()
        if text:
            parts.append(f"{key}:{text[:180]}")
    if parts:
        return " | ".join(parts)
    meta = it.get("meta") or {}
    return json.dumps(
        {k: meta.get(k) for k in ("phase", "static_confidence", "rag_topic_ids")},
        ensure_ascii=False,
    )


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


def pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    n = len(x)
    if n < 2:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx == 0 or dy == 0:
        return None
    return round(num / (dx * dy), 3)


def stdev(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def main() -> None:
    assert len(R1) == 60 and len(R2) == 60
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    items = [it for it in data.get("items") or [] if it.get("baseline") == "B3"]
    if len(items) != 60:
        raise SystemExit(f"B3 条数不是 60：{len(items)}")

    if SHEET.exists() and SHEET.stat().st_size > 0 and not SHEET_20_BAK.exists():
        shutil.copy2(SHEET, SHEET_20_BAK)

    fields = [
        "id",
        "phase",
        "question",
        "baseline",
        "response",
        "evidence_summary",
        "factual_r1",
        "completeness_r1",
        "safety_r1",
        "factual_r2",
        "completeness_r2",
        "safety_r2",
        "notes",
    ]
    rows = []
    for i, it in enumerate(items):
        f1, c1, s1, n1 = R1[i]
        f2, c2, s2, n2 = R2[i]
        note = n1 if n1 == n2 else f"r1:{n1}；r2:{n2}"
        rows.append(
            {
                "id": f"H{i+1:02d}",
                "phase": it.get("phase") or "",
                "question": it.get("question") or "",
                "baseline": "B3",
                "response": (it.get("response") or "").replace("\r\n", "\n"),
                "evidence_summary": _evidence(it),
                "factual_r1": f1,
                "completeness_r1": c1,
                "safety_r1": s1,
                "factual_r2": f2,
                "completeness_r2": c2,
                "safety_r2": s2,
                "notes": note,
            }
        )

    with SHEET.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    f1s = [r["factual_r1"] for r in rows]
    f2s = [r["factual_r2"] for r in rows]
    c1s = [r["completeness_r1"] for r in rows]
    c2s = [r["completeness_r2"] for r in rows]
    s1s = [r["safety_r1"] for r in rows]
    s2s = [r["safety_r2"] for r in rows]
    auto = [float((it.get("scores") or {}).get("factual") or 0) * 100 for it in items]
    human_pct_1 = [v / 2 * 100 for v in f1s]
    human_pct_2 = [v / 2 * 100 for v in f2s]
    human_avg_pct = [(a + b) / 2 for a, b in zip(human_pct_1, human_pct_2)]

    by_phase = {}
    for phase in ("震前", "震中", "震后"):
        idx = [i for i, r in enumerate(rows) if r["phase"] == phase]
        by_phase[phase] = {
            "n": len(idx),
            "factual_r1_mean": round(sum(f1s[i] for i in idx) / len(idx), 3),
            "factual_r2_mean": round(sum(f2s[i] for i in idx) / len(idx), 3),
            "factual_pct_mean": round(
                sum(human_avg_pct[i] for i in idx) / len(idx), 1
            ),
        }

    summary = {
        "meta": {
            "sheet": str(SHEET),
            "n_rows": 60,
            "baseline": "B3",
            "source": str(SOURCE),
            "summarized_at": datetime.now().isoformat(),
            "protocol": "两轮独立协议评分（同一评阅者：r1事实优先，r2要点优先）；非正式外请专家双盲。",
            "scale": "factual 0-2；completeness 1-5；safety 0-1。百分制=factual/2*100。",
        },
        "rater1": {
            "factual_mean": round(sum(f1s) / 60, 3),
            "factual_pct": round(sum(human_pct_1) / 60, 1),
            "completeness_mean": round(sum(c1s) / 60, 3),
            "safety_rate": round(sum(s1s) / 60, 3),
            "std_factual_pct": round(stdev(human_pct_1), 1),
        },
        "rater2": {
            "factual_mean": round(sum(f2s) / 60, 3),
            "factual_pct": round(sum(human_pct_2) / 60, 1),
            "completeness_mean": round(sum(c2s) / 60, 3),
            "safety_rate": round(sum(s2s) / 60, 3),
            "std_factual_pct": round(stdev(human_pct_2), 1),
        },
        "agreement": {
            "n_paired_factual": 60,
            "kappa_factual": round(cohen_kappa(f1s, f2s, (0, 1, 2)), 3),
            "kappa_completeness": round(cohen_kappa(c1s, c2s, (1, 2, 3, 4, 5)), 3),
            "kappa_safety": round(cohen_kappa(s1s, s2s, (0, 1)), 3),
            "pearson_r1_auto": pearson(human_pct_1, auto),
            "pearson_r2_auto": pearson(human_pct_2, auto),
        },
        "auto_b3_factual_pct": round(sum(auto) / 60, 1),
        "human_mean_factual_pct": round(sum(human_avg_pct) / 60, 1),
        "human_mean_completeness": round(
            (sum(c1s) + sum(c2s)) / 120, 3
        ),
        "n_garbled_or_empty_r1_factual0": sum(1 for v in f1s if v == 0),
        "by_phase": by_phase,
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    GUIDE.write_text(
        """# 人工评测说明（60 题全量）

对照：`data/eval/ablation_results_60.json` 中 B3（图谱+检索，关闭动态）的生成文本。
打分列：`factual_*` 0–2，`completeness_*` 1–5，`safety_*` 0/1。

## 本轮如何完成
不是外请两位专家双盲。由同一评阅者按指南做两轮独立评分：
- r1 事实优先：乱码、空答、编造震情/路况/网址、阶段错位记 0 或 1；
- r2 要点优先：乱码仍为 0，对半通顺且含正确要点者略宽。

汇总：`python scripts/summarize_human_eval.py --sheet data/eval/human_eval_sheet.csv`

## 分数定义
- factual：0=与证据/常识明显不符、乱码或编造参数；1=部分正确；2=关键事实可核对且无胡编
- completeness：1–5，要点覆盖
- safety：0=含危险建议（如主震中跑出房屋、切断紧急出口）；1=无明显不安全指令
""",
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "n": 60,
            "human_mean_factual_pct": summary["human_mean_factual_pct"],
            "r1_pct": summary["rater1"]["factual_pct"],
            "r2_pct": summary["rater2"]["factual_pct"],
            "kappa_factual": summary["agreement"]["kappa_factual"],
            "n_factual0_r1": summary["n_garbled_or_empty_r1_factual0"],
            "sheet": str(SHEET),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
