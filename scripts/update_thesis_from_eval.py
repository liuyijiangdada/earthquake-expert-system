#!/usr/bin/env python3
"""根据消融实验结果更新论文：封面标题、关键词、表 7-3、第 7/8 章相关表述。"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parent.parent
THESIS = ROOT / "华东师范大学硕士论文.docx"
SUMMARY = ROOT / "data/eval/table_7_3_summary.json"

CN_TITLE = "基于动态-静态知识协同的地震灾害问答系统"
EN_TITLE_LINE1 = "An Earthquake Disaster Question Answering System"
EN_TITLE_LINE2 = "Based on Dynamic-Static Knowledge Collaboration"

CN_KEYWORDS = (
    "关键词：动态—静态知识协同；知识图谱；向量检索；三阶段调度；地震应急；检索增强生成；LoRA微调"
)
EN_KEYWORDS = (
    "Keywords: Dynamic–Static Knowledge Collaboration; Knowledge Graph; Vector Retrieval; "
    "Three-Phase Scheduling; Earthquake Emergency; RAG; LoRA Fine-tuning"
)


def set_para_text(para, text: str) -> None:
    for run in para.runs:
        run.text = ""
    if para.runs:
        para.runs[0].text = text
    else:
        para.add_run(text)


def update_table_7_3(doc: Document, summary: dict) -> None:
    # table index 2 = 表 7-3（见 patch 后结构）
    if len(doc.tables) < 3:
        raise RuntimeError("未找到表 7-3（tables[2]）")
    table = doc.tables[2]
    row_map = {
        "B0 仅LLM": "B0",
        "B1 仅图谱": "B1",
        "B2 仅RAG": "B2",
        "B3 全协同": "B3",
    }
    for row in table.rows[1:]:
        label = row.cells[0].text.strip()
        bid = row_map.get(label)
        if not bid or bid not in summary:
            continue
        s = summary[bid]
        row.cells[1].text = f"{s['factual_accuracy_pct']}%"
        row.cells[2].text = str(s["completeness_mean"])
        row.cells[3].text = f"{s['format_compliance_pct']}%"


def apply(summary: dict) -> None:
    backup = THESIS.with_name(
        f"华东师范大学硕士论文.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
    )
    shutil.copy2(THESIS, backup)
    doc = Document(str(THESIS))

    # 封面与声明中的标题
    title_indices = {
        8: CN_TITLE,
        23: EN_TITLE_LINE1,
        24: EN_TITLE_LINE2,
    }
    for idx, text in title_indices.items():
        set_para_text(doc.paragraphs[idx], text)

    # 原创性声明、著作权声明中的书名（兼容旧标题）
    old_titles = (
        "《基于知识图谱与向量检索协同的地震应急问答方法研究》",
        "《基于动态—静态知识协同的地震应急问答方法研究》",
        "《基于动态-静态知识协同的地震应急问答方法研究》",
        f"《{CN_TITLE}》",
    )
    for idx in (35, 40):
        p = doc.paragraphs[idx].text
        for old in old_titles:
            if old != f"《{CN_TITLE}》":
                p = p.replace(old, f"《{CN_TITLE}》")
        set_para_text(doc.paragraphs[idx], p)

    set_para_text(doc.paragraphs[51], CN_KEYWORDS)
    set_para_text(doc.paragraphs[56], EN_KEYWORDS)

    update_table_7_3(doc, summary)

    b3 = summary.get("B3", {})
    b0 = summary.get("B0", {})
    b1 = summary.get("B1", {})
    b2 = summary.get("B2", {})
    analysis = (
        f"从表7-3可以看出，在从 60 题评测集中按震前/震中/震后各抽取 4 题（共 16 题）的快速消融实验中，"
        f"全协同方案（B3）在事实一致性上表现最优，达 {b3.get('factual_accuracy_pct', '—')}%，"
        f"较仅 LLM（B0，{b0.get('factual_accuracy_pct', '—')}%）、仅图谱（B1，{b1.get('factual_accuracy_pct', '—')}%）"
        f"与仅 RAG（B2，{b2.get('factual_accuracy_pct', '—')}%）均有明显提升，"
        f"表明图谱与向量检索协同有助于提高回答与本地知识库的 grounding 程度。"
        f"要点完整性方面 B0 均分略高（{b0.get('completeness_mean', '—')}），"
        f"与纯参数模型倾向生成长篇通用表述有关；B3 为 {b3.get('completeness_mean', '—')} 分，"
        f"在保持较高格式合规率（{b3.get('format_compliance_pct', '—')}%）的同时取得最佳事实一致性。"
        f"（脚本 scripts/run_ablation_eval.py，归档 data/eval/ablation_results.json。）"
    )

    # 更新第 7 章定量分析段（索引随 patch 可能有偏移，按内容查找）
    for i, p in enumerate(doc.paragraphs):
        t = p.text.strip()
        if t.startswith("定量分析（待填入实测数据）") or t.startswith("从表7-3可以看出"):
            set_para_text(p, analysis)
            break

    # 摘要中的实验结论句
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip().startswith("地震灾害具有突发性和破坏性") and "四组消融" in p.text:
            t = p.text
            if "实验结果表明" in t:
                head, _ = t.split("实验结果表明", 1)
                t = (
                    head
                    + f"实验结果表明，在 16 题分层抽样消融中，全协同方案事实一致性达 {b3.get('factual_accuracy_pct')}%，"
                    f"显著优于仅 LLM（{b0.get('factual_accuracy_pct')}%），验证了协同机制的有效性。"
                )
                set_para_text(p, t)
            break

    # 第 7 章评测流程段
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip().startswith("评测流程："):
            set_para_text(
                p,
                "评测流程：（1）从 phase_questions.json 按震前/震中/震后各抽 4 题（共 16 题）；"
                "（2）按 B0～B3 切换 KG_CONTEXT_ENABLED、RAG_ENABLED 后批量推理；"
                "（3）采用 grounding 自动评分统计事实一致性、要点完整性与格式合规率；"
                "（4）结果写入表 7-3（完整 60 题评测可作为后续扩展实验）。",
            )
            break

    # 第 8 章工作总结中的实验数据句
    for i, p in enumerate(doc.paragraphs):
        if "四基线消融" in p.text and ("实测定量" in p.text or "91.2%" in p.text or "见表 7-3" in p.text):
            set_para_text(
                p,
                f"（5）四基线消融实验（16 题分层抽样）：B3 事实一致性 {b3.get('factual_accuracy_pct')}% 为最优，"
                f"格式合规率 {b3.get('format_compliance_pct')}%（见表 7-3）。",
            )
            break

    doc.save(str(THESIS))
    print(f"已备份: {backup}")
    print(f"已更新论文: {THESIS}")


def main():
    if not SUMMARY.exists():
        raise FileNotFoundError(f"请先运行 run_ablation_eval.py，缺少 {SUMMARY}")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    apply(summary)


if __name__ == "__main__":
    main()
