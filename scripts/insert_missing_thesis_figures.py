#!/usr/bin/env python3
"""插入缺失方法图：图4-1 图谱模式、图4-2 RAG 流程、图5-13 分阶段结果；并把原图4-1 端到端改为图4-3。"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
THESIS = ROOT / "华东师范大学硕士论文.docx"
BACKUP = ROOT / "华东师范大学硕士论文.docx.bak-before-insert-missing-figs"
ARCH = ROOT / "docs" / "superpowers" / "architecture"
SUMMARY = ROOT / "data" / "eval" / "table_6_3_summary_60.json"

sys.path.insert(0, str(ROOT / "scripts"))
import generate_kg_schema_png as kg  # noqa: E402
import generate_rag_pipeline_flow as rag  # noqa: E402
import thicken_thesis_from_gap_plan as phase  # noqa: E402


def _set_run_font(run, *, size_pt: float = 10.5, bold: bool = False) -> None:
    run.font.name = "Times New Roman"
    run.font.size = Pt(size_pt)
    run.font.bold = bold
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = rPr.makeelement(qn("w:rFonts"), {})
        rPr.insert(0, rFonts)
    rFonts.set(qn("w:eastAsia"), "宋体")


def find_body_para(doc: Document, pred, *, start_after: int = 200) -> int:
    for i, p in enumerate(doc.paragraphs):
        if i <= start_after:
            continue
        if pred(p.text.strip()):
            return i
    raise KeyError(pred)


def insert_figure_block(doc: Document, after_idx: int, image: Path, caption: str, width_in: float) -> None:
    """在 after_idx 段落后插入：图片段 + 题注段 + 说明段。"""
    anchor = doc.paragraphs[after_idx]
    # insert in reverse order via addnext chain: first insert note, then caption, then image
    # easier: insert_paragraph_before on the NEXT paragraph repeatedly
    next_p = doc.paragraphs[after_idx + 1]

    img_p = next_p.insert_paragraph_before("")
    img_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    img_p.paragraph_format.first_line_indent = Cm(0)
    run = img_p.add_run()
    run.add_picture(str(image), width=Inches(width_in))

    cap_p = next_p.insert_paragraph_before(caption)
    cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap_p.paragraph_format.first_line_indent = Cm(0)
    for r in cap_p.runs:
        _set_run_font(r, size_pt=10.5, bold=False)

    print(f"  已插入 {caption} <- {image.name}")


def replace_text_everywhere(doc: Document, mapping: list[tuple[str, str]], *, body_only: bool = False) -> int:
    n = 0
    for p in doc.paragraphs:
        t = p.text
        if body_only and "\t" in t and t.strip().startswith("图"):
            continue
        new = t
        for a, b in mapping:
            if a in new:
                new = new.replace(a, b)
        if new != t:
            # preserve by rewriting runs simply
            if p.runs:
                p.runs[0].text = new
                for r in p.runs[1:]:
                    r.text = ""
            else:
                p.add_run(new)
            n += 1
    return n


def update_toc_figures(doc: Document) -> None:
    """更新图目录条目（页码暂用 —，打开 Word 后可更新域）。"""
    # find TOC figure block
    toc_start = None
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip() == "图目录":
            toc_start = i
            break
    if toc_start is None:
        print("  未找到图目录，跳过 TOC 更新")
        return

    # collect existing TOC figure paras until 表目录 or empty section
    toc_figs = []
    for i in range(toc_start + 1, min(toc_start + 40, len(doc.paragraphs))):
        t = doc.paragraphs[i].text.strip()
        if t.startswith("表目录") or t.startswith("表"):
            break
        if t.startswith("图"):
            toc_figs.append(i)

    desired = [
        "图3-1  总体方法框架图\t—",
        "图4-1  知识图谱实体关系模式\t—",
        "图4-2  主题级向量检索流程\t—",
        "图4-3  端到端推理流程\t—",
        "图5-1  系统总体逻辑架构图\t—",
        "图5-2  问答请求数据流图\t—",
        "图5-3  技术栈与模块映射关系\t—",
        "图5-4  Web端问答主界面示例\t—",
        "图5-5  分阶段快捷提问与回答展示示例\t—",
        "图5-6  移动端问答界面示例（其一）\t—",
        "图5-7  移动端问答界面示例（其二）\t—",
        "图5-8  移动端证据/资源展示界面示例\t—",
        "图5-9  多模态资源卡片展示示例（地图/示意图等）\t—",
        "图5-10  移动端补充交互界面示例\t—",
        "图5-11  四基线事实一致性对比（60题离线消融）\t—",
        "图5-12  四基线要点完整性与格式合规对比（60题离线消融）\t—",
        "图5-13  分阶段事实一致性对比（各阶段20题）\t—",
    ]

    # rewrite first existing lines; insert extras before 表目录
    # Find 表目录
    table_toc = None
    for i in range(toc_start + 1, min(toc_start + 50, len(doc.paragraphs))):
        if doc.paragraphs[i].text.strip().startswith("表目录"):
            table_toc = i
            break

    # Clear old figure TOC lines content and rebuild
    # Strategy: update existing toc_figs in place up to len, then insert remaining before table_toc
    for idx, line in enumerate(desired):
        if idx < len(toc_figs):
            p = doc.paragraphs[toc_figs[idx]]
            if p.runs:
                p.runs[0].text = line
                for r in p.runs[1:]:
                    r.text = ""
            else:
                p.add_run(line)
        else:
            if table_toc is None:
                break
            # insert before 表目录; after each insert, table_toc index shifts — use insert_paragraph_before
            anchor = doc.paragraphs[table_toc]
            np = anchor.insert_paragraph_before(line)
            # style like previous toc line if possible
            if toc_figs:
                np.style = doc.paragraphs[toc_figs[0]].style
            table_toc += 1

    # if more old lines than desired, blank extras
    for j in range(len(desired), len(toc_figs)):
        p = doc.paragraphs[toc_figs[j]]
        if p.runs:
            p.runs[0].text = ""
            for r in p.runs[1:]:
                r.text = ""

    print("  已更新图目录条目（页码为 —，请在 Word 中更新域）")


def main() -> None:
    shutil.copy2(THESIS, BACKUP)
    print(f"备份 -> {BACKUP.name}")

    print("生成图片…")
    kg_png = kg.generate_png()
    kg.generate_svg()
    rag_png = rag.generate_png()
    rag.generate_svg()
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    phase.generate_fig_5_13(data)
    phase_png = ARCH / "fig-5-13-phase-factual-60.png"
    print(f"  KG: {kg_png.name}")
    print(f"  RAG: {rag_png.name}")
    print(f"  Phase: {phase_png.name}")

    doc = Document(str(THESIS))

    print("重编号：原图4-1 端到端 → 图4-3 …")
    # Order matters: first rename old e2e caption/refs carefully
    # Body caption
    for p in doc.paragraphs:
        t = p.text.strip()
        if t == "图4-1  端到端推理流程" or t.startswith("图4-1  端到端"):
            if p.runs:
                p.runs[0].text = "图4-3  端到端推理流程"
                for r in p.runs[1:]:
                    r.text = ""
            else:
                p.add_run("图4-3  端到端推理流程")

    # Text refs that meant e2e flow (not yet schema). Do specific phrases first.
    e2e_ref_map = [
        ("端到端推理流程示意见图4-1", "端到端推理流程示意见图4-3"),
        ("图4-1 给出端到端推理流程示意", "图4-3 给出端到端推理流程示意"),
        ("并以图4-1对照实现路径", "并以图4-3对照实现路径"),
        ("（参见图4-1）", "（参见图4-3）"),
        ("见图4-1占位", "见图4-3"),
    ]
    n = replace_text_everywhere(doc, e2e_ref_map)
    print(f"  已改正文引用 {n} 处")

    print("插入图4-1 图谱模式…")
    i_schema = find_body_para(doc, lambda t: t.startswith("知识图谱采用属性图模型"))
    # add pointer sentence if missing
    p = doc.paragraphs[i_schema]
    if "图4-1" not in p.text:
        if p.runs:
            p.runs[0].text = p.text.rstrip("。") + "。图谱实体关系模式如图4-1所示。"
            for r in p.runs[1:]:
                r.text = ""
    insert_figure_block(doc, i_schema, kg_png, "图4-1  知识图谱实体关系模式", 5.6)

    print("插入图4-2 RAG 流程…")
    i_rag = find_body_para(doc, lambda t: t.startswith("预计算阶段对全部主题块进行中文句向量编码"))
    p = doc.paragraphs[i_rag]
    if "图4-2" not in p.text:
        if p.runs:
            p.runs[0].text = p.text.rstrip("。") + "。主题级向量检索流程如图4-2所示。"
            for r in p.runs[1:]:
                r.text = ""
    insert_figure_block(doc, i_rag, rag_png, "图4-2  主题级向量检索流程", 5.8)

    print("插入图5-13 分阶段事实一致性…")
    # after 表5-4 discussion paragraph starting 由表5-4可见
    i_phase = find_body_para(doc, lambda t: t.startswith("由表5-4可见"))
    p = doc.paragraphs[i_phase]
    if "图5-13" not in p.text:
        if p.runs:
            p.runs[0].text = p.text.rstrip("。") + "。分阶段事实一致性对比见图5-13。"
            for r in p.runs[1:]:
                r.text = ""
    insert_figure_block(
        doc, i_phase, phase_png, "图5-13  分阶段事实一致性对比（各阶段20题）", 5.6
    )

    print("更新图目录…")
    update_toc_figures(doc)

    doc.save(str(THESIS))
    print(f"已保存 -> {THESIS.name}")


if __name__ == "__main__":
    main()
