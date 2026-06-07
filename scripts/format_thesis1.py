#!/usr/bin/env python3
"""根据华东师范大学硕士论文格式要求，调整「华东师范大学硕士论文1.docx」的版式。

主要修正：
  1. 页面边距：上2.54cm、下2.54cm、左3.17cm、右3.17cm
  2. 封面区域：恢复原始居中排版与字号
  3. 声明区域：标题居中黑体，正文无首行缩进
  4. 章标题（第X章、摘要、ABSTRACT、参考文献、致谢等）：黑体 三号(16pt) 居中 加粗
  5. 节标题（X.X）：黑体 四号(14pt) 左对齐 加粗
  6. 小节标题（X.X.X）：黑体 小四(12pt) 左对齐 加粗
  7. 正文：宋体 小四(12pt) 两端对齐 首行缩进2字符 1.5倍行距
  8. 图题：宋体 五号(10.5pt) 居中
  9. 表题：黑体 小四(12pt) 居中 加粗
 10. 关键词：黑体 小四(12pt) 左对齐 加粗
 11. 参考文献：宋体 五号(10.5pt)
 12. 附录/科研情况：标题居中黑体加粗
 13. 签名行：右对齐
 14. 修复由 patch 脚本插入但未设格式的段落
"""
from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parent.parent
THESIS = ROOT / "华东师范大学硕士论文1.docx"


# ── 工具函数 ──────────────────────────────────────────────

def _set_run_font(
    run,
    *,
    east_asia: str = "宋体",
    western: str = "Times New Roman",
    size_pt: float = 12,
    bold: bool = False,
):
    """设置 run 的中英文字体、字号与粗体。"""
    run.font.name = western
    run.font.size = Pt(size_pt)
    run.font.bold = bold
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = rPr.makeelement(qn("w:rFonts"), {})
        rPr.insert(0, rFonts)
    rFonts.set(qn("w:eastAsia"), east_asia)


def _ensure_run(para):
    """确保段落至少有一个 run，返回第一个 run。"""
    if not para.runs:
        para.add_run(para.text or "")
    return para.runs[0]


# ── 封面区域 ──────────────────────────────────────────────

# 封面段落索引与格式定义（基于原始文档结构）
COVER_CN_FORMAT = {
    # 中文封面
    0:  {"ea": "黑体", "size": 14, "bold": True, "align": "CENTER"},   # 2026届...
    1:  {"ea": "宋体", "size": 12, "bold": False, "align": "CENTER"},  # 分类号...
    2:  {"ea": "宋体", "size": 12, "bold": False, "align": "CENTER"},  # 密级...
    4:  {"ea": "黑体", "size": 18, "bold": True, "align": "CENTER"},   # East China Normal University
    5:  {"ea": "黑体", "size": 16, "bold": True, "align": "CENTER"},   # 硕士专业学位论文
    6:  {"ea": "Times New Roman", "size": 12, "bold": True, "align": "CENTER"},  # Master's Degree...
    8:  {"ea": "黑体", "size": 22, "bold": True, "align": "CENTER"},   # 中文论文标题
    10: {"ea": "宋体", "size": 12, "bold": False, "align": "CENTER"},  # 院系
    11: {"ea": "宋体", "size": 12, "bold": False, "align": "CENTER"},  # 专业学位类别
    12: {"ea": "宋体", "size": 12, "bold": False, "align": "CENTER"},  # 专业学位领域
    13: {"ea": "宋体", "size": 12, "bold": False, "align": "CENTER"},  # 指导教师
    14: {"ea": "宋体", "size": 12, "bold": False, "align": "CENTER"},  # 学位申请人
    16: {"ea": "宋体", "size": 12, "bold": False, "align": "CENTER"},  # 日期
}

COVER_EN_FORMAT = {
    18: {"ea": "Times New Roman", "size": 12, "bold": False, "align": "CENTER"},  # Thesis for...
    19: {"ea": "Times New Roman", "size": 12, "bold": False, "align": "CENTER"},  # University code
    20: {"ea": "Times New Roman", "size": 12, "bold": False, "align": "CENTER"},  # Student ID
    21: {"ea": "Times New Roman", "size": 18, "bold": True, "align": "CENTER"},   # East China Normal University
    23: {"ea": "Times New Roman", "size": 16, "bold": True, "align": "CENTER"},   # 英文标题1
    24: {"ea": "Times New Roman", "size": 16, "bold": True, "align": "CENTER"},   # 英文标题2
    26: {"ea": "Times New Roman", "size": 12, "bold": False, "align": "CENTER"},  # Department
    27: {"ea": "Times New Roman", "size": 12, "bold": False, "align": "CENTER"},  # Category
    28: {"ea": "Times New Roman", "size": 12, "bold": False, "align": "CENTER"},  # Field
    29: {"ea": "Times New Roman", "size": 12, "bold": False, "align": "CENTER"},  # Supervisor
    30: {"ea": "Times New Roman", "size": 12, "bold": False, "align": "CENTER"},  # Candidate
    32: {"ea": "Times New Roman", "size": 12, "bold": False, "align": "CENTER"},  # March, 2026
}

# 封面段落索引集合（这些段落不参与通用分类）
COVER_INDICES = set(COVER_CN_FORMAT.keys()) | set(COVER_EN_FORMAT.keys()) | {3, 7, 9, 15, 17, 22, 25, 31}


def fix_cover(doc: Document) -> None:
    """恢复封面区域格式。"""
    all_formats = {**COVER_CN_FORMAT, **COVER_EN_FORMAT}
    for idx, fmt in all_formats.items():
        if idx >= len(doc.paragraphs):
            continue
        p = doc.paragraphs[idx]
        if not p.text.strip():
            continue
        _ensure_run(p)
        # 设置对齐
        align_map = {"CENTER": WD_ALIGN_PARAGRAPH.CENTER, "LEFT": WD_ALIGN_PARAGRAPH.LEFT}
        p.alignment = align_map.get(fmt["align"], WD_ALIGN_PARAGRAPH.CENTER)
        pf = p.paragraph_format
        pf.first_line_indent = Pt(0)
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        # 设置字体
        ea = fmt["ea"]
        western = "Times New Roman" if ea == "Times New Roman" else "Times New Roman"
        for run in p.runs:
            _set_run_font(run, east_asia=ea if ea != "Times New Roman" else "宋体",
                          western=western, size_pt=fmt["size"], bold=fmt["bold"])


# ── 声明区域 ──────────────────────────────────────────────

# 声明区域段落索引（34-46）
DECLARATION_TITLES = {34, 39}  # 两个声明标题
DECLARATION_BODY = {35, 40, 41, 42, 43}  # 声明正文
DECLARATION_SIG = {37, 45, 46}  # 签名行


def fix_declaration(doc: Document) -> None:
    """修复声明区域格式。"""
    for idx in DECLARATION_TITLES:
        if idx >= len(doc.paragraphs):
            continue
        p = doc.paragraphs[idx]
        _ensure_run(p)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf = p.paragraph_format
        pf.first_line_indent = Pt(0)
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        pf.space_before = Pt(12)
        pf.space_after = Pt(6)
        for run in p.runs:
            _set_run_font(run, east_asia="黑体", size_pt=14, bold=True)

    for idx in DECLARATION_BODY:
        if idx >= len(doc.paragraphs):
            continue
        p = doc.paragraphs[idx]
        _ensure_run(p)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf = p.paragraph_format
        pf.first_line_indent = Pt(0)  # 声明正文无首行缩进
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        for run in p.runs:
            _set_run_font(run, east_asia="宋体", size_pt=12)

    for idx in DECLARATION_SIG:
        if idx >= len(doc.paragraphs):
            continue
        p = doc.paragraphs[idx]
        _ensure_run(p)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        pf = p.paragraph_format
        pf.first_line_indent = Pt(0)
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        for run in p.runs:
            _set_run_font(run, east_asia="宋体", size_pt=12)


# ── 正文区域段落分类 ──────────────────────────────────────

# 正文区域范围：从"摘要"（48）到"参考文献"之前
# 参考文献区域：从"参考文献"（313）到"致谢"之前
# 致谢区域：从"致谢"（330）到文末

def _find_para_index(doc, text_start):
    """找到以指定文本开头的段落索引。"""
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip().startswith(text_start):
            return i
    return None


def classify_para(text: str) -> str:
    """根据段落文本判断类型。"""
    t = text.strip()
    if not t:
        return "empty"
    # 章标题
    if re.match(r"^第[一二三四五六七八]章\s", t):
        return "chapter"
    # 特殊标题
    if t in ("摘要", "ABSTRACT", "目录", "参考文献", "致谢"):
        return "chapter"
    # 附录/科研情况标题
    if t == "攻读硕士学位期间科研情况":
        return "appendix_title"
    # 附录子标题
    if t in ("已申请的软著", "参与的科研课题", "发表的论文"):
        return "appendix_subtitle"
    # 小节标题（必须在节标题之前判断）
    if re.match(r"^\d+\.\d+\.\d+\s", t):
        return "subsection"
    # 节标题
    if re.match(r"^\d+\.\d+\s{2}", t):
        return "section"
    # 图题
    if re.match(r"^图\d+-\d+\s", t):
        return "figure_caption"
    # 表题
    if re.match(r"^表\d+-\d+\s", t):
        return "table_caption"
    # 关键词
    if t.startswith("关键词：") or t.startswith("Keywords:"):
        return "keywords"
    # 参考文献条目
    if re.match(r"^\[\d+\]\s", t):
        return "reference"
    # 签名行
    if re.match(r"^(刘一江|二零二六)", t):
        return "signature"
    return "body"


def style_paragraph(para, *, kind: str) -> None:
    """按类型设置段落格式。"""
    if kind == "empty":
        return

    pf = para.paragraph_format

    if kind == "chapter":
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        pf.space_before = Pt(12)
        pf.space_after = Pt(6)
        pf.first_line_indent = Pt(0)
        for run in para.runs:
            _set_run_font(run, east_asia="黑体", size_pt=16, bold=True)

    elif kind == "section":
        para.alignment = WD_ALIGN_PARAGRAPH.LEFT
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        pf.space_before = Pt(6)
        pf.space_after = Pt(3)
        pf.first_line_indent = Pt(0)
        for run in para.runs:
            _set_run_font(run, east_asia="黑体", size_pt=14, bold=True)

    elif kind == "subsection":
        para.alignment = WD_ALIGN_PARAGRAPH.LEFT
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        pf.space_before = Pt(3)
        pf.space_after = Pt(3)
        pf.first_line_indent = Pt(0)
        for run in para.runs:
            _set_run_font(run, east_asia="黑体", size_pt=12, bold=True)

    elif kind == "figure_caption":
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf.first_line_indent = Pt(0)
        pf.space_before = Pt(3)
        pf.space_after = Pt(3)
        for run in para.runs:
            _set_run_font(run, east_asia="宋体", size_pt=10.5)

    elif kind == "table_caption":
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf.first_line_indent = Pt(0)
        pf.space_before = Pt(3)
        pf.space_after = Pt(3)
        for run in para.runs:
            _set_run_font(run, east_asia="黑体", size_pt=12, bold=True)

    elif kind == "keywords":
        para.alignment = WD_ALIGN_PARAGRAPH.LEFT
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        pf.first_line_indent = Pt(0)
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
        for run in para.runs:
            _set_run_font(run, east_asia="黑体", size_pt=12, bold=True)

    elif kind == "reference":
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        pf.first_line_indent = Pt(0)
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
        for run in para.runs:
            _set_run_font(run, east_asia="宋体", size_pt=10.5)

    elif kind == "appendix_title":
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        pf.space_before = Pt(12)
        pf.space_after = Pt(6)
        pf.first_line_indent = Pt(0)
        for run in para.runs:
            _set_run_font(run, east_asia="黑体", size_pt=16, bold=True)

    elif kind == "appendix_subtitle":
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        pf.space_before = Pt(6)
        pf.space_after = Pt(3)
        pf.first_line_indent = Pt(0)
        for run in para.runs:
            _set_run_font(run, east_asia="黑体", size_pt=14, bold=True)

    elif kind == "signature":
        para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        pf.first_line_indent = Pt(0)
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
        for run in para.runs:
            _set_run_font(run, east_asia="宋体", size_pt=12)

    elif kind == "body":
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        pf.first_line_indent = Pt(24)
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
        for run in para.runs:
            _set_run_font(run, east_asia="宋体", size_pt=12)


def apply_page_setup(doc: Document) -> None:
    """设置页面边距。"""
    for sec in doc.sections:
        sec.top_margin = Cm(2.54)
        sec.bottom_margin = Cm(2.54)
        sec.left_margin = Cm(3.17)
        sec.right_margin = Cm(3.17)


def apply_paragraph_styles(doc: Document) -> None:
    """遍历所有段落，按分类设置格式。跳过封面和声明区域。"""
    for i, para in enumerate(doc.paragraphs):
        # 跳过封面区域（0-32）
        if i in COVER_INDICES or i < 33:
            continue
        # 跳过声明区域（34-46），由 fix_declaration 处理
        if 34 <= i <= 46:
            continue

        kind = classify_para(para.text)
        if kind == "empty":
            continue
        _ensure_run(para)
        style_paragraph(para, kind=kind)


def apply_table_styles(doc: Document) -> None:
    """设置表格内文字格式：宋体五号，单倍行距。"""
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    para.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                    para.paragraph_format.space_before = Pt(0)
                    para.paragraph_format.space_after = Pt(0)
                    for run in para.runs:
                        _set_run_font(run, east_asia="宋体", size_pt=10.5)


# ── 主流程 ────────────────────────────────────────────────

def format_thesis(source: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)

    # 备份
    backup = source.with_name(
        f"{source.stem}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
    )
    shutil.copy2(source, backup)
    print(f"已备份: {backup}")

    doc = Document(str(source))

    # 1. 页面设置
    apply_page_setup(doc)
    print("已设置页面边距: 上2.54cm 下2.54cm 左3.17cm 右3.17cm")

    # 2. 封面区域
    fix_cover(doc)
    print("已修复封面区域格式")

    # 3. 声明区域
    fix_declaration(doc)
    print("已修复声明区域格式")

    # 4. 正文区域段落格式
    apply_paragraph_styles(doc)
    print("已应用段落格式（章节标题、正文、图题、表题、关键词、参考文献、附录等）")

    # 5. 表格格式
    apply_table_styles(doc)
    print("已应用表格格式")

    # 保存
    doc.save(str(source))
    print(f"已保存: {source}")


if __name__ == "__main__":
    format_thesis(THESIS)
