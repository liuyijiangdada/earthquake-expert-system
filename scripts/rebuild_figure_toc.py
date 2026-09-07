#!/usr/bin/env python3
"""重新生成图目录。

图目录区域此前被清空（TOC 域内容丢失）。本脚本：
1. 扫描正文中所有图标题（去重，排除正文引用句）
2. 为没有书签的图标题补书签
3. 清空图目录区域的空段落
4. 仿照表目录条目结构，生成图目录条目（标题 + 制表符 + PAGEREF 域）
5. 用户在 Word 里「更新域」后 PAGEREF 自动填充正确页码
"""
from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

from docx import Document
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def is_figure_caption(text: str) -> bool:
    """判断段落是否是图标题（非正文引用）。"""
    t = text.strip()
    m = re.match(r"^图\s*\d+-\d+\s+\S", t)
    if not m:
        return False
    if len(t) > 40:
        return False
    # 排除正文引用句（含动词/描述词）
    if re.search(r"给出|展示|说明|如图|所示|描述|为图|是图", t):
        return False
    return True


def get_bookmark(para) -> str | None:
    xml = para._p.xml
    m = re.search(r'w:bookmarkStart[^>]*w:name="([^"]+)"', xml)
    return m.group(1) if m else None


def add_bookmark(para, name: str, bm_id: int) -> str:
    """在段落首尾插入书签。"""
    bm_start = parse_xml(
        f'<w:bookmarkStart {nsdecls("w")} w:id="{bm_id}" w:name="{name}"/>'
    )
    bm_end = parse_xml(
        f'<w:bookmarkEnd {nsdecls("w")} w:id="{bm_id}"/>'
    )
    para._p.insert(0, bm_start)
    para._p.append(bm_end)
    return name


def make_toc_entry(fig_label: str, title: str, bookmark: str, ppr_xml: str):
    """构造一个图目录条目段落元素。"""
    # 转义标题里的特殊字符
    safe_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    entry_xml = (
        f'<w:p {nsdecls("w")}>'
        f'{ppr_xml}'
        f'<w:hyperlink w:anchor="{bookmark}" w:history="1">'
        f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="宋体" '
        f'w:hAnsi="Times New Roman"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
        f'<w:t xml:space="preserve">{fig_label}  {safe_title}</w:t></w:r>'
        f'<w:r><w:tab/></w:r>'
        f'</w:hyperlink>'
        f'<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
        f'<w:r><w:instrText xml:space="preserve"> PAGEREF {bookmark} \\h </w:instrText></w:r>'
        f'<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
        f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="宋体" '
        f'w:hAnsi="Times New Roman"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
        f'<w:t>1</w:t></w:r>'
        f'<w:r><w:fldChar w:fldCharType="end"/></w:r>'
        f'</w:p>'
    )
    return parse_xml(entry_xml)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: rebuild_figure_toc.py <docx_path>")
        return 1
    docx_path = Path(sys.argv[1])
    doc = Document(str(docx_path))
    paras = doc.paragraphs

    # 1. 识别图标题（去重）
    print("=== 1. 识别图标题 ===")
    fig_titles = []  # [(para, fig_label, title, bookmark)]
    seen = set()
    for p in paras:
        if not is_figure_caption(p.text):
            continue
        m = re.match(r"^(图\s*\d+-\d+)\s+(.+)$", p.text.strip())
        if not m:
            continue
        label = re.sub(r"\s+", "", m.group(1))  # "图3-1"
        if label in seen:
            continue
        seen.add(label)
        title = m.group(2).strip()
        fig_titles.append([p, label, title, None])
        print(f"  {label}  {title}")

    # 2. 补书签
    print()
    print("=== 2. 补书签 ===")
    bm_id = 9000
    for entry in fig_titles:
        p, label, title, _ = entry
        bm = get_bookmark(p)
        if bm is None:
            bm = f"_TocFig{bm_id}"
            add_bookmark(p, bm, bm_id)
            bm_id += 1
            print(f"  补书签 {bm} -> {label}  {title}")
        entry[3] = bm

    # 3. 定位图目录区域（"图目录"标题 到 "表目录"标题 之间）
    print()
    print("=== 3. 定位图目录区域 ===")
    fig_toc_idx = None
    tbl_toc_idx = None
    for i, p in enumerate(paras):
        if p.text.strip() == "图目录":
            fig_toc_idx = i
        elif p.text.strip() == "表目录":
            tbl_toc_idx = i
            break
    if fig_toc_idx is None or tbl_toc_idx is None:
        print("未找到「图目录」或「表目录」标题段落")
        return 1
    print(f"  图目录标题: 段落{fig_toc_idx}")
    print(f"  表目录标题: 段落{tbl_toc_idx}")

    # 获取模板 pPr（从表目录第一个条目）
    template_entry = None
    for i in range(tbl_toc_idx + 1, len(paras)):
        if "PAGEREF" in paras[i]._p.xml:
            template_entry = paras[i]
            break
    if template_entry is None:
        print("未找到表目录条目作为模板")
        return 1
    pPr = template_entry._p.find(qn("w:pPr"))
    ppr_xml = etree.tostring(pPr, encoding="unicode").strip() if pPr is not None else ""
    # 移除 pPr 里的命名空间声明（外层 p 已有）
    ppr_xml = re.sub(r' xmlns:[a-z0-9]+="[^"]*"', "", ppr_xml)
    print(f"  模板条目: 段落{tbl_toc_idx + 1}, pStyle={pPr.find(qn('w:pStyle')).get(qn('w:val')) if pPr is not None and pPr.find(qn('w:pStyle')) is not None else 'N/A'}")

    # 4. 删除图目录区域的空段落（图目录标题+1 到 表目录标题-1）
    print()
    print("=== 4. 清空图目录区域空段落 ===")
    fig_heading_p = paras[fig_toc_idx]._p
    tbl_heading_p = paras[tbl_toc_idx]._p
    # 收集 fig_toc_idx+1 到 tbl_toc_idx-1 之间的段落元素
    to_remove = []
    cur = fig_heading_p.getnext()
    while cur is not None and cur is not tbl_heading_p:
        nxt = cur.getnext()
        if cur.tag == qn("w:p"):
            to_remove.append(cur)
        cur = nxt
    print(f"  待删除空段落数: {len(to_remove)}")
    for elem in to_remove:
        elem.getparent().remove(elem)

    # 5. 生成并插入图目录条目
    print()
    print("=== 5. 生成并插入图目录条目 ===")
    prev = fig_heading_p
    for p, label, title, bm in fig_titles:
        entry_elem = make_toc_entry(label, title, bm, ppr_xml)
        prev.addnext(entry_elem)
        prev = entry_elem
        print(f"  插入: {label}  {title}  -> {bm}")

    doc.save(str(docx_path))
    print()
    print(f"已保存：{docx_path}")
    print(f"共插入 {len(fig_titles)} 个图目录条目。")
    print()
    print("请在 WPS/Word 中打开文档，右键图目录 → 更新域 → 更新整个目录，")
    print("PAGEREF 域将自动填充正确页码。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
