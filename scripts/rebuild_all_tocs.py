#!/usr/bin/env python3
"""综合重建图目录与表目录。

修复问题：
- 图目录缺图4-3、部分条目 PAGEREF 丢失 → 全部清空重建
- 表目录条目全部丢失 → 从零重建
- 表5-3 有两个（实验环境配置 / 设计对比），如实列入

条目按「章号-序号」数字排序；用静态 PAGEREF 域引用书签，
用户在 Word 更新域即可刷新页码，不会被 TOC \\c 清空。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from lxml import etree


def is_caption(text: str, kind: str) -> bool:
    """判断段落是否是图/表标题（非正文引用）。kind='图'或'表'。"""
    t = text.strip()
    if not re.match(rf"^{kind}\s*\d+-\d+\s+\S", t):
        return False
    if kind == "图" and len(t) > 40:
        return False
    if kind == "表" and len(t) > 50:
        return False
    verbs = "给出|展示|说明|如{0}|所示|描述|为{0}|是{0}".format(kind)
    if re.search(verbs, t[:15]):
        return False
    return True


def parse_label(text: str, kind: str):
    """从标题文本提取 (label, title)。返回 (章号, 序号, label_str, title)。"""
    m = re.match(rf"^({kind}\s*\d+-\d+)\s+(.+)$", text.strip())
    if not m:
        return None
    label = re.sub(r"\s+", "", m.group(1))  # "图3-1"
    nums = re.search(r"(\d+)-(\d+)", label)
    chapter = int(nums.group(1))
    seq = int(nums.group(2))
    title = m.group(2).strip()
    return (chapter, seq, label, title)


def get_bookmark(para) -> str | None:
    m = re.search(r'w:bookmarkStart[^>]*w:name="([^"]+)"', para._p.xml)
    return m.group(1) if m else None


def add_bookmark(para, name: str, bm_id: int) -> str:
    bm_start = parse_xml(
        f'<w:bookmarkStart {nsdecls("w")} w:id="{bm_id}" w:name="{name}"/>'
    )
    bm_end = parse_xml(f'<w:bookmarkEnd {nsdecls("w")} w:id="{bm_id}"/>')
    para._p.insert(0, bm_start)
    para._p.append(bm_end)
    return name


def make_toc_entry(label: str, title: str, bookmark: str, ppr_xml: str):
    """构造目录条目段落元素。"""
    safe_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    entry_xml = (
        f'<w:p {nsdecls("w")}>'
        f'{ppr_xml}'
        f'<w:hyperlink w:anchor="{bookmark}" w:history="1">'
        f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="宋体" '
        f'w:hAnsi="Times New Roman"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
        f'<w:t xml:space="preserve">{label}  {safe_title}</w:t></w:r>'
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


def clear_area(start_heading_elem, stop_predicate):
    """删除 start_heading 之后、直到 stop_predicate 为真的段落。返回删除数。"""
    to_remove = []
    cur = start_heading_elem.getnext()
    while cur is not None:
        if cur.tag != qn("w:p"):
            cur = cur.getnext()
            continue
        # 取段落文本
        texts = cur.findall(f".//{qn('w:t')}")
        full = "".join(t.text or "" for t in texts).strip()
        if stop_predicate(full):
            break
        to_remove.append(cur)
        cur = cur.getnext()
    for elem in to_remove:
        elem.getparent().remove(elem)
    return len(to_remove)


def collect_captions(paras, kind: str, min_idx: int = 200):
    """收集正文图/表标题，按编号排序。返回 [(para, label, title, bookmark)]。"""
    items = []
    for i, p in enumerate(paras):
        if i < min_idx:
            continue
        if not is_caption(p.text, kind):
            continue
        parsed = parse_label(p.text, kind)
        if parsed is None:
            continue
        chapter, seq, label, title = parsed
        bm = get_bookmark(p)
        items.append((chapter, seq, i, p, label, title, bm))
    # 按编号排序
    items.sort(key=lambda x: (x[0], x[1], x[2]))
    return [(p, label, title, bm) for _, _, _, p, label, title, bm in items]


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: rebuild_all_tocs.py <docx_path>")
        return 1
    docx_path = Path(sys.argv[1])
    doc = Document(str(docx_path))
    paras = doc.paragraphs

    # 定位图目录、表目录标题
    fig_toc_idx = tbl_toc_idx = None
    for i, p in enumerate(paras):
        if p.text.strip() == "图目录":
            fig_toc_idx = i
        elif p.text.strip() == "表目录":
            tbl_toc_idx = i
            break
    if fig_toc_idx is None or tbl_toc_idx is None:
        print("未找到「图目录」或「表目录」标题")
        return 1

    # 获取模板 pPr（从表目录标题后找，或用默认）
    ppr_xml = ""
    for i in range(tbl_toc_idx + 1, len(paras)):
        if "PAGEREF" in paras[i]._p.xml or paras[i].text.strip():
            pPr = paras[i]._p.find(qn("w:pPr"))
            if pPr is not None:
                ppr_xml = etree.tostring(pPr, encoding="unicode").strip()
                ppr_xml = re.sub(r' xmlns:[a-z0-9]+="[^"]*"', "", ppr_xml)
            break
    if not ppr_xml:
        # 默认 pPr
        ppr_xml = (
            '<w:pPr><w:pStyle w:val="14"/>'
            '<w:tabs><w:tab w:val="right" w:leader="dot" w:pos="8314"/></w:tabs>'
            '</w:pPr>'
        )

    # 收集图标题、表标题
    figs = collect_captions(paras, "图")
    tbls = collect_captions(paras, "表")

    print(f"图标题: {len(figs)} 个")
    for p, label, title, bm in figs:
        print(f"  {label}  {title}  bm={bm}")
    print(f"表标题: {len(tbls)} 个")
    for p, label, title, bm in tbls:
        print(f"  {label}  {title}  bm={bm}")

    # 补书签
    print("\n=== 补书签 ===")
    bm_id = 9100
    for lst in (figs, tbls):
        for idx, (p, label, title, bm) in enumerate(lst):
            if bm is None:
                new_bm = f"_TocAuto{bm_id}"
                add_bookmark(p, new_bm, bm_id)
                bm_id += 1
                lst[idx] = (p, label, title, new_bm)
                print(f"  补 {new_bm} -> {label}  {title}")

    # 清空图目录区域（图目录标题后，到表目录标题前）
    fig_heading = paras[fig_toc_idx]._p
    tbl_heading = paras[tbl_toc_idx]._p
    n1 = clear_area(fig_heading, lambda t: t == "表目录")
    print(f"\n清空图目录区域: 删除 {n1} 段")

    # 清空表目录区域（表目录标题后，到第一个"第X章"或非空非表段落）
    def tbl_stop(t):
        return bool(re.match(r"^第[一二三四五六七八九十]+章", t)) or t in ("参考文献", "致谢")
    n2 = clear_area(tbl_heading, tbl_stop)
    print(f"清空表目录区域: 删除 {n2} 段")

    # 生成图目录条目
    print("\n=== 插入图目录条目 ===")
    prev = fig_heading
    for p, label, title, bm in figs:
        elem = make_toc_entry(label, title, bm, ppr_xml)
        prev.addnext(elem)
        prev = elem
        print(f"  {label}  {title}")

    # 生成表目录条目
    print("\n=== 插入表目录条目 ===")
    prev = tbl_heading
    for p, label, title, bm in tbls:
        elem = make_toc_entry(label, title, bm, ppr_xml)
        prev.addnext(elem)
        prev = elem
        print(f"  {label}  {title}")

    doc.save(str(docx_path))
    print(f"\n已保存：{docx_path}")
    print(f"图目录 {len(figs)} 条，表目录 {len(tbls)} 条。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
