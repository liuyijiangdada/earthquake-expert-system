#!/usr/bin/env python3
"""调整图/表目录样式，参考媛媛论文。

1. 「图目录」「表目录」标题居中
2. 目录条目左对齐（移除继承的 jc=center）
3. 确保条目有点引导线制表符（右对齐 dot leader）
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn


def set_jc(para, val: str | None):
    """设置段落对齐。val=None 表示移除 jc。"""
    pPr = para._p.find(qn("w:pPr"))
    if pPr is None:
        if val is None:
            return
        pPr = parse_xml(f'<w:pPr {nsdecls("w")}></w:pPr>')
        para._p.insert(0, pPr)
    jc = pPr.find(qn("w:jc"))
    if val is None:
        if jc is not None:
            pPr.remove(jc)
    else:
        if jc is None:
            jc = parse_xml(f'<w:jc {nsdecls("w")} w:val="{val}"/>')
            pPr.append(jc)
        else:
            jc.set(qn("w:val"), val)


def ensure_tabs(para):
    """确保段落 pPr 有右对齐点引导线制表符。"""
    pPr = para._p.find(qn("w:pPr"))
    if pPr is None:
        pPr = parse_xml(f'<w:pPr {nsdecls("w")}></w:pPr>')
        para._p.insert(0, pPr)
    tabs = pPr.find(qn("w:tabs"))
    if tabs is None:
        tabs = parse_xml(f'<w:tabs {nsdecls("w")}></w:tabs>')
        pPr.append(tabs)
    # 检查是否已有 right+dot tab
    for tab in tabs.findall(qn("w:tab")):
        if tab.get(qn("w:val")) == "right" and tab.get(qn("w:leader")) == "dot":
            return True
    # 添加
    tab = parse_xml(f'<w:tab {nsdecls("w")} w:val="right" w:leader="dot" w:pos="8295"/>')
    tabs.append(tab)
    return False


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: fix_toc_style.py <docx_path>")
        return 1
    doc = Document(sys.argv[1])
    paras = doc.paragraphs

    # 定位图目录、表目录标题
    fig_idx = tbl_idx = None
    for i, p in enumerate(paras):
        if p.text.strip() == "图目录":
            fig_idx = i
        elif p.text.strip() == "表目录":
            tbl_idx = i
            break
    if fig_idx is None or tbl_idx is None:
        print("未找到图目录/表目录标题")
        return 1

    # 1. 标题居中
    print("=== 1. 标题居中 ===")
    for idx, name in [(fig_idx, "图目录"), (tbl_idx, "表目录")]:
        set_jc(paras[idx], "center")
        print(f"  [{idx}] {name} -> 居中")

    # 2. 图目录条目: 左对齐 + 确保tabs
    print("\n=== 2. 图目录条目样式 ===")
    cnt = 0
    for i in range(fig_idx + 1, tbl_idx):
        p = paras[i]
        t = p.text.strip()
        if not t:
            continue
        set_jc(p, None)  # 移除jc,默认左对齐
        ensure_tabs(p)
        cnt += 1
    print(f"  处理 {cnt} 个图目录条目 -> 左对齐 + 点引导线")

    # 3. 表目录条目: 左对齐 + 确保tabs
    print("\n=== 3. 表目录条目样式 ===")
    cnt = 0
    for i in range(tbl_idx + 1, len(paras)):
        p = paras[i]
        t = p.text.strip()
        if not t:
            continue
        if re.match(r"^第[一二三四五六七八九十]+章", t):
            break
        set_jc(p, None)
        ensure_tabs(p)
        cnt += 1
    print(f"  处理 {cnt} 个表目录条目 -> 左对齐 + 点引导线")

    doc.save(sys.argv[1])
    print(f"\n已保存：{sys.argv[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
