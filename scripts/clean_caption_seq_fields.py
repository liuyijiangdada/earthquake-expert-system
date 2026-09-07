#!/usr/bin/env python3
"""清理图表标题段落里多余的 SEQ 自动编号域。

背景：每个「表 X-Y  标题」「图 X-Y  标题」末尾被挂了一个 Word SEQ 域
（域代码 `SEQ 表 \* ARABIC` / `SEQ 图 \* ARABIC`），渲染出一个多余的「1」。
本脚本遍历所有段落，删除其中的 SEQ 域结构
（fldChar begin → instrText(SEQ) → fldChar separate → 缓存值 → fldChar end），
保留段落的正常文字 run。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

CAPTION_RE = re.compile(r"^\s*[图表]\s*\d")


def _is_field_begin(r_elem) -> bool:
    for fc in r_elem.findall(qn("w:fldChar")):
        if fc.get(qn("w:fldCharType")) == "begin":
            return True
    return False


def _is_field_end(r_elem) -> bool:
    for fc in r_elem.findall(qn("w:fldChar")):
        if fc.get(qn("w:fldCharType")) == "end":
            return True
    return False


def _field_instr_text(r_elem) -> str:
    parts = []
    for it in r_elem.findall(qn("w:instrText")):
        parts.append(it.text or "")
    return "".join(parts)


def _remove_seq_fields_in_paragraph(p_elem) -> int:
    """删除段落里的 SEQ 域，返回删除的域个数。"""
    children = list(p_elem)
    removed = 0
    i = 0
    while i < len(children):
        child = children[i]
        # 只在 w:r 元素里找 fldChar begin
        if child.tag != qn("w:r") or not _is_field_begin(child):
            i += 1
            continue
        # 找到域开始，往后扫描直到域结束，同时判断是否是 SEQ 域
        seq_instr = None
        j = i
        is_seq = False
        while j < len(children):
            cur = children[j]
            if cur.tag == qn("w:r"):
                instr = _field_instr_text(cur)
                if instr and "SEQ" in instr.upper():
                    is_seq = True
                    seq_instr = instr.strip()
                if _is_field_end(cur):
                    break
            j += 1
        # j 指向域结束 run（或越界）
        if is_seq:
            end_idx = j if j < len(children) else len(children) - 1
            # 删除 [i, end_idx] 区间
            for k in range(end_idx, i - 1, -1):
                if k < len(children):
                    p_elem.remove(children[k])
            removed += 1
            # 重新读取 children
            children = list(p_elem)
            i = 0
            continue
        i += 1
    return removed


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: clean_caption_seq_fields.py <docx_path>")
        return 1
    docx_path = Path(sys.argv[1])
    if not docx_path.exists():
        print(f"not found: {docx_path}")
        return 1

    doc = Document(str(docx_path))
    total_removed = 0
    touched_paras = 0
    for para in doc.paragraphs:
        text = para.text.strip()
        if not CAPTION_RE.match(text):
            continue
        removed = _remove_seq_fields_in_paragraph(para._p)
        if removed:
            total_removed += removed
            touched_paras += 1
            print(f"  清理 {removed} 个 SEQ 域 | {text[:40]}")

    doc.save(str(docx_path))
    print()
    print(f"共清理 {total_removed} 个 SEQ 域，涉及 {touched_paras} 个图表标题段落。")
    print(f"已保存：{docx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
