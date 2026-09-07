#!/usr/bin/env python3
"""删除附录 A、D、E、F，保留 B、C 并重编号为 A、B。

操作清单：
1. 改写两处失效引用（段落 433、520）
2. 更新保留引用「见附录 B」→「见附录 A」（段落 384）
3. 正文附录标题重编号：B→A、C→B（含小节 B.1/B.2→A.1/A.2）
4. 目录条目重编号：B→A、C→B
5. 删除正文附录 A(682-686)、D(693-698)、E(699-704)、F(705-749)
6. 删除目录条目 A、D、E、F（段落 132、140、143、146）
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document


def replace_in_paragraph(para, old: str, new: str) -> bool:
    """在段落里替换文本，处理单 run 与跨 run 两种情况。"""
    for r in para.runs:
        if old in (r.text or ""):
            r.text = (r.text or "").replace(old, new)
            return True
    full = para.text
    if old in full:
        new_full = full.replace(old, new)
        if para.runs:
            para.runs[0].text = new_full
            for r in para.runs[1:]:
                r.text = ""
        return True
    return False


def replace_regex_in_paragraph(para, pattern: str, new: str) -> bool:
    """用正则在段落里替换（跨 run 时合并到第一个 run）。"""
    for r in para.runs:
        if re.search(pattern, r.text or ""):
            r.text = re.sub(pattern, new, r.text or "")
            return True
    full = para.text
    if re.search(pattern, full):
        new_full = re.sub(pattern, new, full)
        if para.runs:
            para.runs[0].text = new_full
            for r in para.runs[1:]:
                r.text = ""
        return True
    return False


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: cleanup_appendix_adef.py <docx_path>")
        return 1
    docx_path = Path(sys.argv[1])
    doc = Document(str(docx_path))
    paras = doc.paragraphs

    print("=== 1. 改写失效引用 ===")
    # 段落 433：删除末句「完整伪代码见附录 F.3。」
    ok433 = replace_in_paragraph(paras[433], "完整伪代码见附录 F.3。", "")
    print(f"  [433] 删除「完整伪代码见附录 F.3。」: {ok433}")
    # 段落 520：删除「（完整 60 题见附录 A）」全角括号
    ok520a = replace_regex_in_paragraph(paras[520], r"[（(]\s*完整\s*60\s*题见附录\s*A\s*[)）]", "")
    print(f"  [520] 删除「（完整60题见附录A）」: {ok520a}")

    print("=== 2. 更新保留引用 ===")
    # 段落 384：「见附录 B」→「见附录 A」
    ok384 = replace_in_paragraph(paras[384], "见附录 B", "见附录 A")
    print(f"  [384] 「见附录B」→「见附录A」: {ok384}")

    print("=== 3. 正文附录标题重编号 ===")
    ok687 = replace_in_paragraph(paras[687], "附录 B", "附录 A")
    ok689 = replace_in_paragraph(paras[689], "B.1", "A.1")
    ok690 = replace_in_paragraph(paras[690], "B.2", "A.2")
    ok691 = replace_in_paragraph(paras[691], "附录 C", "附录 B")
    print(f"  [687] 附录B→附录A: {ok687}")
    print(f"  [689] B.1→A.1: {ok689}")
    print(f"  [690] B.2→A.2: {ok690}")
    print(f"  [691] 附录C→附录B: {ok691}")

    print("=== 4. 目录条目重编号 ===")
    ok136 = replace_in_paragraph(paras[136], "附录 B", "附录 A")
    ok139 = replace_in_paragraph(paras[139], "附录 C", "附录 B")
    print(f"  [136] 目录 附录B→附录A: {ok136}")
    print(f"  [139] 目录 附录C→附录B: {ok139}")

    print("=== 5. 收集并删除正文附录 A/D/E/F 段落 ===")
    # 正文附录 A: 682-686, D: 693-698, E: 699-704, F: 705-749
    body_del_indices = (
        list(range(682, 687))
        + list(range(693, 699))
        + list(range(699, 705))
        + list(range(705, 750))
    )
    # 目录条目 A、D、E、F
    toc_del_indices = [132, 140, 143, 146]
    to_delete = [paras[i] for i in body_del_indices + toc_del_indices]
    print(f"  待删除段落数: {len(to_delete)} (正文{len(body_del_indices)} + 目录{len(toc_del_indices)})")

    for p in to_delete:
        p._element.getparent().remove(p._element)

    doc.save(str(docx_path))
    print()
    print(f"已保存：{docx_path}")
    print(f"删除段落数：{len(to_delete)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
