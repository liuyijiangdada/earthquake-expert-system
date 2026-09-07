#!/usr/bin/env python3
"""修正第五章表编号冲突。

按出现顺序重新编号表5-1 ~ 表5-10，并同步更新正文引用。
映射基于「最近同编号表标题」策略：对每个引用，找同旧编号的表标题中
段落位置最近的，取其新编号。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document


# 表标题映射: 段落索引 -> (旧编号, 新编号)
TITLE_MAP = {
    457: ("5-3", "5-1"),   # 实验环境配置
    460: ("5-1", "5-2"),   # 离线消融实验配置快照
    462: ("5-4", "5-3"),   # 60题评测问集样例
    465: ("5-2", "5-4"),   # 四基线60题自动评分汇总
    474: ("5-3", "5-5"),   # 与GraphRAG/KnowledGPT设计对比
    477: ("5-5", "5-6"),   # B0—B3事实一致性细分
    481: ("5-4", "5-7"),   # 分阶段自动评分汇总
    492: ("5-6", "5-8"),   # 典型错误案例与根因归因
    500: ("5-7", "5-9"),   # 小规模试评结果
    505: ("5-8", "5-10"),  # 在线演示路径时延
}

# 正文引用映射: 段落索引 -> (旧编号, 新编号)
REF_MAP = {
    456: ("5-3", "5-1"),
    459: ("5-1", "5-2"),
    461: ("5-4", "5-3"),
    473: ("5-3", "5-5"),
    476: ("5-5", "5-6"),
    478: ("5-5", "5-6"),
    480: ("5-4", "5-7"),
    482: ("5-4", "5-7"),
    487: ("5-2", "5-4"),
    491: ("5-6", "5-8"),
    493: ("5-6", "5-8"),
    499: ("5-7", "5-9"),
    506: ("5-8", "5-10"),
}


def replace_in_paragraph(para, old_num: str, new_num: str) -> bool:
    """在段落里把 表<空格>old_num 替换为 表<空格>new_num，保留空格格式。"""
    changed = False
    # 匹配 表 + 可选空格 + old_num，保留空格
    pattern = re.compile(r"(表\s*)" + re.escape(old_num))
    for r in para.runs:
        if r.text and pattern.search(r.text):
            r.text = pattern.sub(rf"\g<1>{new_num}", r.text)
            changed = True
            return changed
    # 跨 run 情况：合并到第一个 run
    full = para.text
    if pattern.search(full):
        new_full = pattern.sub(rf"\g<1>{new_num}", full)
        if para.runs:
            para.runs[0].text = new_full
            for r in para.runs[1:]:
                r.text = ""
        changed = True
    return changed


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: fix_table_numbering.py <docx_path>")
        return 1
    docx_path = Path(sys.argv[1])
    doc = Document(str(docx_path))
    paras = doc.paragraphs

    print("=== 修正表标题编号 ===")
    for idx, (old, new) in TITLE_MAP.items():
        if idx >= len(paras):
            print(f"  [段落{idx}] 超出范围，跳过")
            continue
        p = paras[idx]
        before = p.text.strip()[:40]
        ok = replace_in_paragraph(p, old, new)
        after = p.text.strip()[:40]
        flag = "✓" if ok else "✗ 未找到"
        print(f"  [{idx}] {old}→{new} {flag}: {before} => {after}")

    print("\n=== 修正正文引用 ===")
    for idx, (old, new) in REF_MAP.items():
        if idx >= len(paras):
            print(f"  [段落{idx}] 超出范围，跳过")
            continue
        p = paras[idx]
        before = p.text
        ok = replace_in_paragraph(p, old, new)
        after = p.text
        # 找替换位置的上下文
        m_old = re.search(r"表\s*" + re.escape(old), before)
        ctx_before = before[max(0, m_old.start() - 8):m_old.end() + 8] if m_old else ""
        m_new = re.search(r"表\s*" + re.escape(new), after)
        ctx_after = after[max(0, m_new.start() - 8):m_new.end() + 8] if m_new else ""
        flag = "✓" if ok else "✗ 未找到"
        print(f"  [{idx}] {old}→{new} {flag}: ...{ctx_before}... => ...{ctx_after}...")

    doc.save(str(docx_path))
    print(f"\n已保存：{docx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
