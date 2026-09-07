#!/usr/bin/env python3
"""清理图表目录条目里多余的「1」缓存文本。

背景：图表目录（TOC \\c "图"/"表"）是之前生成的，条目文字是从旧标题
（带 SEQ 域、末尾渲染出「1」）抓取的静态缓存。正文 SEQ 域已清除，
但目录条目里的「1」仍作为静态文本残留，形如：
    「图3-1  总体方法框架图1 \t23」
本脚本遍历图表目录条目，删除标题文字与制表符/页码之间多余的「1」。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document

CAPTION_RE = re.compile(r"^\s*[图表]\s*\d")


def _clean_toc_entry(para) -> bool:
    """清理单个目录条目里的多余「1」，返回是否修改。"""
    changed = False
    runs = para.runs
    # 找到标题文字 run（含「图/表」与「-」）
    caption_idx = None
    for i, r in enumerate(runs):
        t = r.text or ""
        if ("图" in t or "表" in t) and "-" in t:
            caption_idx = i
            break
    if caption_idx is None:
        return False

    # 情况1：标题 run 末尾带「1」（如「...定位1 」）
    # 用负向后顾确保「1」前面不是「-」，避免删掉「5-1」这类编号里的 1
    cap_run = runs[caption_idx]
    cap_text = cap_run.text or ""
    new_text = re.sub(r"(?<!-)1\s*$", "", cap_text)
    if new_text != cap_text:
        cap_run.text = new_text
        changed = True

    # 情况2：标题 run 后面紧跟一个纯「1」/「1 」run
    for j in range(caption_idx + 1, len(runs)):
        r = runs[j]
        t = (r.text or "").strip()
        if t == "":
            continue
        if t == "1":
            r.text = ""
            changed = True
            continue
        # 遇到制表符或页码数字，说明已进入页码区，停止
        if "\t" in (r.text or "") or re.match(r"^\d+$", t):
            break
        # 其它非空内容（如「1 」），若是标题后缀的1
        if re.match(r"^1\s+$", r.text or ""):
            r.text = ""
            changed = True
            continue
        break
    return changed


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: clean_toc_caption_text.py <docx_path>")
        return 1
    docx_path = Path(sys.argv[1])
    if not docx_path.exists():
        print(f"not found: {docx_path}")
        return 1

    doc = Document(str(docx_path))
    touched = 0
    for i, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if not CAPTION_RE.match(text):
            continue
        # 只处理目录条目（含制表符或末尾页码），跳过正文标题
        if "\t" not in para.text:
            continue
        before = para.text
        if _clean_toc_entry(para):
            after = para.text
            touched += 1
            print(f"  [{i}] {before.strip()[:45]}")
            print(f"     -> {after.strip()[:45]}")

    doc.save(str(docx_path))
    print()
    print(f"共清理 {touched} 个图表目录条目。")
    print(f"已保存：{docx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
