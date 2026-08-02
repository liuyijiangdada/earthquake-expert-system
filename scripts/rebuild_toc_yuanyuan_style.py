#!/usr/bin/env python3
"""按媛媛论文样式重建主目录、图目录、表目录。"""
from __future__ import annotations

import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

from docx import Document
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from lxml import etree

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCX = ROOT / "华东师范大学硕士论文.docx"

CAPTION_VERBS = re.compile(r"给出|展示|说明|如图|如表|所示|描述|为图|为表|是图|是表")

FRONT_BOOKMARKS = (
    ("摘要", "_TocFrontAbstract", 8001),
    ("ABSTRACT", "_TocFrontABSTRACT", 8002),
    ("Abstract", "_TocFrontABSTRACT", 8002),
    ("图目录", "_TocFrontFigList", 8003),
    ("表目录", "_TocFrontTblList", 8004),
)


def is_caption(text: str, kind: str) -> bool:
    t = text.strip()
    if not re.match(rf"^{kind}\s*\d+-\d+\s+\S", t):
        return False
    if kind == "图" and len(t) > 60:
        return False
    if kind == "表" and len(t) > 80:
        return False
    if CAPTION_VERBS.search(t[:20]):
        return False
    return True


def parse_label(text: str, kind: str):
    m = re.match(rf"^({kind}\s*\d+-\d+)\s+(.+)$", text.strip())
    if not m:
        return None
    label = re.sub(r"\s+", "", m.group(1))
    nums = re.search(r"(\d+)-(\d+)", label)
    if not nums:
        return None
    return int(nums.group(1)), int(nums.group(2)), label, m.group(2).strip()


def backup_docx(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = path.with_suffix(path.suffix + f".bak-toc-{stamp}")
    shutil.copy2(path, bak)
    return bak


def set_outline_levels(doc: Document) -> int:
    """为摘要/ABSTRACT/图目录/表目录/章/节/小节/参考文献/致谢/附录设大纲级别。"""
    count = 0
    body_started = False
    for p in doc.paragraphs:
        t = p.text.strip()
        if t == "第一章  绪论" or re.match(r"^第一章", t):
            # 正文第一章（目录后的那次）也会设；目录区旧条目随后删除
            pass
        level = None
        if t in ("摘要", "ABSTRACT", "Abstract", "图目录", "表目录", "参考文献", "致谢"):
            level = 0
        elif re.match(r"^附录\s*[A-Z]", t) or re.match(r"^附录[A-Z]", t):
            level = 0
        elif re.match(r"^第[一二三四五六七八九十]+章", t) and len(t) < 50:
            level = 0
        elif re.match(r"^[A-Z]\.\d+\s", t) and len(t) < 40:  # A.1
            level = 1
        elif re.match(r"^\d+\.\d+\.\d+\s", t) and len(t) < 60:
            level = 2
        elif re.match(r"^\d+\.\d+\s", t) and len(t) < 50:
            level = 1
        if level is None:
            continue
        pPr = p._p.find(qn("w:pPr"))
        if pPr is None:
            pPr = parse_xml(f"<w:pPr {nsdecls('w')}/>")
            p._p.insert(0, pPr)
        existing = pPr.find(qn("w:outlineLvl"))
        if existing is not None:
            pPr.remove(existing)
        pPr.append(parse_xml(f'<w:outlineLvl {nsdecls("w")} w:val="{level}"/>'))
        count += 1
    return count


def clear_between(start_elem, stop_predicate) -> int:
    removed = 0
    cur = start_elem.getnext()
    while cur is not None:
        if cur.tag != qn("w:p"):
            cur = cur.getnext()
            continue
        texts = cur.findall(f".//{qn('w:t')}")
        full = "".join(t.text or "" for t in texts).strip()
        if stop_predicate(full):
            break
        nxt = cur.getnext()
        cur.getparent().remove(cur)
        removed += 1
        cur = nxt
    return removed


def insert_toc_field(after_elem) -> None:
    toc_xml = (
        f'<w:p {nsdecls("w")}>'
        f"<w:pPr/>"
        f'<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
        f'<w:r><w:instrText xml:space="preserve"> TOC \\o "1-3" \\h \\z \\u </w:instrText></w:r>'
        f'<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
        f'<w:r><w:t>（请在 WPS/Word 中右键此处→更新域，生成目录）</w:t></w:r>'
        f'<w:r><w:fldChar w:fldCharType="end"/></w:r>'
        f"</w:p>"
    )
    after_elem.addnext(parse_xml(toc_xml))


def ensure_bookmark(para, name: str, bm_id: int) -> str:
    """若段落尚无指定书签则插入，返回书签名。"""
    for child in para._p:
        if child.tag == qn("w:bookmarkStart") and child.get(qn("w:name")) == name:
            return name
    para._p.insert(
        0,
        parse_xml(f'<w:bookmarkStart {nsdecls("w")} w:id="{bm_id}" w:name="{name}"/>'),
    )
    para._p.append(parse_xml(f'<w:bookmarkEnd {nsdecls("w")} w:id="{bm_id}"/>'))
    return name


def _first_para(doc: Document, text: str):
    for p in doc.paragraphs:
        if p.text.strip() == text:
            return p
    return None


def _ensure_front_bookmarks(doc: Document) -> list[str]:
    ensured: list[str] = []
    seen_names: set[str] = set()
    for heading, bm_name, bm_id in FRONT_BOOKMARKS:
        if bm_name in seen_names:
            continue
        para = _first_para(doc, heading)
        if para is None and heading == "ABSTRACT":
            para = _first_para(doc, "Abstract")
        if para is None:
            continue
        ensure_bookmark(para, bm_name, bm_id)
        ensured.append(bm_name)
        seen_names.add(bm_name)
    return ensured


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    dry_run = "--dry-run" in args
    args = [a for a in args if a != "--dry-run"]
    docx_path = Path(args[0]) if args else DEFAULT_DOCX
    if not docx_path.exists():
        print(f"文件不存在: {docx_path}")
        return 1

    if dry_run:
        doc = Document(str(docx_path))
        toc = _first_para(doc, "目录")
        fig = _first_para(doc, "图目录")
        tbl = _first_para(doc, "表目录")
        body_ch1 = None
        after_tbl = False
        for p in doc.paragraphs:
            t = p.text.strip()
            if tbl is not None and p._p is tbl._p:
                after_tbl = True
                continue
            if after_tbl and re.match(r"^第[一二三四五六七八九十]+章", t) and len(t) < 50:
                body_ch1 = p
                break
        print(f"dry-run: 目录={toc is not None} 图目录={fig is not None} 表目录={tbl is not None} 正文第一章={body_ch1 is not None}")
        if toc is None or fig is None:
            print("定位失败：缺少「目录」或「图目录」")
            return 1
        n = set_outline_levels(doc)
        print(f"大纲级别(未保存): {n}")
        bookmarks = _ensure_front_bookmarks(doc)
        print(f"前端书签(未保存): {bookmarks}")
        return 0

    bak = backup_docx(docx_path)
    print(f"备份: {bak}")
    doc = Document(str(docx_path))
    n = set_outline_levels(doc)
    print(f"大纲级别: {n}")

    toc_heading = _first_para(doc, "目录")
    fig_heading = _first_para(doc, "图目录")
    if toc_heading is None or fig_heading is None:
        print("未找到「目录」或「图目录」标题")
        return 1

    removed = clear_between(toc_heading._p, lambda t: t == "图目录")
    print(f"清空主目录区: {removed}")
    insert_toc_field(toc_heading._p)

    bookmarks = _ensure_front_bookmarks(doc)
    print(f"前端书签: {bookmarks}")

    # 图/表目录重建与完整收尾在 Task 3
    doc.save(str(docx_path))
    print(f"已保存: {docx_path}")
    print("请在 WPS/Word 中打开文档，右键目录→更新域。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
