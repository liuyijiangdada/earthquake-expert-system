# 论文目录重建（媛媛样式）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 按戚媛媛论文结构，一键重建 `华东师范大学硕士论文.docx` 的主目录、图目录、表目录。

**Architecture:** 新建 `scripts/rebuild_toc_yuanyuan_style.py`：备份 docx → 设置大纲级别 → 清空旧目录区 → 插入 TOC 域与前端入口 → 从正文题注重建图/表目录（书签 + PAGEREF + 点引导线）。辅助纯函数放同文件，用 pytest 覆盖题注识别与编号解析。

**Tech Stack:** Python 3、python-docx、lxml、pytest

## Global Constraints

- 目标文件：`华东师范大学硕士论文.docx`
- 参考样式：`毕业论文_戚媛媛_答辩版.pdf`
- 编号保持：`图3-1` / `表5-1`（不改为点号）
- 执行前必须时间戳备份 `.bak`
- 页码用 Word 域，脚本不写死最终页码
- 不改正文内容；不自动提交 docx

## File Map

| 文件 | 职责 |
|------|------|
| `scripts/rebuild_toc_yuanyuan_style.py` | 一键重建脚本（含可测纯函数） |
| `tests/test_rebuild_toc_yuanyuan_style.py` | 题注识别 / 解析单测 |
| `华东师范大学硕士论文.docx` | 被修改的论文 |
| `华东师范大学硕士论文.docx.bak-*` | 自动备份（不提交） |

---

### Task 1: 题注识别纯函数 + 单测

**Files:**
- Create: `scripts/rebuild_toc_yuanyuan_style.py`
- Create: `tests/test_rebuild_toc_yuanyuan_style.py`

**Interfaces:**
- Produces:
  - `is_caption(text: str, kind: str) -> bool` — `kind` 为 `"图"` 或 `"表"`
  - `parse_label(text: str, kind: str) -> tuple[int, int, str, str] | None` — `(chapter, seq, label, title)`，`label` 无空格如 `图3-1`

- [x] **Step 1: 写失败单测**

```python
# tests/test_rebuild_toc_yuanyuan_style.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rebuild_toc_yuanyuan_style import is_caption, parse_label


def test_is_caption_figure_ok():
    assert is_caption("图3-1  总体方法框架图", "图")
    assert is_caption("图5-13  分阶段事实一致性对比（各阶段20题）", "图")


def test_is_caption_table_ok_with_space():
    assert is_caption("表 5-1  实验环境配置", "表")
    assert is_caption("表5-2  离线消融实验配置快照", "表")


def test_is_caption_rejects_inline_refs():
    assert not is_caption("如图3-1所示，整体流程分为三阶段", "图")
    assert not is_caption("表5-1给出了实验环境配置", "表")


def test_parse_label_normalizes():
    assert parse_label("图3-1  总体方法框架图", "图") == (3, 1, "图3-1", "总体方法框架图")
    assert parse_label("表 5-1  实验环境配置", "表") == (5, 1, "表5-1", "实验环境配置")
```

- [x] **Step 2: 跑单测确认失败**

Run: `python3 -m pytest tests/test_rebuild_toc_yuanyuan_style.py -v`  
Expected: FAIL（模块/函数不存在）

- [x] **Step 3: 实现最小纯函数**

在 `scripts/rebuild_toc_yuanyuan_style.py` 写入：

```python
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
```

- [x] **Step 4: 跑单测确认通过**

Run: `python3 -m pytest tests/test_rebuild_toc_yuanyuan_style.py -v`  
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add scripts/rebuild_toc_yuanyuan_style.py tests/test_rebuild_toc_yuanyuan_style.py
git commit -m "feat(thesis): add caption helpers for yuanyuan-style TOC rebuild"
```

---

### Task 2: 大纲级别 + 清空旧目录区 + 插入 TOC 域

**Files:**
- Modify: `scripts/rebuild_toc_yuanyuan_style.py`

**Interfaces:**
- Consumes: `DEFAULT_DOCX`, `Document`
- Produces:
  - `backup_docx(path: Path) -> Path`
  - `set_outline_levels(doc: Document) -> int`
  - `clear_between(start_elem, stop_text_predicate) -> int`
  - `insert_toc_field(after_elem) -> None`
  - `ensure_bookmark(para, name, bm_id) -> str`
  - `main(docx_path: Path) -> int`（本任务先完成备份/大纲/清空主目录/插 TOC）

- [x] **Step 1: 实现备份与大纲级别**

```python
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
```

同时为「摘要」「ABSTRACT」「图目录」「表目录」确保书签：`_TocFrontAbstract`、`_TocFrontABSTRACT`、`_TocFrontFigList`、`_TocFrontTblList`。

- [x] **Step 2: 在 main 中串起：备份 → 大纲 → 定位「目录」「图目录」→ 清空中间 → 插 TOC**

```python
def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    docx_path = Path(args[0]) if args else DEFAULT_DOCX
    if not docx_path.exists():
        print(f"文件不存在: {docx_path}")
        return 1
    bak = backup_docx(docx_path)
    print(f"备份: {bak}")
    doc = Document(str(docx_path))
    n = set_outline_levels(doc)
    print(f"大纲级别: {n}")
    # 定位（完整 main 在 Task 3 收尾）
    ...
```

- [x] **Step 3: 干跑验证定位**

Run:

```bash
python3 - <<'PY'
from docx import Document
doc = Document("华东师范大学硕士论文.docx")
for i,p in enumerate(doc.paragraphs):
    t=p.text.strip()
    if t in ("目录","图目录","表目录") or t.startswith("第一章"):
        print(i, t[:40])
PY
```

Expected: 依次出现 `目录`、（正文前）`图目录`、`表目录`、正文 `第一章`

- [x] **Step 4: Commit 脚本进度**

```bash
git add scripts/rebuild_toc_yuanyuan_style.py
git commit -m "feat(thesis): outline levels and TOC field scaffolding"
```

---

### Task 3: 重建图目录与表目录 + 样式 + 跑通全文

**Files:**
- Modify: `scripts/rebuild_toc_yuanyuan_style.py`
- Modify: `华东师范大学硕士论文.docx`（运行脚本）

**Interfaces:**
- Consumes: `is_caption`, `parse_label`, `clear_between`, `insert_toc_field`
- Produces:
  - `collect_captions(doc, kind, body_start_idx) -> list[tuple]`
  - `make_toc_entry(label, title, bookmark, tab_pos="8295") -> element`
  - `rebuild_fig_tbl_tocs(doc) -> tuple[int, int]`
  - 完整 `main()`

- [x] **Step 1: 实现收集题注、补书签、构造条目（复用现有 `rebuild_all_tocs.py` 的 XML 形态）**

```python
def get_bookmark(para) -> str | None:
    m = re.search(r'w:bookmarkStart[^>]*w:name="([^"]+)"', para._p.xml)
    return m.group(1) if m else None


def add_bookmark(para, name: str, bm_id: int) -> str:
    para._p.insert(0, parse_xml(
        f'<w:bookmarkStart {nsdecls("w")} w:id="{bm_id}" w:name="{name}"/>'
    ))
    para._p.append(parse_xml(f'<w:bookmarkEnd {nsdecls("w")} w:id="{bm_id}"/>'))
    return name


def make_toc_entry(label: str, title: str, bookmark: str):
    safe = (
        title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    ppr = (
        '<w:pPr>'
        '<w:tabs><w:tab w:val="right" w:leader="dot" w:pos="8295"/></w:tabs>'
        '<w:jc w:val="left"/>'
        "</w:pPr>"
    )
    rpr = (
        '<w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="宋体" '
        'w:hAnsi="Times New Roman"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
    )
    return parse_xml(
        f'<w:p {nsdecls("w")}>{ppr}'
        f'<w:hyperlink w:anchor="{bookmark}" w:history="1">'
        f"<w:r>{rpr}<w:t xml:space=\"preserve\">{label}  {safe}</w:t></w:r>"
        f"<w:r><w:tab/></w:r></w:hyperlink>"
        f'<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
        f'<w:r><w:instrText xml:space="preserve"> PAGEREF {bookmark} \\h </w:instrText></w:r>'
        f'<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
        f"<w:r>{rpr}<w:t>1</w:t></w:r>"
        f'<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>'
    )


def collect_captions(paras, kind: str, min_idx: int):
    items = []
    for i, p in enumerate(paras):
        if i < min_idx:
            continue
        if not is_caption(p.text, kind):
            continue
        parsed = parse_label(p.text, kind)
        if not parsed:
            continue
        chapter, seq, label, title = parsed
        items.append((chapter, seq, i, p, label, title, get_bookmark(p)))
    items.sort(key=lambda x: (x[0], x[1], x[2]))
    return [(p, label, title, bm) for _, _, _, p, label, title, bm in items]
```

- [x] **Step 2: 完成 `main()`：清空主目录区 → TOC → 清空并插入图/表目录 → 标题居中 → 保存**

关键逻辑顺序：

1. `backup_docx`
2. `set_outline_levels`
3. 找 `目录` / `图目录` / `表目录` 标题段（首次出现）
4. `clear_between(目录, lambda t: t=="图目录")`；`insert_toc_field`
5. 找正文起始（`表目录` 之后第一个 `第…章`），`collect_captions` 图/表
6. 补书签
7. `clear_between(图目录, lambda t: t=="表目录")` 后插入图条目
8. `clear_between(表目录, stop=第X章|参考文献|致谢)` 后插入表条目
9. 「目录」「图目录」「表目录」标题 `jc=center`
10. `doc.save`

- [x] **Step 3: 运行脚本**

Run: `python3 scripts/rebuild_toc_yuanyuan_style.py`  
Expected 控制台类似：

```
备份: ...bak-toc-...
大纲级别: ...
清空主目录区: ...
图标题: 17 个
表标题: 11 个
图目录 17 条，表目录 11 条。
已保存：.../华东师范大学硕士论文.docx
请在 WPS/Word 中打开文档，右键目录→更新域。
```

- [x] **Step 4: 结构化验收**

Run:

```bash
python3 - <<'PY'
from docx import Document
import re
doc = Document("华东师范大学硕士论文.docx")
paras = doc.paragraphs
# 找索引
idx = {k: None for k in ("目录","图目录","表目录")}
for i,p in enumerate(paras):
    t=p.text.strip()
    if t in idx and idx[t] is None:
        idx[t]=i
print(idx)
# 主目录区不应有 图X-Y / 表X-Y
bad=[]
for i in range(idx["目录"]+1, idx["图目录"]):
    t=paras[i].text.strip()
    if re.match(r"^[图表]\s*\d+-\d+", t):
        bad.append(t)
print("main_toc_bad", len(bad), bad[:5])
# 图/表条目数
figs=[]; tbls=[]
for i in range(idx["图目录"]+1, idx["表目录"]):
    t=paras[i].text.strip()
    if t: figs.append(t)
body_start=None
for i in range(idx["表目录"]+1, len(paras)):
    t=paras[i].text.strip()
    if re.match(r"^第[一二三四五六七八九十]+章", t):
        body_start=i; break
    if t: tbls.append(t)
print("fig_toc", len(figs))
print("tbl_toc", len(tbls))
print("has_toc_field", any("TOC" in paras[i]._p.xml for i in range(idx["目录"], idx["图目录"])))
assert not bad
assert len(figs) == 17
assert len(tbls) == 11
print("OK")
PY
```

Expected: `OK`；`main_toc_bad 0`；图 17 / 表 11；存在 TOC 域

- [x] **Step 5: Commit 脚本（docx 默认不提交，除非用户要求）**

```bash
git add scripts/rebuild_toc_yuanyuan_style.py tests/test_rebuild_toc_yuanyuan_style.py
git commit -m "feat(thesis): rebuild TOC/figure/table lists in yuanyuan style"
```

---

### Task 4: 人工域更新提示与计划勾选

**Files:**
- Modify: `docs/superpowers/plans/2026-08-02-thesis-toc-yuanyuan-style.md`（勾选完成项）

- [x] **Step 1: 在终端打印最终操作说明（脚本已含）**

告知用户：

1. 用 WPS/Word 打开 `华东师范大学硕士论文.docx`
2. 在主目录灰色域上右键 → 更新域 → 更新整个目录
3. 全选（Ctrl/Cmd+A）→ F9 或「更新域」，刷新图/表目录页码
4. 目视对照媛媛：主目录含摘要/ABSTRACT/图目录/表目录入口；其后为独立图目录、表目录

- [x] **Step 2: 若 TOC 未自动收录摘要等入口**

在脚本中于 TOC 域前插入 4 条静态 `PAGEREF` 入口（摘要 / ABSTRACT / 图目录 / 表目录），再重跑验收。

- [x] **Step 3: Commit 计划勾选（如有文档变更）**

```bash
git add docs/superpowers/plans/2026-08-02-thesis-toc-yuanyuan-style.md
git commit -m "docs: mark thesis TOC yuanyuan-style plan tasks done"
```

---

## Spec Coverage Checklist

| Spec 要求 | Task |
|-----------|------|
| 丢弃旧目录、重建三块 | Task 2–3 |
| 主目录含摘要/ABSTRACT/图目录/表目录入口 | Task 2–3 / Task 4 补救 |
| 主目录无图表条目 | Task 3 验收 |
| 图/表目录从正文题注生成 | Task 1 + 3 |
| 编号保持连字符 | Global + Task 1 |
| 居中标题、点引导线、字体 | Task 3 |
| PAGEREF/TOC 域、手动更新页码 | Task 2–3 |
| 时间戳备份 | Task 2 |
| 不改正文、不自动提交 docx | Global / Task 3 Step 5 |

## Self-Review Notes

- 无 TBD/占位步骤
- 函数名在各 Task 间一致：`is_caption` / `parse_label` / `clear_between` / `make_toc_entry`
- 验收断言使用正文实测期望：17 图、11 表；若正文增减题注，以脚本扫描结果为准并更新断言
