# 5.4 系统实验与分析理顺 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 定点改写 `华东师范大学硕士论文.docx` 的 5.4 节：真表格、重生成并插入图5-11/5-12、理顺叙述、新增表5-4。

**Architecture:** 独立脚本 `scripts/polish_thesis_ch5_4.py` 备份 → 删除 5.4 至 5.5 前段落 → 插入正文/表格/图片；图表由同脚本用 matplotlib 生成到 `docs/superpowers/architecture/fig-5-11*.png` / `fig-5-12*.png`。

**Tech Stack:** Python 3、python-docx、matplotlib、Pillow

## Global Constraints

- 目标文件：`华东师范大学硕士论文.docx`；备份：`…bak-before-ch5-4-polish`
- 数字：`data/eval/table_6_3_summary_60.json`；不重跑实验、不虚构问卷
- 仅改 5.4；不整章重跑 merge；不加图5-13
- 不自动 git commit（除非用户另嘱）

---

### Task 1: 生成图5-11 / 图5-12

**Files:**
- Create: `scripts/generate_ch5_4_eval_figures.py`（或内嵌于 polish 脚本）
- Create: `docs/superpowers/architecture/fig-5-11-factual-60.png`
- Create: `docs/superpowers/architecture/fig-5-12-completeness-format-60.png`

- [x] **Step 1:** 用 JSON overall 画柱状图；中文字体 PingFang SC / Heiti SC；图内标题「图5-11…」「图5-12…」
- [x] **Step 2:** 目视确认无乱码、数值为 52.4/66.8/41.2/82.5 等

### Task 2: 定点改写 5.4

**Files:**
- Create: `scripts/polish_thesis_ch5_4.py`
- Modify: `华东师范大学硕士论文.docx`

- [x] **Step 1:** 实现备份、定位 5.4→5.5、删除区间、插入 h2/h3/body、4 张 Word 表、2 张图
- [x] **Step 2:** 运行 `python scripts/polish_thesis_ch5_4.py`
- [x] **Step 3:** 验收：表5-1～5-4 为真表；图已插入；无 `docs/superpowers/`；5.1–5.3 inline_shapes 未大幅减少；B3=82.5

### Task 3: 修改说明

**Files:**
- Create or update: `docs/论文修改内容说明_5-6章合并.md` 或短说明追加一段 5.4 polish

- [x] **Step 1:** 记录备份名、脚本、图表变化
