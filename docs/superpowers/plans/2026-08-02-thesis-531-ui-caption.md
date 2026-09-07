# 5.3.1 操作说明与版式 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在论文 5.3.1 补全每图操作+意义说明，并缩放 Web/移动端插图。

**Architecture:** 用 python-docx 就地改 `华东师范大学硕士论文.docx` 的段落文本与 inline 图片 extent；图片文件不替换。

**Tech Stack:** Python 3、python-docx、EMU 尺寸计算

## Global Constraints

- 仅改 5.3.1（标题至 5.3.2 之前）
- Web 图宽 12.5cm；移动端 6.5cm；居中
- 每图一段；「登陆」→「登录」；图号不变

---

### Task 1: 改写 5.3.1 正文与图题

**Files:**
- Modify: `华东师范大学硕士论文.docx`
- Spec: `docs/superpowers/specs/2026-08-02-thesis-531-ui-caption-design.md`

- [x] Step 1: 定位段落区间（5.3.1 → 5.3.2）
- [x] Step 2: 更新节首总述（三入口底栏）
- [x] Step 3: 图题统一「登录」；为图5-4/5-6/5-7/5-8/5-9 插入或补齐说明段；校正图5-5、图5-10 说明
- [x] Step 4: 设置图片宽高（Web 12.5cm / 移动 6.5cm）并居中对齐
- [x] Step 5: 打印区间验收：每图后有说明；尺寸符合；5.3.2 未动
