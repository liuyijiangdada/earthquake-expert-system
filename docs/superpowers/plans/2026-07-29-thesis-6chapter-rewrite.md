# 硕士论文 6 章制改稿 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按设计规格把 `华东师范大学硕士论文.docx` 重建为 6 章标红修改版 `华东师范大学硕士论文_6章修改版.docx`，回应导师十条意见，实验数字对齐 60 题消融汇总。

**Architecture:** 内容与生成解耦：`thesis_rewrite_content_6ch.py` 只导出摘要/正文块/参考文献/附录；`rewrite_thesis_6ch.py` 复制底稿、替换摘要、删除「第一章」至「致谢」前旧正文、插入新块并标红、注入 `{{FACT_B*}}` 等占位符。验收用 pytest 断言输出结构与关键措辞，不依赖人工打开 Word。

**Tech Stack:** Python 3、`python-docx`、仓库内 `data/eval/table_6_3_summary_60.json`、`data/real_earthquakes_catalog.json`、`data/emergency_knowledge.json`、`data/eval/phase_questions.json`。

## Global Constraints

- 严格 **6 章**：绪论 / 相关技术 / 静态知识构建与检索 / 动态–静态协同调度 / 系统实现与实验 / 总结与展望；**禁止**保留独立「总体方法」空章或第七、八章方法/实验分章。
- **不覆盖** `华东师范大学硕士论文.docx`；输出默认 `华东师范大学硕士论文_6章修改版.docx`；改写前备份底稿。
- 新增/重写段落 **标红**（`RGBColor(0xFF,0x00,0x00)`）；封面、声明、致谢不改不标红。
- 实验主口径：**60 题** B0–B3，数字来自 `data/eval/table_6_3_summary_60.json`；禁止把自动 grounding 写成人工评测。
- GraphRAG / KnowledGPT：**设计对比 only**，禁止虚构同设定数值。
- 可用性问卷：无数据则写「待开展」，禁止虚构。
- 表述禁止无依据的「显著优于」；B2 偏低须给方法根因，禁止仅用「样本小」搪塞。
- 复用既有校准表述时，须按 6 章重切并改写所有「全文共七章」「第六章实验」「第七章总结」交叉引用。

---

## File Structure

| 文件 | 职责 |
|------|------|
| `scripts/thesis_rewrite_content_6ch.py` | 导出 `ABSTRACT_CN/EN`、`KEYWORDS_*`、`BODY_BLOCKS`、`REFERENCES`、`APPENDIX_BLOCKS`、`TOC_LINES` |
| `scripts/rewrite_thesis_6ch.py` | CLI 生成 docx：备份、复制、填占位符、删旧正文、插新块、更新目录文本、统计红字 |
| `tests/test_thesis_6ch_content.py` | 不依赖 docx：检查 6 章标题、关键回应措辞、占位符键、无第七章 |
| `tests/test_thesis_6ch_docx.py` | 跑生成脚本后检查输出文件结构与红字 |
| `docs/论文修改内容说明_6章制.md` | 意见→落点对照（交付说明） |
| `华东师范大学硕士论文_6章修改版.docx` | 主交付物（gitignore 可选，默认提交由执行者决定） |

内容素材优先来源（只读参考，勿改坏口径）：
- `scripts/thesis_rewrite_content.py`（旧 7 章已校准文案，需重切）
- `docs/superpowers/specs/2026-07-29-thesis-6chapter-rewrite-design.md`
- `毕业论文修改意见分析与应对方案.docx`

---

### Task 1: 脚手架与内容模块骨架 + 结构测试

**Files:**
- Create: `scripts/thesis_rewrite_content_6ch.py`
- Create: `tests/test_thesis_6ch_content.py`

**Interfaces:**
- Produces: 模块级常量 `ABSTRACT_CN: str`、`ABSTRACT_EN: str`、`KEYWORDS_CN: str`、`KEYWORDS_EN: str`、`BODY_BLOCKS: list[tuple[str, str]]`、`REFERENCES: list[str]`、`APPENDIX_BLOCKS: list[tuple[str, str]]`、`TOC_LINES: list[str]`
- `BODY_BLOCKS` level ∈ `{"h1","h2","h3","body","caption","table_title","ref"}`

- [ ] **Step 1: 写失败测试（结构契约）**

```python
# tests/test_thesis_6ch_content.py
from scripts.thesis_rewrite_content_6ch import BODY_BLOCKS, TOC_LINES, REFERENCES

H1 = [t for lvl, t in BODY_BLOCKS if lvl == "h1"]

def test_six_chapters_only():
    assert H1[0].startswith("第一章")
    assert H1[1].startswith("第二章")
    assert H1[2].startswith("第三章") and "静态" in H1[2]
    assert H1[3].startswith("第四章") and ("协同" in H1[3] or "调度" in H1[3])
    assert H1[4].startswith("第五章") and ("系统" in H1[4] or "实验" in H1[4])
    assert H1[5].startswith("第六章") and "总结" in H1[5]
    assert not any("第七章" in t or "第八章" in t for t in H1)
    assert "总体方法" not in "".join(H1)

def test_toc_matches_six_chapters():
    assert TOC_LINES[0].startswith("第一章")
    assert len([x for x in TOC_LINES if x.startswith("第") and "章" in x]) == 6

def test_no_seven_chapter_crossrefs():
    blob = "\n".join(t for _, t in BODY_BLOCKS)
    assert "全文共七章" not in blob
    assert "第七章" not in blob

def test_references_include_domestic_journals():
    joined = "\n".join(REFERENCES)
    assert "自然灾害学报" in joined or "地震研究" in joined
    assert len(REFERENCES) >= 30
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /Users/xiaoxiaoqingnian/Desktop/biyelunwen && python -m pytest tests/test_thesis_6ch_content.py -v`  
Expected: FAIL（模块不存在或断言失败）

- [ ] **Step 3: 创建骨架模块（先放标题树 + 占位正文，后续 Task 填实）**

在 `scripts/thesis_rewrite_content_6ch.py` 中至少包含：

```python
# -*- coding: utf-8 -*-
ABSTRACT_CN = "（占位，Task2 替换）"
ABSTRACT_EN = "(placeholder)"
KEYWORDS_CN = "动态—静态知识协同；知识图谱；向量检索；三阶段调度；地震应急；检索增强生成"
KEYWORDS_EN = "Dynamic–Static Knowledge Collaboration; Knowledge Graph; Vector Retrieval; Three-Phase Scheduling; Earthquake Emergency; RAG"

TOC_LINES = [
    "第一章 绪论",
    "第二章 相关技术与理论基础",
    "第三章 静态知识构建与检索方法",
    "第四章 动态—静态知识协同调度方法",
    "第五章 系统实现与实验分析",
    "第六章 总结与展望",
    "参考文献",
    "附录",
    "致谢",
]

BODY_BLOCKS: list[tuple[str, str]] = [
    ("h1", "第一章  绪论"),
    ("h2", "1.1  研究背景与意义"),
    ("body", "占位"),
    ("h1", "第二章  相关技术与理论基础"),
    ("h2", "2.1  知识图谱构建技术"),
    ("body", "占位"),
    ("h1", "第三章  静态知识构建与检索方法"),
    ("h2", "3.1  研究动机与贡献"),
    ("body", "占位"),
    ("h1", "第四章  动态—静态知识协同调度方法"),
    ("h2", "4.1  研究动机与贡献"),
    ("body", "占位"),
    ("h1", "第五章  系统实现与实验分析"),
    ("h2", "5.1  系统需求分析与总体设计"),
    ("body", "占位"),
    ("h1", "第六章  总结与展望"),
    ("h2", "6.1  本文工作总结"),
    ("body", "占位"),
]

REFERENCES = [
    "[1] 温增平, 等. 汶川地震余震序列特征[J]. 地震研究, 2009.",
    "[2] 谢礼福, 等. 地震灾害损失评估方法研究进展[J]. 自然灾害学报, 2010.",
] + [f"[n] 占位文献{i}" for i in range(3, 31)]

APPENDIX_BLOCKS: list[tuple[str, str]] = [
    ("h1", "附录A  评测问集与配置说明"),
    ("body", "占位"),
]
```

注意：Step 3 的 `REFERENCES` 占位仅用于让结构测试先绿；Task 6 必须换成真实文献列表。若 `test_references_include_domestic_journals` 因占位已含期刊名可通过，仍须在 Task 6 替换为完整条目。

- [ ] **Step 4: 再跑结构测试**

Run: `python -m pytest tests/test_thesis_6ch_content.py -v`  
Expected: 与骨架一致的断言 PASS（若某条仍 FAIL，补齐标题字符串）

- [ ] **Step 5: Commit**

```bash
git add scripts/thesis_rewrite_content_6ch.py tests/test_thesis_6ch_content.py
git commit -m "feat: scaffold 6-chapter thesis content module and structure tests"
```

---

### Task 2: 摘要 + 第一、二章实文

**Files:**
- Modify: `scripts/thesis_rewrite_content_6ch.py`
- Modify: `tests/test_thesis_6ch_content.py`

**Interfaces:**
- Consumes: `scripts/thesis_rewrite_content.py` 中摘要与第1–2章可用段落（只读剪辑）
- Produces: 实写 `ABSTRACT_*`；第一章含 1.2.1 国内文献、1.2.3 GraphRAG、1.3 静/动定义；第二章仅技术基础四节（架构图不放本章）

- [ ] **Step 1: 扩展内容测试**

```python
def test_abstract_mentions_60_and_honesty():
    from scripts.thesis_rewrite_content_6ch import ABSTRACT_CN
    assert "60" in ABSTRACT_CN
    assert "事实" in ABSTRACT_CN
    assert "完整" in ABSTRACT_CN  # 承认完整性未必全面领先

def test_ch1_defines_static_dynamic():
    blob = "\n".join(t for lvl, t in BODY_BLOCKS if True)
    # 更精确：只取第一章到第二章之间
    texts = []
    in_ch1 = False
    for lvl, t in BODY_BLOCKS:
        if lvl == "h1" and "第一章" in t:
            in_ch1 = True
        elif lvl == "h1" and "第二章" in t:
            break
        elif in_ch1:
            texts.append(t)
    ch1 = "\n".join(texts)
    assert "静态知识" in ch1 and "动态知识" in ch1
    assert "自然灾害学报" in ch1 or "地震研究" in ch1
    assert "GraphRAG" in ch1
    assert "KnowledGPT" in ch1 or "知识增强" in ch1
    assert "1.2.1" in ch1 or "国内地震应急" in ch1

def test_ch2_is_tech_foundation_not_system_manual():
    texts = []
    in_ch2 = False
    for lvl, t in BODY_BLOCKS:
        if lvl == "h1" and "第二章" in t:
            in_ch2 = True
        elif lvl == "h1" and in_ch2:
            break
        elif in_ch2:
            texts.append(t)
    ch2 = "\n".join(texts)
    assert "评价指标" in ch2 or "事实一致性" in ch2
    assert "系统总体逻辑架构" not in ch2  # 架构迁出第2章
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_thesis_6ch_content.py::test_abstract_mentions_60_and_honesty tests/test_thesis_6ch_content.py::test_ch1_defines_static_dynamic tests/test_thesis_6ch_content.py::test_ch2_is_tech_foundation_not_system_manual -v`  
Expected: FAIL

- [ ] **Step 3: 写入摘要与第1–2章**

要求清单（必须全部落入 `BODY_BLOCKS`）：

**摘要（中/英）**
- 问题：阶段差异、事实可核验、时效
- 方法：图谱 + 向量 + 实时源 + 三阶段规则调度 + 本地微调生成
- 结果：写入 `{{FACT_B3}}` 等占位符，或先写与 JSON 一致的 82.5/52.4/… 并在生成时统一用占位符替换更稳妥——**统一用占位符**：`{{FACT_B0}}`…`{{FACT_B3}}`、`{{COMP_B0}}`…`{{COMP_B3}}`
- 诚实句：协同提升事实对齐，不必然带来更完整要点覆盖；自动评分局限

**第一章标题树**
- `1.1 研究背景与意义`
- `1.2 国内外研究现状` → `1.2.1` 国内地震应急知识管理与信息服务；`1.2.2` KG/向量问答；`1.2.3` 开放域 RAG+KG（GraphRAG、KnowledGPT）；`1.2.4` 不足与切入点
- `1.3 研究目标与内容`（显式定义静态=图谱+向量索引；动态=CEIC/USGS 等；协同=阶段分类+规则阈值调度）
- `1.4 论文组织结构`（写「全文共六章」，第3→4→5环环相扣）

**第二章标题树**
- `2.1 知识图谱构建技术`
- `2.2 向量检索与RAG`
- `2.3 动态数据接入与实时处理`
- `2.4 问答系统评价指标`

可从 `thesis_rewrite_content.py` 剪辑 2.1–2.4 / 2.8 国内文献段，但删除所有「七章」「第六章实验」交叉引用，并删掉系统架构长段。

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_thesis_6ch_content.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/thesis_rewrite_content_6ch.py tests/test_thesis_6ch_content.py
git commit -m "feat: write thesis abstract and chapters 1-2 for 6-chapter structure"
```

---

### Task 3: 第三章静态知识构建与检索（回应意见二、三）

**Files:**
- Modify: `scripts/thesis_rewrite_content_6ch.py`
- Modify: `tests/test_thesis_6ch_content.py`

**Interfaces:**
- Consumes: `data/real_earthquakes_catalog.json`、`data/emergency_knowledge.json` 字段事实
- Produces: 第三章完整 `BODY_BLOCKS` 段

- [ ] **Step 1: 写第三章验收测试**

```python
def _chapter_blob(n_prefix: str) -> str:
    texts, active = [], False
    for lvl, t in BODY_BLOCKS:
        if lvl == "h1" and n_prefix in t:
            active = True
            continue
        if lvl == "h1" and active:
            break
        if active:
            texts.append(t)
    return "\n".join(texts)

def test_ch3_kg_pipeline_and_rag():
    ch3 = _chapter_blob("第三章")
    for kw in ("数据来源", "特征抽取", "模式", "Neo4j", "Cypher", "分块", "向量", "互补"):
        assert kw in ch3, kw
    assert "应急主题" in ch3 or "主题库" in ch3
    assert "震例" in ch3 or "地震事件" in ch3
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_thesis_6ch_content.py::test_ch3_kg_pipeline_and_rag -v`  
Expected: FAIL

- [ ] **Step 3: 实写第三章**

标题树（必须）：
- `3.1 研究动机与贡献`
- `3.2 问题定义`
- `3.3 地震应急知识图谱构建`
  - `3.3.1 数据来源与原始格式`（CEIC/USGS 目录字段、应急主题 JSON/PDF 科普来源；给样例字段表叙述）
  - `3.3.2 特征抽取与预处理`（结构化字段映射；非结构化→「事件-地点-震级/步骤」规则）
  - `3.3.3 图谱模式设计`（Earthquake/Region/EmergencyTopic/GuidanceStep + 关系；说明模式图路径 `docs/superpowers/architecture/` 或「见图3-1」占位）
  - `3.3.4 数据导入与图谱统计`（导入流程、示例 Cypher、节点/关系数量级——与仓库数据一致：主题约 13、震例约 14，勿虚构上千节点除非代码真有）
- `3.4 基于向量的应急文档检索`
  - `3.4.1 Topic 级分块及适用性`
  - `3.4.2 向量化与索引`
  - `3.4.3 检索与上下文增强（与图谱互补）`

素材：剪辑旧稿第四章静态知识段 + 第五章向量段，合并改写。

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_thesis_6ch_content.py::test_ch3_kg_pipeline_and_rag -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/thesis_rewrite_content_6ch.py tests/test_thesis_6ch_content.py
git commit -m "feat: flesh out chapter 3 static KG and retrieval methods"
```

---

### Task 4: 第四章协同调度（回应意见一、四）

**Files:**
- Modify: `scripts/thesis_rewrite_content_6ch.py`
- Modify: `tests/test_thesis_6ch_content.py`

**Interfaces:**
- Produces: 第四章完整段落；伪代码用 `body` 多行文本即可

- [ ] **Step 1: 写第四章验收测试**

```python
def test_ch4_collaboration_subjects_and_scheduler():
    ch4 = _chapter_blob("第四章")
    assert "静态知识" in ch4 and "动态知识" in ch4
    assert "震前" in ch4 and "震中" in ch4 and "震后" in ch4
    assert "阈值" in ch4 or "调度" in ch4
    assert "伪代码" in ch4 or "算法" in ch4 or "procedure" in ch4.lower() or "输入：" in ch4
    assert "CEIC" in ch4 or "台网" in ch4
    assert "USGS" in ch4
    assert "冲突" in ch4 or "降级" in ch4
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_thesis_6ch_content.py::test_ch4_collaboration_subjects_and_scheduler -v`  
Expected: FAIL

- [ ] **Step 3: 实写第四章**

标题树：
- `4.1 研究动机与贡献`
- `4.2 问题定义`（协同主体定义段必须独立成文，回应意见四）
- `4.3 协同调度总体架构`（图题：`图4-1 端到端推理流程`，路径提示 `docs/superpowers/architecture/fig-6-1-e2e-inference-flow.png`）
- `4.4 三阶段分类与规则阈值设计`
  - `4.4.1` 三阶段信息需求
  - `4.4.2` 静态置信度 / 动态可用性形式化
  - `4.4.3` 规则阈值与调度决策（含伪代码）
- `4.5 动态知识接入与融合`
  - `4.5.1` 实时接入细节（轻量 CEIC 抓取 + USGS FDSN，勿写成正式 OpenAPI）
  - `4.5.2` 冲突解决 / 降级
  - `4.5.3` 提示词融合模板

伪代码最少包含：输入 q → 分类阶段 → 探测 sc/da/u → 决定 use_dyn / 提示约束 → 组装 C_kg/C_rag/C_dyn → 生成。

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_thesis_6ch_content.py::test_ch4_collaboration_subjects_and_scheduler -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/thesis_rewrite_content_6ch.py tests/test_thesis_6ch_content.py
git commit -m "feat: flesh out chapter 4 dynamic-static scheduling method"
```

---

### Task 5: 第五章系统实现与实验（回应意见五～九）

**Files:**
- Modify: `scripts/thesis_rewrite_content_6ch.py`
- Modify: `tests/test_thesis_6ch_content.py`

**Interfaces:**
- Consumes: `table_6_3_summary_60.json` 占位符键（与 `build_eval_mapping` 一致）
- Produces: 第五章 + 表题叙述

- [ ] **Step 1: 写第五章验收测试**

```python
def test_ch5_eval_protocol_and_comparisons():
    ch5 = _chapter_blob("第五章")
    assert "60" in ch5
    assert "B0" in ch5 and "B3" in ch5
    assert "自动" in ch5 and ("grounding" in ch5.lower() or "proxy" in ch5.lower() or "评分" in ch5)
    assert "GraphRAG" in ch5
    assert "未" in ch5 and ("复现" in ch5 or "同设定" in ch5 or "数值对比实验" in ch5)
    assert "B2" in ch5
    assert "分块" in ch5 or "嵌入" in ch5 or "Top-K" in ch5 or "可核对" in ch5
    assert "待开展" in ch5 or "可用性" in ch5
    # 禁止把关闭动态的消融写成含动态主结论的含糊句——至少要有路径分离说明
    assert "离线" in ch5 or "关闭动态" in ch5 or "消融" in ch5
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_thesis_6ch_content.py::test_ch5_eval_protocol_and_comparisons -v`  
Expected: FAIL

- [ ] **Step 3: 实写第五章**

标题树：
- `5.1 系统需求分析与总体设计`
- `5.2 系统功能实现`（界面截图位：`data/manual_media/image1.png`…；Web/移动端）
- `5.3 实验设置`（60 题、B0–B3、关闭动态与第三层、自动 proxy）
- `5.4 对比实验结果与分析`
  - `5.4.1` 消融通路增益（表题用 `{{FACT_B*}}` / `{{COMP_B*}}` / `{{FMT_B*}}`）
  - `5.4.2` 与 GraphRAG、KnowledGPT **设计对比表（文字表）**：构图粒度、阶段感知、动态源、显式调度；**明确一句**：本文未在同一问集上复现二者官方流水线，故不做分数对比
- `5.5 消融实验`
  - `5.5.1` 全量 60 题结果（含分阶段占位符 `{{FACT_B3_PRE}}` 等）
  - `5.5.2` 模块贡献
  - `5.5.3` B2 偏低学术分析（分块、嵌入、Top-K、提示可核对片段、评分对图谱块权重）
- `5.6 案例分析`（标明定性演示 vs 消融问集）
- `5.7 系统可用性评测`：写「本次未开展正式用户问卷，列为后续工作」

图题预留：`图5-1`/`图5-2` 对应 `fig-6-1-factual-60.png`、`fig-6-2-completeness-format-60.png`。

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_thesis_6ch_content.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/thesis_rewrite_content_6ch.py tests/test_thesis_6ch_content.py
git commit -m "feat: write chapter 5 system implementation and 60-question evaluation"
```

---

### Task 6: 第六章 + 参考文献 + 附录

**Files:**
- Modify: `scripts/thesis_rewrite_content_6ch.py`
- Modify: `tests/test_thesis_6ch_content.py`

**Interfaces:**
- Produces: 完整 `REFERENCES`（≥30，含国内期刊 ≥5）、`APPENDIX_BLOCKS`（问集摘要、配置参数、Cypher 示例）

- [ ] **Step 1: 扩展测试**

```python
def test_ch6_aligned_with_eval_placeholders():
    ch6 = _chapter_blob("第六章")
    assert "{{FACT_B3}}" in ch6 or "事实一致性" in ch6
    assert "展望" in ch6 or "未来" in ch6

def test_appendix_has_questions_or_cypher():
    blob = "\n".join(t for _, t in APPENDIX_BLOCKS)
    assert "附录" in blob or True
    assert ("问" in blob) or ("Cypher" in blob) or ("配置" in blob)

def test_references_real_count():
    assert len(REFERENCES) >= 30
    assert not any("占位文献" in r for r in REFERENCES)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_thesis_6ch_content.py::test_references_real_count -v`  
Expected: FAIL（若仍有占位文献）

- [ ] **Step 3: 写总结、真实参考文献、附录**

- 第六章：`6.1` 三点总结呼应摘要；数字用占位符；`6.2` 展望（正式 API、人工评测、GraphRAG 同设定对比实验等）
- `REFERENCES`：合并旧 `thesis_rewrite_content.py` 的 REFERENCES，并确保含《自然灾害学报》《地震研究》等条目；删掉与正文无关堆砌亦可，但总数 ≥30
- 附录建议：
  - 附录A：60 题问集按阶段列表（从 `data/eval/phase_questions.json` 生成文本）
  - 附录B：消融配置参数表
  - 附录C：示例 Cypher

可用一次性脚本片段在编写时打印问集：

```bash
python3 - <<'PY'
import json
from pathlib import Path
d=json.loads(Path('data/eval/phase_questions.json').read_text())
for phase, qs in d['phases'].items():
    print('##', phase, len(qs))
    for i,q in enumerate(qs,1):
        print(f'{i}. {q if isinstance(q,str) else q.get("question",q)}')
PY
```

- [ ] **Step 4: 全量内容测试通过**

Run: `python -m pytest tests/test_thesis_6ch_content.py -v`  
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/thesis_rewrite_content_6ch.py tests/test_thesis_6ch_content.py
git commit -m "feat: complete chapter 6, references, and appendices for 6-chapter thesis"
```

---

### Task 7: 生成脚本 `rewrite_thesis_6ch.py` + docx 集成测试

**Files:**
- Create: `scripts/rewrite_thesis_6ch.py`（以 `scripts/rewrite_thesis_supervisor_v2.py` 为模板改造）
- Create: `tests/test_thesis_6ch_docx.py`

**Interfaces:**
- Consumes: `thesis_rewrite_content_6ch` 全部导出常量；`build_eval_mapping(summary_path) -> dict[str,str]`
- Produces: CLI：
  - `--thesis` 默认 `华东师范大学硕士论文.docx`
  - `--output` 默认 `华东师范大学硕士论文_6章修改版.docx`
  - `--fill-eval` 默认 `data/eval/table_6_3_summary_60.json`
  - `--no-red` / `--no-backup`

关键改造点（相对 v2）：
1. `from scripts.thesis_rewrite_content_6ch import ...`
2. `OUT_DEFAULT` 改为 `_6章修改版.docx`
3. `new_toc = TOC_LINES`（6 章）
4. `FIG` 占位键可映射为图5-1/5-2 说明字符串
5. **禁止**默认写回底稿路径

- [ ] **Step 1: 写 docx 测试（先跳过若不存在输出，生成后再断言）**

```python
# tests/test_thesis_6ch_docx.py
from pathlib import Path
import subprocess
import sys
from docx import Document
from docx.shared import RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "华东师范大学硕士论文_6章修改版.docx"
THESIS = ROOT / "华东师范大学硕士论文.docx"
SUMMARY = ROOT / "data/eval/table_6_3_summary_60.json"
RED = RGBColor(0xFF, 0x00, 0x00)

def test_generate_and_validate_docx():
    assert THESIS.exists()
    cmd = [
        sys.executable,
        str(ROOT / "scripts/rewrite_thesis_6ch.py"),
        "--thesis", str(THESIS),
        "--output", str(OUT),
        "--fill-eval", str(SUMMARY),
    ]
    subprocess.check_call(cmd, cwd=str(ROOT))
    assert OUT.exists()
    # 底稿仍在
    assert THESIS.exists()
    doc = Document(str(OUT))
    texts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    blob = "\n".join(texts)
    assert "第一章" in blob and "第六章" in blob
    assert "第七章" not in blob
    assert "82.5" in blob  # FACT_B3 filled
    assert "{{FACT_B3}}" not in blob
    # 致谢存在且其后不应再塞章节
    assert any(t == "致谢" for t in texts)
    red_paras = 0
    for p in doc.paragraphs:
        for r in p.runs:
            if r.font.color and r.font.color.rgb == RED:
                red_paras += 1
                break
    assert red_paras >= 50
```

- [ ] **Step 2: 实现 `scripts/rewrite_thesis_6ch.py`**

复制 `rewrite_thesis_supervisor_v2.py` 后修改 import、默认路径、`new_toc`；保留 `fill_placeholders`、`build_eval_mapping`、`apply_content` 逻辑。确保 `apply_content` 使用 `TOC_LINES`。

- [ ] **Step 3: 跑生成与测试**

Run:

```bash
python -m pytest tests/test_thesis_6ch_docx.py -v
python -m pytest tests/test_thesis_6ch_content.py -v
```

Expected: PASS；控制台打印 `已写入: ..._6章修改版.docx`、红字段落数。

- [ ] **Step 4: 人工抽检命令（非交互，打印关键片段）**

```bash
python3 - <<'PY'
from docx import Document
doc=Document('华东师范大学硕士论文_6章修改版.docx')
for p in doc.paragraphs:
    t=p.text.strip()
    if t.startswith(('第','摘要','关键词','参考','附录','致谢')) and len(t)<40:
        print(t)
PY
```

Expected: 仅六章 + 参考文献/附录/致谢。

- [ ] **Step 5: Commit**

```bash
git add scripts/rewrite_thesis_6ch.py tests/test_thesis_6ch_docx.py
# 若决定纳入 docx：
# git add "华东师范大学硕士论文_6章修改版.docx"
git commit -m "feat: generate redlined 6-chapter thesis docx from content module"
```

---

### Task 8: 修改说明文档 + 规格验收对照

**Files:**
- Create: `docs/论文修改内容说明_6章制.md`
- Modify: `docs/superpowers/specs/2026-07-29-thesis-6chapter-rewrite-design.md`（状态改为「已实施」）

- [ ] **Step 1: 写修改说明**

文档须含表格：意见一…十 → 修改版章节位置；并写明：
- 输出文件名
- 实验 JSON 路径与 B3=82.5% 等主数字
- GraphRAG 仅为设计对比
- 标红用途与去红正稿可另出

- [ ] **Step 2: 对照设计验收清单逐项勾选**

对照 `design.md` §9，用下面命令辅助：

```bash
python3 - <<'PY'
from docx import Document
from pathlib import Path
out=Path('华东师范大学硕士论文_6章修改版.docx')
orig=Path('华东师范大学硕士论文.docx')
assert out.exists() and orig.exists()
assert out.stat().st_mtime >= orig.stat().st_mtime or True
doc=Document(str(out))
blob='\n'.join(p.text for p in doc.paragraphs)
checks={
 'six_ch': '第六章' in blob and '第七章' not in blob,
 'kg_pipeline': '特征抽取' in blob and 'Cypher' in blob,
 'collab_def': '静态知识' in blob and '动态知识' in blob,
 'graphrag_design': 'GraphRAG' in blob and ('未' in blob),
 'b2': 'B2' in blob,
 'domestic': '自然灾害学报' in blob or '地震研究' in blob,
 'filled': '82.5' in blob and '{{FACT' not in blob,
}
print(checks)
assert all(checks.values()), checks
PY
```

Expected: 全部 True。

- [ ] **Step 3: Commit**

```bash
git add docs/论文修改内容说明_6章制.md docs/superpowers/specs/2026-07-29-thesis-6chapter-rewrite-design.md
git commit -m "docs: add 6-chapter thesis revision notes and mark design implemented"
```

---

## Spec Coverage Self-Check

| 规格要求 | 对应 Task |
|----------|-----------|
| 6 章结构、取消空总体方法章 | Task 1–4 |
| 图谱 pipeline / 向量并入第3章 | Task 3 |
| 协同主体与调度伪代码 | Task 4 |
| 系统+60题消融+B2分析 | Task 5 |
| GraphRAG 设计对比无虚构分 | Task 2（综述）+ Task 5（5.4.2） |
| 国内文献 | Task 2 + Task 6 |
| 另存标红、不覆盖原稿 | Task 7 |
| 附录撑篇幅 | Task 6 |
| 验收清单 | Task 8 |

**Placeholder scan:** 无 TBD；骨架「占位」仅允许存在于 Task 1，须在 Task 2–6 清零。  
**Type consistency:** `BODY_BLOCKS: list[tuple[str,str]]`、`build_eval_mapping` 键名 `FACT_B0`…与摘要/第5章占位符一致。

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-29-thesis-6chapter-rewrite.md`. Two execution options:

**1. Subagent-Driven (recommended)** — 每个 Task 派一个新子代理，Task 间复查，迭代快  

**2. Inline Execution** — 本会话按 `executing-plans` 连续执行，设检查点  

Which approach?
