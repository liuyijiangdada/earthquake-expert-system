# RAG + 知识图谱集成（v1）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在主应用 `app.py` 的 `generate_response` 中并行注入 Neo4j 图谱上下文与基于 `data/emergency_knowledge.json` 的向量检索片段，并支持 `KG_CONTEXT_ENABLED` / `RAG_ENABLED` 消融开关。

**Architecture:** 分块策略锁定为 spec **方案 A**（每 `topic` 一条 chunk）。启动时（或懒加载）用嵌入模型将 chunk 编码为向量，查询时对用户问题编码后做余弦相似度 Top-K。提示词固定顺序：`【知识图谱】` → `【参考资料】` → `【问题】` → 约束；冲突规则按 spec 第 5 节写入 user 内容。嵌入或索引失败时记录日志并降级为「无 RAG 结果」，不中断 API。

**Tech Stack:** Python 3、Flask（现有）、`numpy`（已有）、新增 **`sentence-transformers`**（封装 BGE 等中文句向量，便于本地 `local_files_only` 缓存）；不引入 Chroma/FAISS 服务（语料规模小，内存矩阵 + Top-K 足够）。测试使用 **`pytest`**。

**Spec 依据：** `@docs/superpowers/specs/2026-03-25-rag-kg-integration-design.md`

---

## File map（创建 / 修改）

| 路径 | 职责 |
|------|------|
| `rag/__init__.py` | 包初始化（可为空） |
| `rag/emergency_rag.py` | 从 JSON 生 chunk、（可选）嵌入器封装、`search(query)->List[dict]` |
| `config/config.py` | `KG_CONTEXT_ENABLED`、`RAG_ENABLED`、嵌入模型名、Top-K、单块最大长度、`API_DEBUG_RAG` 等 |
| `app.py` | 初始化 RAG（try/except）、在 `generate_response` 中拼装新提示词结构 |
| `requirements.txt` | 增加 `sentence-transformers`、`pytest` |
| `tests/test_emergency_chunks.py` | 分块与元数据单元测试（无 GPU/无大模型） |
| `tests/test_emergency_rag_search.py` | 使用固定随机向量的 stub 嵌入器测试 Top-K 行为 |
| `tests/test_emergency_rag_live.py` | 可选：真实嵌入模型集成测试（`RUN_LIVE_EMBED_TESTS=1`） |

**本阶段不修改：** `kg/neo4j_kg.py`（spec 冻结）。  
**范围外（注明即可）：** `llm/serve_earthquake_expert.py` 仅接受原始 `prompt`，v1 不强制改；若论文需双入口一致，可在本计划全部完成后再开小型 follow-up。

---

### Task 1: 分块纯函数与单元测试（无嵌入）

**Files:**
- Create: `tests/test_emergency_chunks.py`
- Create: `rag/__init__.py`
- Create: `rag/emergency_rag.py`（先只实现 `load_emergency_chunks(path) -> list[dict]`）

**约定：** 每个 topic 输出一条记录：`{"topic_id": str, "title": str, "source": str, "text": str}`，其中 `text` = `title` + `category` + 换行 + 各 `step.text` 按 `order` 拼接；**忽略** `meta` 与 `topic_relations`（与 spec 一致）。

- [ ] **Step 1: 写失败测试**

在 `tests/test_emergency_chunks.py` 中：

```python
import os
import json
import pytest

# 项目根目录：tests/conftest.py 里可加 pytest 的 rootdir fixture，或用手动路径：
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def test_load_emergency_chunks_count_and_ids():
    from rag.emergency_rag import load_emergency_chunks
    path = os.path.join(ROOT, "data", "emergency_knowledge.json")
    chunks = load_emergency_chunks(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(chunks) == len(data["topics"])
    ids = {c["topic_id"] for c in chunks}
    assert "em_indoor" in ids and "em_kit" in ids
    assert all(c["text"] and c["title"] for c in chunks)
```

- [ ] **Step 2: 运行测试，确认失败**

在项目根目录执行：  
`pip install pytest -q && pytest tests/test_emergency_chunks.py -v`  
（根目录可用 `cd "$(git rev-parse --show-toplevel)"` 定位。）  
Expected: **FAIL**（`load_emergency_chunks` 不存在或 ImportError）

- [ ] **Step 3: 最小实现**

在 `rag/emergency_rag.py` 中实现 `load_emergency_chunks`，仅解析 JSON，不加载 torch/sentence_transformers。

- [ ] **Step 4: 运行测试通过**

Run: `pytest tests/test_emergency_chunks.py -v`  
Expected: **PASS**

- [ ] **Step 5: Commit**

```bash
git add rag/ tests/test_emergency_chunks.py
git commit -m "test: emergency knowledge chunk loader"
```

---

### Task 2: 检索逻辑 + Stub 嵌入器测试

**Files:**
- Modify: `rag/emergency_rag.py`（新增 `EmergencyRAG` 类或函数：`__init__(chunks, embed_fn)`、`search(query, top_k) -> list[dict]`）
- Create: `tests/test_emergency_rag_search.py`

**检索定义：** `score` = 查询向量与 chunk 向量的余弦相似度（`numpy` 实现即可）；返回 `{"topic_id", "title", "source", "text", "score"}` 列表按 score 降序截断 Top-K。

- [ ] **Step 1: 写失败测试**

使用 3 个手工 chunk 与 stub：`embed(text)` 返回固定低维向量（例如 hash 到确定性向量），保证「与查询最相近」的 topic 可预测。

```python
import numpy as np
import pytest

def test_top_k_ordering_with_stub_embedder():
    from rag.emergency_rag import EmergencyRAG

    chunks = [
        {"topic_id": "a", "title": "t1", "source": "s", "text": "alpha"},
        {"topic_id": "b", "title": "t2", "source": "s", "text": "beta gamma"},
    ]
    # stub: 维数 4，query 与 chunk b 更相似
    def embed(text: str) -> np.ndarray:
        if "beta" in text:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return np.array([0.0, 1.0, 0.0, 0.0])

    rag = EmergencyRAG(chunks=chunks, embed=embed, embed_query=lambda q: np.array([1.0, 0.0, 0.0, 0.0]))
    hits = rag.search("anything", top_k=2)
    assert hits[0]["topic_id"] == "b"
```

（实现时可合并 `embed` / `embed_query` 为同一函数，测试相应调整。）

- [ ] **Step 2: pytest 确认失败**

Run: `pytest tests/test_emergency_rag_search.py -v`  
Expected: **FAIL**

- [ ] **Step 3: 实现 `EmergencyRAG.search`**

- [ ] **Step 4: pytest 通过**

- [ ] **Step 5: Commit**

```bash
git add rag/emergency_rag.py tests/test_emergency_rag_search.py
git commit -m "feat: emergency RAG cosine Top-K with injectable embedder"
```

---

### Task 3: SentenceTransformer 真实嵌入与优雅降级

**Files:**
- Modify: `rag/emergency_rag.py`（工厂函数 `build_emergency_rag_from_config(config) -> Optional[EmergencyRAG]`）
- Modify: `requirements.txt`（`sentence-transformers`）

**行为：**

- 从 `config.EMERGENCY_KNOWLEDGE_FILE`（或绝对路径）加载 chunks；预计算所有 chunk 向量并缓存于内存。
- `from sentence_transformers import SentenceTransformer`，模型名 `config.RAG_EMBEDDING_MODEL`（默认 `BAAI/bge-small-zh-v1.5`）。若项目需与 `app.py` 一致完全离线，在加载时使用与主模型相同的缓存目录策略；**首次**需在有网环境 `huggingface-cli download` 或等价方式拉取模型到缓存。
- `try/except`：任何失败返回 `None`，由调用方记录 `logging.exception`。

- [ ] **Step 1: 添加可选集成测试（标记跳过若无模型）**

`tests/test_emergency_rag_live.py`：

```python
import os
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_EMBED_TESTS") != "1",
    reason="set RUN_LIVE_EMBED_TESTS=1 to run embedding integration test",
)

def test_build_rag_live():
    from config.config import Config
    from rag.emergency_rag import build_emergency_rag_from_config
    rag = build_emergency_rag_from_config(Config())
    assert rag is not None
    hits = rag.search("室内地震怎么躲", top_k=3)
    assert any("em_indoor" == h["topic_id"] for h in hits)
```

- [ ] **Step 2: 实现 `build_emergency_rag_from_config`**

- [ ] **Step 3: 本地有缓存时** Run: `RUN_LIVE_EMBED_TESTS=1 pytest tests/test_emergency_rag_live.py -v`  
Expected: **PASS**；无模型时跳过不阻断 CI。

- [ ] **Step 4: Commit**

```bash
git add rag/emergency_rag.py requirements.txt tests/test_emergency_rag_live.py
git commit -m "feat: optional SentenceTransformer backend for emergency RAG"
```

---

### Task 4: Config 开关与路径

**Files:**
- Modify: `config/config.py`

新增（名称可与下述等价，但需在 `app.py` 中一致使用）：

```python
# RAG / KG 消融与检索参数
KG_CONTEXT_ENABLED = True
RAG_ENABLED = True
RAG_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
RAG_TOP_K = 5
RAG_MAX_CHUNK_CHARS = 800
API_DEBUG_RAG = False  # True 时在 JSON 响应附带 rag_hits 元信息
```

路径：复用已有 `EMERGENCY_KNOWLEDGE_FILE`。

- [ ] **Step 1: 修改 `config/config.py`**
- [ ] **Step 2: Commit**

```bash
git add config/config.py
git commit -m "config: KG/RAG toggles and embedding settings"
```

---

### Task 5: `app.py` 提示词拼装与 RAG 接入

**Files:**
- Modify: `app.py`

**要点：**

1. 模块级：`emergency_rag = None`，在模型加载成功后（或并行）调用 `build_emergency_rag_from_config(config)`；失败则 `emergency_rag = None` 并打日志。
2. `generate_response` 内：
   - 若 `config.KG_CONTEXT_ENABLED`：**执行现有图谱逻辑**填充 `kg_context`；否则不执行图谱查询分支，`kg_body = "（本路径已关闭）"`。
   - 若启用 KG 且 `kg_context` 非空：`kg_body = kg_context`（可将现有「【知识图谱信息】」等子标题保留在字符串内，整体置于 `【知识图谱】` 下）；若启用但为空：`kg_body = "（无）"`。
   - 若 `config.RAG_ENABLED` 且 `emergency_rag` 非空：对 `input_text` 调用 `search`，按 `RAG_MAX_CHUNK_CHARS` 截断每条 `text`，格式化为编号列表，作为 `【参考资料】`；若无 hits：`（无相关条目）`。若 RAG 关闭：`（本路径已关闭）`；若 RAG 开但后端 `None`：`（无相关条目）`（并已在日志中说明失败原因）。
3. **User 消息正文模板**（与 spec 对齐，可微调措辞但保留五段顺序与占位）：

```text
你是一个地震知识专家。请结合【知识图谱】与【参考资料】回答问题。

【知识图谱】
{kg_section}

【参考资料】
{rag_section}

【问题】
{input_text}

规则：数值、时间、震级、地点等可验证事实以知识图谱为准；参考资料仅作步骤与表述补充。若两者均未提供有效条目，可基于常识回答，并简要说明未命中本地知识库。
回答要求：（保留现有 1～5 条要点，并将第 2 条改为「优先采用知识图谱中的可验证事实，并合理利用参考资料」）
```

4. `tokenizer(..., max_length=1024)` **保持不变**；若超长，优先缩短 `【参考资料】` 条目数或每条条目字符（在代码里对 RAG 文本做 `[:RAG_MAX_CHUNK_CHARS]` 已缓解）。

5. **可选：** `query_type == "llm"` 时若 `API_DEBUG_RAG`，在 `jsonify` 中增加 `debug: {"rag_topic_ids": [...], "kg_enabled": ..., "rag_enabled": ...}`。

6. **离线注意：** `app.py` 已设 `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` 时，若本机无嵌入模型缓存，`SentenceTransformer` 会失败并走 RAG 降级；需先在有网环境将 `RAG_EMBEDDING_MODEL` 拉入 HF 缓存（与 Task 3 说明一致）。

- [ ] **Step 1: 手写或用脚本确认** 启动应用后，对同一问题切换 `.env` 或临时改 `Config` 三种组合，检查响应与日志（集成测试可选，不强制 pytest Flask client，以免加载大模型）。
- [ ] **Step 2: Commit**

```bash
git add app.py
git commit -m "feat: parallel KG + emergency RAG prompt in generate_response"
```

---

### Task 6: 依赖与简短运行说明

**Files:**
- Modify: `requirements.txt`（若 Task 3 已加 sentence-transformers / pytest 则检查重复）
- 可选：在现有 `MODEL_SETUP.md` 或 `PROJECT_IMPLEMENTATION.md` **追加一小节**（2～4 句）：RAG 依赖、默认模型名、离线缓存、`RAG_ENABLED=false` 关闭方式 — **仅当仓库已有文档惯例**；否则可省略，避免用户未要求的文档膨胀。

- [ ] **Step 1:** `pip install -r requirements.txt` 在干净 venv 验证
- [ ] **Step 2: Commit**（若有文档）

```bash
git add requirements.txt MODEL_SETUP.md
git commit -m "docs: note emergency RAG embedding offline setup"
```

---

## 验收核对（对照 spec §7）

- [ ] `tests/test_emergency_chunks.py` 与 `tests/test_emergency_rag_search.py` 在默认 `pytest` 下通过。
- [ ] 设置 `KG_CONTEXT_ENABLED=False` 时，提示中 `【知识图谱】` 为 `（本路径已关闭）`。
- [ ] 设置 `RAG_ENABLED=False` 时，`【参考资料】` 为 `（本路径已关闭）`。
- [ ] 嵌入加载失败时服务仍可启动，RAG 段落为 `（无相关条目）`，日志有异常栈或错误信息。

---

## Plan Review Loop

本计划写完并已保存后，应由独立审阅者对照 spec 做完整性检查（见 `plan-document-reviewer-prompt.md`）。

---

## Execution Handoff

计划保存路径：`docs/superpowers/plans/2026-03-25-rag-kg-integration.md`。

**执行方式二选一：**

1. **Subagent-Driven（推荐）** — 每任务派生子代理，任务间 review；需 **@superpowers:subagent-driven-development**。  
2. **Inline Execution** — 本会话内按勾选逐步执行；需 **@superpowers:executing-plans**。

请选择 **1** 或 **2**（或自行直接实现并勾选）。
