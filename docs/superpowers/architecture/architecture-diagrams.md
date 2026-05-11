# 系统架构图（Mermaid 源）

在 VS Code 安装 **Mermaid** 插件预览，或复制代码块到 [Mermaid Live Editor](https://mermaid.live) 导出 **PNG/SVG** 插入 Word。

---

## 图 2-1-1 系统总体逻辑架构

```mermaid
flowchart TB
  subgraph client["展示层"]
    B["浏览器\nstatic/index.html"]
  end

  subgraph app["应用服务层 Flask"]
    F["app.py\n/api/query · /api/update-data · /"]
  end

  subgraph data["数据与模型层"]
    N[("Neo4j\n知识图谱")]
    R["RAG 内存索引\nemergency_rag.py"]
    L["本地大模型\ntransformers + peft\n基座 + LoRA"]
  end

  subgraph cfg["配置"]
    C["config/config.py"]
  end

  B -->|HTTP JSON| F
  F --> N
  F --> R
  F --> L
  C -.-> F
  C -.-> N
  C -.-> R
  C -.-> L
```

---

## 图 2-1-2 问答请求数据流（`query_type: llm`）

```mermaid
flowchart TB
  U[用户输入] --> API["POST /api/query · llm"]
  API --> GR[generate_response]
  GR --> KG{KG_CONTEXT_ENABLED?}
  KG -->|是| NQ["Neo4j 子图查询\n省名 / 震级 / 应急语境"]
  KG -->|否| SK1[图谱段关闭占位]
  NQ --> KC[图谱上下文文本]
  SK1 --> KC
  GR --> RAG{RAG 已加载?}
  RAG -->|是| VQ["句向量 + 余弦 Top-K"]
  RAG -->|否| SK2[无相关条目]
  VQ --> RS[参考资料文本]
  SK2 --> RS
  KC --> PROMPT["拼装 user 提示\n【知识图谱】【参考资料】【问题】"]
  RS --> PROMPT
  PROMPT --> CHAT["聊天模板\n&lt;|system|&gt; … &lt;|user|&gt; … &lt;|assistant|&gt;"]
  CHAT --> TOK["tokenizer · left-truncate"]
  TOK --> GEN[model.generate]
  GEN --> OUT["JSON response"]
```

> 说明：Mermaid 中尖括号需转义为 `&lt;` `&gt;`，导出图时显示为 `<|system|>` 等。

---

## 图 2-1-2b 简化纵向数据流（适合窄栏排版）

```mermaid
flowchart TB
  U[用户] --> W[Web 前端]
  W -->|llm| A[Flask]
  A --> G[Neo4j 查询]
  A --> V[向量检索]
  G --> P[提示融合]
  V --> P
  P --> M[LoRA 大模型生成]
  M --> W
```

---

## 图 2-1-3 技术栈与模块映射

```mermaid
flowchart LR
  subgraph mod["功能模块"]
    M1[对话问答]
    M2[地震列表/筛选]
    M3[数据刷新]
  end

  subgraph impl["实现"]
    I1["generate_response\n+ LLM"]
    I2["query_type: kg\nNeo4j Cypher"]
    I3["update_from_realtime_data\nUSGS 等"]
  end

  M1 --> I1
  M2 --> I2
  M3 --> I3
```

| 模块 | 主要代码 / 技术 | 职责 |
|------|-----------------|------|
| 前端 | `static/index.html` | 对话、地震列表、筛选、触发更新 |
| 应用网关 | `app.py`（Flask） | 路由、组装 KG/RAG 上下文、调用 LLM |
| 知识图谱 | `kg/neo4j_kg.py` + Neo4j | 地震事件、区域、应急主题与步骤 |
| 向量检索 | `rag/emergency_rag.py` + SentenceTransformer | `emergency_knowledge.json` 分块嵌入与 Top-K |
| 大模型 | Hugging Face `transformers` + `peft` | 基座 `MODEL_NAME` + 本地 LoRA 目录 |
| 配置 | `config/config.py` | Neo4j、路径、RAG/LLM 开关与超参 |

---

## 图：服务启动顺序（可选，第 6 章部署）

```mermaid
flowchart TD
  S1[加载 Config] --> S2[初始化 Neo4j 与导入]
  S2 --> S3[加载 tokenizer / 基座 / LoRA]
  S3 --> S4[构建 RAG 索引可选]
  S4 --> S5[启动 Flask 监听端口]
```
