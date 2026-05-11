# 第6章 协同推理与系统实现

**说明：** 与项目根目录 `app.py`、`config/config.py` 及 `static/index.html` 一致。可经 `scripts/export_md_to_docx.py` 导出 Word。

---

## 6.1 图谱上下文与检索片段的融合策略

### 6.1.1 融合顺序、分隔符与角色说明

大模型单次推理的 user 侧内容由固定**自然语言模板**拼接而成，整体顺序为：

1. **任务与角色引导**：声明为地震知识专家，要求结合【知识图谱】与【参考资料】作答。  
2. **【知识图谱】块**：承接第 4 章输出，为纯文本段落；无命中时为「（无）」，关闭图谱上下文时为「（本路径已关闭）」。  
3. **【参考资料】块**：承接第 5 章 `_build_rag_section` 输出；无 RAG 或未检索到条目时为相应占位字符串。  
4. **规则段**：明确可验证事实以知识图谱为准、参考资料作补充；二者均无效时可常识作答并说明未命中本地库。  
5. **回答要求**：直接作答、优先图谱事实、简洁、禁用特定强调符号等条目化约束。  
6. **【问题】块**：用户原始问句置于**全段末尾**。

该顺序使模型先看到外部知识，再看到行为约束，最后聚焦具体问句，符合指令跟随的常见实践，并与截断策略配套（见 6.1.3）。

System 角色单独设为简短「地震专家」说明，与 user 长提示分离，便于与微调数据中的**多轮标记格式**对齐。

### 6.1.2 与微调对话模板的对齐（system/user 标记）

推理时将 `messages` 列表线性化为单一字符串，再交给分词器。映射规则为：

- `system` → `<|system|>{content}</s>`  
- `user` → `<|user|>{content}</s>`  
- `assistant` → `<|assistant|>{content}</s>`（本路径仅用于拼接历史时扩展，当前为一轮 user）  

在完整 user 内容之后**追加** `<|assistant|>` 作为生成起点，与 Qwen 系等聊天微调格式一致。若训练与推理标记不一致，会导致续写分布偏移甚至空输出，因此 LoRA 训练脚本与 `app.py` 必须保持同一套分隔符与结束符约定。

解码后对生成片段剔除可能残留的 `</s>`、`<|system|>` 等字面串，避免返回给用户。

### 6.1.3 长提示下的左侧截断与问题保尾策略

应急要点与 Top-K 片段可能使 user 提示很长。分词器设置 **`truncation_side = "left"`**：在超过 `max_length`（配置项 `LLM_INPUT_MAX_TOKENS`，默认 4096）时从**左侧**丢弃较早 token，**保留字符串物理尾部**。

由于【问题】与紧随其后的 `<|assistant|>` 位于整段模板末端，左侧截断可更大限度保留**真实用户问句**与**生成起始标记**；若采用默认右侧截断，易砍掉末尾问句或 assistant 标记，造成解码异常或前端「未收到有效回答」类现象。

---

## 6.2 本地大模型推理与解码设置

### 6.2.1 基座模型与 LoRA 加载流程

应用启动时在导入 `transformers` / `huggingface_hub` **之前**设置环境变量 `HF_HUB_OFFLINE`、`TRANSFORMERS_OFFLINE`，降低初始化阶段误连外网概率。项目根目录 `_APP_ROOT` 用于解析 LoRA 相对路径，保证工作目录变化时仍能定位 `adapter_config.json`。

加载顺序为：（1）`AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True, …)` 读取**基座**（配置中如 `Qwen/Qwen1.5-1.8B`，依赖本机缓存路径）；（2）`AutoTokenizer.from_pretrained`：若 LoRA 目录下存在 `tokenizer.json`，**优先从 LoRA 目录**加载词表，避免再对 `MODEL_NAME` 走 Hub；（3）`PeftModel.from_pretrained(base_model, MODEL_DIR, local_files_only=True)` 挂载适配器；（4）`model.to(device)` 并 `eval()`。

缺少 `adapter_config.json` 时直接报错退出，防止运行期静默回退到未微调基座而难以察觉。

### 6.2.2 解码超参与防复读设置

`model.generate` 的参数与 `config.config` 对齐，主要包括：`max_new_tokens`（`LLM_MAX_NEW_TOKENS`）、`temperature`（`LLM_TEMPERATURE`）、`do_sample`（`LLM_DO_SAMPLE`）、`top_p`（`LLM_TOP_P`）、`repetition_penalty`（`LLM_REPETITION_PENALTY`）、`no_repeat_ngram_size`（`LLM_NO_REPEAT_NGRAM_SIZE`）。其中重复惩罚与 n-gram 禁止重复用于减轻应急条文场景下**同句循环**问题。

推理使用 `torch.no_grad()`，仅取输入长度之后的 token id 序列解码为字符串，避免将提示再次输出。

异常时返回固定错误提示字符串，前端可统一展示重试建议。

### 6.2.3 设备选择（CPU / CUDA / MPS）与精度

设备优先级为：**MPS**（Apple GPU）可用则选用，否则 **CUDA**，最后 **CPU**。数据类型上，GPU/MPS 使用半精度 `float16` 以节约显存与带宽；CPU 使用 `float32` 以保证数值稳定。该策略在消费级笔记本与服务器上均可自动适配。

---

## 6.3 系统架构与关键模块实现

### 6.3.1 Web 服务与 API 设计（查询、图谱、数据更新）

Flask 应用在模块导入阶段完成 Neo4j 初始化（`Neo4jKG.run()`）、可选 RAG 索引构建及模型加载，再注册路由：

- **`GET /`**：返回 `static/index.html`。  
- **`POST /api/query`**：JSON 体含 `query_type` 与 `params`。  
  - `kg`：`params.type` 为 `all`、`by_region`、`by_magnitude`、`by_depth` 之一，返回 `{ results: [...] }`。  
  - `llm`：`params.input` 为用户问题，返回 `{ response: "..." }`；若开启 `API_DEBUG_RAG`，可附加 `debug`（含图谱/RAG 开关及 `rag_topic_ids` 等）。  
- **`POST /api/update-data`**：调用图谱增量更新，返回 `status`、`updated`、`message` 或错误信息。

缺参或类型非法时返回 400 及 `error` 字段，便于前端分支处理。

### 6.3.2 前端交互与错误提示

前端通过 `fetch` 调用上述接口。对话路径在响应非 ok 或存在 `error` 时展示服务不可用类提示；若 `response` 非字符串或去空白后为空，则展示**「未收到有效回答，请换种问法或稍后重试。」**，与后端截断、解码失败等边界行为衔接。地震列表与筛选走 `kg` 分支，与对话解耦。数据更新按钮调用 `update-data` 并根据返回展示成功条数或失败原因。

### 6.3.3 配置项、离线约束与可复现部署

**配置项**：`config/config.py` 集中管理 Neo4j 连接、数据文件路径、`MODEL_NAME`、`FINETUNED_MODEL_PATH`、RAG 嵌入模型与 Top-K、LLM 输入长度与解码参数、服务监听 `DEPLOY_HOST`/`DEPLOY_PORT`、日志级别等。消融实验通过 `KG_CONTEXT_ENABLED`、`RAG_ENABLED` 等布尔项切换，无需改业务代码。

**离线约束**：基座、LoRA、SentenceTransformer 权重均需预先置于本机缓存；`local_files_only=True` 贯穿模型与适配器加载。启动命令建议在项目根目录执行 `python app.py`，以保证相对路径与静态资源解析一致。

**可复现性**：固定配置快照、Neo4j 数据导出、随机种子（若对采样解码做对比实验）与依赖版本记录，可作为论文实验环境说明的补充（第 7 章）。

---

## 本章小结

本章说明了**图谱与 RAG 文本在提示中的拼接顺序**、**system/user/assistant 模板与微调对齐**、**左侧截断与问题保尾**；概括了**基座 + LoRA 离线加载**、**generate 超参**及**设备与精度选择**；并整理了 **Flask API**、**前端错误与空回答处理**以及**配置与离线部署要点**。第 7 章在此基础上展开实验与结果分析。
