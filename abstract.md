# 基于知识图谱与向量检索协同的地震应急问答方法研究

## 中文摘要

地震应急场景下，公众与基层人员往往需要快速、可核对的事实性回答，而单纯依赖大语言模型易产生幻觉，且难以稳定利用结构化事件信息与规范化应急指引。针对地震应急问答任务，研究一种**知识图谱与向量检索协同**的方法：在结构化侧，基于 Neo4j 构建地震事件、区域及应急主题、处置步骤等实体关系，并按规则从用户问题中触发子图查询，生成图谱上下文；在非结构化侧，对应急知识文本进行分块与句向量编码，采用余弦相似度进行 Top-K 检索，生成参考资料片段。将图谱上下文与检索片段与领域提示模板拼接后，输入经 LoRA 微调的本地大语言模型完成生成，从而在统一推理流程中兼顾**可解释的结构化约束**与**灵活的文本知识覆盖**。在此基础上，给出系统实现要点，包括离线部署约束下的模型与嵌入加载、长提示左侧截断策略及 Web 交互方式。实验与案例分析表明，相较于仅使用单一知识形态的基线，协同方法在事实一致性与回答完整性方面具有优势。最后对方法局限与改进方向进行讨论。

**关键词：** 知识图谱；向量检索；地震应急；问答系统；大语言模型

---

## Abstract

Earthquake disasters are characterized by suddenness and destructiveness. Information needs differ markedly across the three phases of pre-earthquake preparedness, during-earthquake response, and post-earthquake recovery. Conventional question-answering approaches struggle to simultaneously satisfy the systematic nature of static knowledge and the timeliness of dynamic intelligence. Building upon a collaborative framework of knowledge graphs and vector retrieval, this study further develops a dynamic–static knowledge cooperative earthquake emergency QA system that integrates a Neo4j knowledge graph, Milvus/in-memory vector retrieval, unified CEIC+USGS dynamic earthquake information (core/earthquake_feed.py), three-phase question classification, and an uncertainty-aware scheduler, with local inference based on Qwen1.5-1.8B+LoRA. The main work includes: (1) designing a knowledge graph schema for earthquake events, regions, emergency topics, and procedural steps, and implementing query-driven subgraph retrieval; (2) applying BAAI/bge-small-zh-v1.5 to chunk emergency knowledge at the topic level, embedding the chunks, and performing Top-K semantic retrieval; (3) proposing pre-, during-, and post-earthquake phase classification and a three-signal scheduling strategy based on static_confidence, dynamic_availability, and urgency, routing static graph context, vector retrieval, and dynamic data sources by phase; (4) implementing QueryContextBuilder for unified context assembly, together with response_guard quality fallback, multimodal resource matching, and Layer-3 output enrichment such as epicenter maps, shelter information, and aftershock charts; (5) deploying Web and mobile clients based on Flask+Vue 3 and an Android app, and providing a phased evaluation query-set API. Four ablation settings—LLM-only, graph-only, RAG-only, and full collaboration—are constructed via configuration switches; metrics are stratified by pre-, during-, and post-earthquake phases to verify improvements in factual consistency, key-point completeness, and format compliance brought by the collaborative mechanism and phase-aware scheduling (quantitative results are presented in Chapter 7).

**Keywords:** knowledge graph; vector retrieval; earthquake emergency; question answering; large language model

---

## 英文题目（可选）

*Research on Earthquake Emergency Question Answering via Knowledge Graph and Vector Retrieval Collaboration*
