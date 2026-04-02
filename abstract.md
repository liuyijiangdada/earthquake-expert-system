# 基于知识图谱与向量检索协同的地震应急问答方法研究

## 中文摘要

地震应急场景下，公众与基层人员往往需要快速、可核对的事实性回答，而单纯依赖大语言模型易产生幻觉，且难以稳定利用结构化事件信息与规范化应急指引。针对地震应急问答任务，研究一种**知识图谱与向量检索协同**的方法：在结构化侧，基于 Neo4j 构建地震事件、区域及应急主题、处置步骤等实体关系，并按规则从用户问题中触发子图查询，生成图谱上下文；在非结构化侧，对应急知识文本进行分块与句向量编码，采用余弦相似度进行 Top-K 检索，生成参考资料片段。将图谱上下文与检索片段与领域提示模板拼接后，输入经 LoRA 微调的本地大语言模型完成生成，从而在统一推理流程中兼顾**可解释的结构化约束**与**灵活的文本知识覆盖**。在此基础上，给出系统实现要点，包括离线部署约束下的模型与嵌入加载、长提示左侧截断策略及 Web 交互方式。实验与案例分析表明，相较于仅使用单一知识形态的基线，协同方法在事实一致性与回答完整性方面具有优势。最后对方法局限与改进方向进行讨论。

**关键词：** 知识图谱；向量检索；地震应急；问答系统；大语言模型

---

## Abstract

Earthquake emergency scenarios demand timely and verifiable answers. Relying solely on large language models (LLMs) may cause hallucinations and weak utilization of structured event records and standardized guidance. This thesis studies a collaborative approach that combines a knowledge graph with vector retrieval for earthquake emergency question answering. On the structured side, a Neo4j graph models earthquakes, regions, emergency topics, and procedural steps; rule-based triggers extract subgraphs from user queries to form graph context. On the unstructured side, emergency knowledge is chunked and encoded by sentence embeddings; top-K passages are retrieved via cosine similarity. Graph context and retrieved passages are fused with a domain prompt and fed into a locally deployed LLM fine-tuned with LoRA. The pipeline integrates structured constraints with flexible textual coverage under offline deployment, including left-side truncation for long prompts. Experiments and case studies show improvements over single-modality baselines in factual consistency and completeness. Limitations and future work are discussed.

**Keywords:** knowledge graph; vector retrieval; earthquake emergency; question answering; large language model

---

## 英文题目（可选）

*Research on Earthquake Emergency Question Answering via Knowledge Graph and Vector Retrieval Collaboration*
