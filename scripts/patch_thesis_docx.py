#!/usr/bin/env python3
"""根据当前代码实现，润色《华东师范大学硕士论文.docx》正文模块。"""
from __future__ import annotations

import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parent.parent
THESIS = ROOT / "华东师范大学硕士论文.docx"


def set_para_text(para, text: str) -> None:
    for run in para.runs:
        run.text = ""
    if para.runs:
        para.runs[0].text = text
    else:
        para.add_run(text)


def insert_after(para, text: str) -> Paragraph:
    new_p = OxmlElement("w:p")
    para._p.addnext(new_p)
    new_para = Paragraph(new_p, para._parent)
    if text:
        new_para.add_run(text)
    return new_para


def insert_block_after(para, lines: list[str]) -> Paragraph:
    last = para
    for line in lines:
        last = insert_after(last, line)
    return last


# 按段落索引替换（索引来自 2026-06 版 docx 结构）
REPLACEMENTS: dict[int, str] = {
    49: (
        "地震灾害具有突发性和破坏性，震前防御、震中应急与震后恢复三阶段的信息需求差异显著，"
        "传统问答方法难以同时满足静态知识的系统性与动态情报的实时性。本研究在知识图谱与向量检索协同框架之上，"
        "进一步构建动态—静态知识协同的地震应急问答系统：融合 Neo4j 知识图谱、Milvus/内存向量检索、"
        "USGS 动态震情、三阶段问题分类与不确定性感知调度器，并采用 Qwen1.5-1.8B+LoRA 本地推理。"
        "具体工作包括：（1）设计地震事件、区域、应急主题与处置步骤的知识图谱模式，实现问题驱动的子图查询；"
        "（2）采用 BAAI/bge-small-zh-v1.5 对应急知识 topic 级分块并向量化，完成 Top-K 语义检索；"
        "（3）提出震前/震中/震后阶段分类与 static_confidence、dynamic_availability、urgency "
        "三信号调度策略，按阶段路由静态图谱、向量检索与动态数据源；"
        "（4）实现 QueryContextBuilder 统一组装上下文，配合多模态资源匹配、震中地图/避难所/余震图等第三层输出增强；"
        "（5）基于 Flask+Vue 3 与 Android 客户端完成 Web/移动端部署，提供分阶段评测问集接口。"
        "通过配置开关构造仅 LLM、仅图谱、仅 RAG 与全协同四组消融实验，并按震前/震中/震后分阶段统计指标，"
        "验证协同机制与阶段调度对事实一致性、要点完整性与格式合规性的提升（定量结果见第 7 章）。"
    ),
    51: "关键词：知识图谱；向量检索；动态—静态知识协同；地震应急；三阶段调度；检索增强生成；LoRA微调",
    54: (
        "Earthquake disasters are characterized by suddenness and destructiveness. Information needs differ "
        "markedly across pre-earthquake preparedness, during-earthquake response, and post-earthquake recovery. "
        "This study extends knowledge-graph and vector-retrieval collaboration with a dynamic–static knowledge "
        "cooperative QA system integrating Neo4j, Milvus/in-memory RAG, USGS dynamic feeds, three-phase "
        "classification, and an uncertainty-aware scheduler, with local Qwen1.5-1.8B+LoRA inference. "
        "Contributions include: (1) KG schema and triggered subgraph queries; (2) topic-level chunking and "
        "Top-K retrieval with bge-small-zh-v1.5; (3) pre/during/post phase routing via static_confidence, "
        "dynamic_availability, and urgency signals; (4) QueryContextBuilder and Layer-3 enrichment (maps, shelters, "
        "aftershock charts); (5) Flask+Vue 3 and Android deployment with a phased evaluation API. "
        "Four ablation baselines and phase-stratified metrics validate the collaboration and scheduling mechanism."
    ),
    56: (
        "Keywords: Knowledge Graph; Vector Retrieval; Dynamic–Static Knowledge Collaboration; "
        "Earthquake Emergency; Three-Phase Scheduling; RAG; LoRA Fine-tuning"
    ),
    80: "围绕地震应急自然语言问答任务，研究工作主要包括：",
    81: (
        "（1）知识图谱构建与问题驱动的子图查询。设计地震事件、区域、应急主题与处置步骤模式，"
        "基于 Neo4j 完成数据导入；依据省区名称、震级阈值及应急触发词查询子图，并序列化为图谱上下文。"
    ),
    82: (
        "（2）应急知识向量索引与 Top-K 检索。对 emergency_knowledge.json 进行 topic 级分块，"
        "使用 BAAI/bge-small-zh-v1.5 编码，支持 Milvus 持久化或内存矩阵（RAG_USE_MEMORY_RAG）两种模式。"
    ),
    83: (
        "（3）动态—静态知识协同与不确定性感知调度。实现 PhaseClassifier 判定震前/震中/震后，"
        "DynamicRetriever 接入 USGS 实时目录（中国区过滤与 TTL 缓存），Scheduler 综合三信号决定 "
        "KG/RAG/动态源的启用优先级，并由 services/context_builder.py 统一组装提示。"
    ),
    84: (
        "（4）协同融合、多模态与系统实现。文本路径采用 Qwen1.5-1.8B+LoRA 与左侧截断保尾策略；"
        "图片路径支持 Qwen2-VL 懒加载；第三层增强输出避难所、震中地图、余震时序图与政策 RSS；"
        "完成 Flask Web、Vue 3 SPA 与 Android 客户端，并提供 GET /api/eval-set 分阶段评测问集。"
    ),
    85: "1.3.2  创新点",
    86: (
        "（1）面向地震应急三阶段的动态—静态知识协同流程。在图谱—向量双通路基础上，引入阶段分类、"
        "动态震情探测与不确定性感知调度，使系统能按震前/震中/震后自动调整知识源权重与回答语气。"
    ),
    87: (
        "（2）可解释的三信号调度与上下文统一组装。通过 static_confidence、dynamic_availability、urgency "
        "量化路由决策，QueryContextBuilder 避免重复检索并输出 meta 元信息，便于实验分析与论文案例复现。"
    ),
    88: (
        "（3）可消融、可分阶段的实验框架。配置项 KG_CONTEXT_ENABLED、RAG_ENABLED、"
        "PHASE_CLASSIFIER_ENABLED、DYNAMIC_RETRIEVAL_ENABLED 等支持四基线对比；"
        "data/eval/phase_questions.json 提供震前/震中/震后各 20 题，支撑分阶段人工评测。"
    ),
    91: (
        "第2章介绍相关技术与系统架构。第3章给出任务定义、三阶段协同框架与评价思路。"
        "第4章阐述知识图谱构建与查询。第5章介绍向量检索与上下文增强。"
        "第6章说明协同推理、调度器、多模态与系统实现。第7章给出实验环境、分阶段评测流程与结果分析。"
        "第8章总结全文并展望未来工作。"
    ),
    96: (
        "系统自上而下分为展示层、应用服务层、协同调度层、知识数据层与模型层，配置由 config/config.py "
        "与 .env（如高德 Web 服务 Key）统一管理。"
    ),
    97: (
        "展示层包括 Vue 3 SPA（frontend/，构建产物 static/spa/）与 Android 客户端（disaster_app/android/）。"
        "Web 端提供三阶段快捷提问胶囊、阶段标签、多模态资源卡片与 meta 元信息展示；"
        "Android 端通过 Retrofit 调用同一 Flask API（模拟器默认 http://10.0.2.2:8000/）。"
    ),
    98: (
        "应用服务层由 app.py 实现，主要接口包括：POST /api/query（llm/kg）、POST /api/multimodal-query（图文）、"
        "POST /api/update-data、GET /api/eval-set（分阶段评测问集）、POST /api/phase-classify、"
        "GET /api/amap/static-map（震中静态图代理，Key 不暴露前端）及 /generated-media/ 动态生成资源。"
        "文本问答经 QueryContextBuilder.prepare() 组装上下文后调用 generate_response。"
    ),
    99: (
        "协同调度层（core/）含 PhaseClassifier、Scheduler、DynamicRetriever、MultimodalOutput，"
        "以及 knowledge_signals 计算 static_confidence 与 dynamic_availability。"
        "知识数据层含 Neo4j（kg/neo4j_kg.py）、RAG（rag/emergency_rag.py，Milvus 或内存）、"
        "shelters.json、multimodal_resources.json 与 USGS 动态源。"
        "模型层为常驻 Qwen1.5-1.8B+LoRA 与懒加载 Qwen2-VL-2B-Instruct。"
    ),
    100: (
        "配置模块贯穿各层，支持 KG/RAG/阶段分类/动态检索/第三层增强等开关，保证部署与消融实验可复现。"
    ),
    104: (
        "（1）文本问答流（query_type: llm）：前端提交 params.input 与可选 history（最近 3 轮）。"
        "服务端先经 PhaseClassifier 判定震前/震中/震后，Scheduler 结合三信号决定 KG/RAG/动态源优先级；"
        "QueryContextBuilder 并行探测知识信号并组装【知识图谱】【参考资料】【动态信息】【阶段提示】等块，"
        "再套入 Qwen 对话模板，左侧截断保尾后由 LoRA 模型生成；响应可含元信息 meta（phase、schedule_reasoning、"
        "static_confidence、media_resources 等）及第三层地图/图表链接。"
    ),
    105: (
        "（2）图谱直连流（query_type: kg）：按 params.type 调用 all/by_region/by_magnitude/by_depth，"
        "不经大模型，供列表筛选与低延迟查询。"
    ),
    106: (
        "（3）数据更新流：POST /api/update-data 触发 kg.update_from_realtime_data()，"
        "对接 USGS GeoJSON 增量写入 Neo4j；动态问答路径另由 DynamicRetriever 带缓存拉取实时目录。"
    ),
    110: (
        "模块映射：前端（Vue/Android）→ app.py 路由 → services/context_builder.py → "
        "kg/neo4j_kg.py、rag/emergency_rag.py、core/dynamic_retriever.py → "
        "services/output_enricher.py（第三层）→ transformers+peft / qwen_vl_handler。"
        "docker-compose.yml 提供 Neo4j 与 Milvus 栈；本地开发可设 RAG_USE_MEMORY_RAG=True 免 Milvus。"
    ),
    146: (
        "任务目标。给定用户自然语言问句 q，系统输出中文回答 a，并尽量附带与阶段匹配的多模态资源"
        "（示意图、地图、余震图等）。问题覆盖震前科普准备、震中实时避险、震后评估恢复等场景。"
    ),
    147: (
        "形式化描述。记知识图谱 G、应急分块集合 D、动态源 Δ（USGS 等）。文本路径抽象为："
        "a = f_θ(Φ(q,G), Ψ(q,D), Δ(q), Σ(q), q)，其中 Φ 为图谱上下文，Ψ 为 RAG Top-K，"
        "Δ 为动态震情摘要，Σ 为阶段分类与调度产生的 prompt_suffix 与 validity_hint。"
        "纯图谱路径返回结构化列表，不经 f_θ。"
    ),
    148: (
        "用户交互。Web/Android 通过 POST /api/query 提交 llm 或 kg 请求；"
        "评测脚本可调用 GET /api/eval-set 获取震前/震中/震后各 20 条标准问句；"
        "图片问答走 POST /api/multimodal-query。"
    ),
    155: (
        "（5）消融与分阶段对比。关闭 KG_CONTEXT_ENABLED 或 RAG_ENABLED 构造四基线；"
        "进一步关闭 PHASE_CLASSIFIER_ENABLED、DYNAMIC_RETRIEVAL_ENABLED 观察阶段调度贡献；"
        "在 phase_questions.json 上按震前/震中/震后分别统计指标。"
    ),
    158: (
        "静态分支 Φ（图谱+RAG）职责：图谱提供可核对事件与主题—步骤关系（省区/震级/应急触发）；"
        "RAG 从 emergency_knowledge.json 召回语义相近 topic 片段。二者标注 phase_tag 与 temporal_validity（半衰期），"
        "供调度器判断静态置信度。"
    ),
    159: (
        "动态分支 Δ 职责：DynamicRetriever 在震中或高紧急度问句下拉取 USGS 实时目录，"
        "经中国区 bbox 过滤与缓存后写入【动态信息】；震后政策类问题可触发 policy_rss 摘要。"
        "MultimodalOutput 按阶段与关键词匹配 static/emergency 示意图资源。"
    ),
    160: (
        "调度原则。震前默认静态优先（use_dynamic=False）；震中默认开启动态且 KG 高优先级；"
        "震后按紧急度决定是否拉取动态源。当 urgency 高且 dynamic_availability 达标时优先动态；"
        "当 static_confidence≥0.9 时以静态为主并提示可追问最新震情；动态不可用时自动降级并标注 reliability_hint。"
    ),
    162: (
        "端到端流程：（1）阶段分类 PhaseClassifier.classify(q)；（2）探测知识信号 compute_knowledge_signals；"
        "（3）Scheduler.decide() 生成 ScheduleDecision；（4）组装 Φ、Ψ、Δ 与阶段 prompt_suffix；"
        "（5）LoRA 生成；（6）OutputEnricher 附加地图/避难所/余震图等媒体；（7）_sanitize_response 过滤异常标记。"
    ),
    166: (
        "对照设置：B0 仅 LLM；B1 仅图谱；B2 仅 RAG；B3 全协同（默认）。"
        "扩展对比可关闭阶段分类或动态检索，检验震中实时问句与震前科普问句上的差异增益。"
        "同一基座、LoRA、解码超参与分阶段问集，仅切换配置后批量请求并保存回答。"
    ),
    218: (
        "user 提示顺序：（1）角色引导；（2）【对话历史】（可选）；（3）【知识图谱】；（4）【参考资料】；"
        "（5）【动态信息】；（6）阶段调度 suffix 与 validity_hint；（7）规则与回答要求；（8）【问题】置于末尾。"
        "震中紧急问句的 suffix 要求短句、可执行指令，避免冗长解释。"
    ),
    222: (
        "对话模板与微调对齐：system/user/assistant 使用 <|system|>…</s> 标记；"
        "完整 user 后追加 <|assistant|> 作为生成起点。解码后剔除残留标记。"
        "多轮对话注入最近 CHAT_HISTORY_MAX_ROUNDS=3 轮，单条消息截断至 500 字符。"
    ),
    224: (
        "分词器 truncation_side=left，LLM_INPUT_MAX_TOKENS=4096，保留末尾【问题】与 assistant 标记，"
        "避免右侧截断导致空回答。"
    ),
    234: (
        "Flask 启动时完成 Neo4j 初始化、RAG 索引、阶段分类器、调度器、动态检索与第三层增强模块加载。"
        "GET / 返回 SPA；llm 分支返回 response、meta、media_resources；"
        "multimodal-query 走 Qwen2-VL 懒加载路径。"
    ),
    236: (
        "Web 前端（Vue 3 + Vite）采用应急深色主题。AppHeader 展示系统能力标签；"
        "QuickPanel 按震前/震中/震后分组快捷问句；ChatMessages 渲染 Markdown 与多模态卡片；"
        "Composer 支持文本发送。开发模式下可展示 phase、static_confidence、reliability_hint 等字段。"
    ),
    237: (
        "App.vue 管理对话状态、history 拼装与 meta 展示；api.js 封装 queryLlm、queryKg、updateData 等接口。"
    ),
    238: "ChatMessages.vue 负责消息列表、多模态资源展示与自动滚动。",
    239: "Composer.vue 提供输入框、发送状态与快捷键。",
    240: "QuickPanel.vue 提供三阶段胶囊式快捷提问。",
    241: "SearchHistory.vue 基于 localStorage 保存最近 30 条搜索记录。",
    242: "StatsBar.vue 与 InsightChart.vue 展示震级统计饼图。",
    243: (
        "Android 端（Jetpack Compose）复用同一 API：ChatRepository 提交 query_type=llm 与 history；"
        "HomePage 展示三阶段指引，SafetyPage 提供分阶段安全要点。ApiClient 默认指向开发机 8000 端口。"
    ),
    247: (
        "config/config.py 管理全部开关：KG_CONTEXT_ENABLED、RAG_ENABLED、PHASE_CLASSIFIER_ENABLED、"
        "DYNAMIC_RETRIEVAL_ENABLED、SCHEDULER_ENABLED、LAYER3_*、META_PAYLOAD_ENABLED 等。"
        "启动命令：docker compose up -d neo4j && python app.py，默认监听 0.0.0.0:8000。"
    ),
    252: (
        "实验环境：Apple Silicon（MPS）或 NVIDIA GPU 笔记本/服务器，Python 3.9，Neo4j 5-community（Docker），"
        "可选 Milvus v2.4.23（docker-compose）；本地开发可将 RAG_USE_MEMORY_RAG=True 以简化依赖。"
        "推理设备自动选择 MPS/CUDA/CPU，采样实验可固定 torch.manual_seed=42。"
    ),
    255: (
        "模型配置：基座 Qwen/Qwen2.5-7B-Instruct，LoRA 目录 llm/earthquake_expert_qwen25_7b，"
        "嵌入 BAAI/bge-small-zh-v1.5；图文路径 Qwen/Qwen2-VL-2B-Instruct（懒加载）。"
        "高德静态地图通过服务端代理，未配置 Key 时降级 OSM/URI。"
    ),
    257: (
        "图谱规模：Earthquake/Region/EmergencyTopic/GuidanceStep 节点及 OCCURRED_IN、HAS_STEP 等关系"
        "（以 Neo4j 实际统计为准）；RAG 为 topic 级 chunk，条数等于应急主题数；"
        "shelters.json 提供避难所 POI；multimodal_resources.json 提供分阶段示意图元数据。"
    ),
    258: (
        "评测问集：data/eval/phase_questions.json，震前/震中/震后各 20 条，共 60 条，"
        "通过 GET /api/eval-set 导出。RAG_TOP_K=5，RAG_MAX_CHUNK_CHARS=800。"
    ),
    264: (
        "评测流程：（1）按 B0～B3 切换配置并重启服务；（2）对 60 条问句批量 POST /api/query；"
        "（3）保存 response 与 meta；（4）两名标注者按事实一致性（0/0.5/1）、要点完整性（1～5）、"
        "格式合规（0/1）打分，并额外记录阶段判定是否正确、动态源是否被合理触发；"
        "（5）按震前/震中/震后汇总均值。表 7-3 数值请在实测后填入。"
    ),
    269: (
        "定量分析（待填入实测数据）：预期 B3 在混合问句与震中实时问句上综合最优；"
        "B1 在纯震情列表题接近 B3；B2 在震前规程题要点完整性较好；"
        "关闭动态检索后震中问句事实一致性应下降。请根据实验记录更新表 7-3。"
    ),
    270: (
        "分阶段观察：震前问句应主要命中 RAG 与静态科普，dynamic_availability 低；"
        "震中问句应触发 USGS 动态段与短句式回答；震后问句侧重政策 RSS 与房屋评估流程图。"
        "meta 元信息可用于核对调度 reasoning 与阶段标签是否一致。"
    ),
    271: (
        "协同机制验证：全协同在事实一致性上应显著优于 B0；相对 B1/B2 的增益体现在混合问句，"
        "说明单一路径不足以覆盖三阶段差异化需求。"
    ),
    275: (
        "案例1（震中+动态）：问「刚才地震多大？震中在哪？」B0 易幻觉；B3 应触发震中阶段、"
        "拉取 USGS 动态摘要并附震中地图（高德或 OSM 降级），事实一致性优于 B0/B2。"
    ),
    276: (
        "案例2（震前+RAG）：问「家庭应急包该准备什么？」B3 以震前调度为主，RAG 召回应急准备 topic，"
        "MultimodalOutput 附应急包示意图，完整性优于仅图谱的 B1。"
    ),
    277: (
        "案例3（混合）：问「甘肃 6 级以上地震有哪些，应该怎么避险？」B3 图谱返回事件列表，"
        "RAG 补充避险步骤，阶段标签为震中或通用，体现动态—静态协同。"
    ),
    281: (
        "本文围绕地震应急问答，提出并实现动态—静态知识协同方法，在图谱—向量—大模型框架上"
        "扩展三阶段调度、动态震情与多模态增强。主要工作包括："
    ),
    282: (
        "（1）Neo4j 知识图谱模式与子图查询，支撑可核对震情与应急步骤；"
    ),
    283: (
        "（2）topic 级 RAG（bge-small-zh-v1.5，Milvus/内存双模式）；"
    ),
    284: (
        "（3）震前/震中/震后分类、三信号调度、QueryContextBuilder 与第三层输出增强；"
    ),
    285: (
        "（4）Flask+Vue 3 Web、Android 客户端、Qwen2-VL 图文问答与分阶段评测接口；"
    ),
    286: (
        "（5）四基线消融与 60 题分阶段评测流程。实测定量结果见第 7 章（定稿前请替换为真实实验数据）。"
    ),
    288: (
        "（1）接入中国地震台网等国内权威动态源，替代单一 USGS 外链；扩充避难所与救援资源实体。"
    ),
    289: (
        "（2）将规则触发升级为神经语义解析或 Text2Cypher，提升复杂问句的图谱覆盖率。"
    ),
    290: (
        "（3）深化多模态：前端完整展示 layer3 地图缩略图，Android 端支持图片预览与导航跳转。"
    ),
    291: (
        "（4）推理加速：量化与 vLLM 部署，降低震中场景首 token 延迟。"
    ),
    292: (
        "（5）自动化评测脚本与更大规模标注集，报告标注者一致性与分阶段显著性检验。"
    ),
}

# 在锚点段落后插入新小节（从后往前插入，避免索引漂移）
INSERTIONS: list[tuple[int, list[str]]] = [
    (
        139,
        [
            "2.6  三阶段分类与不确定性感知调度",
            "2.6.1  震前、震中、震后的场景划分",
            "地震应急贯穿震前防御、震中响应、震后恢复。震前侧重科普与准备（长半衰期静态知识）；"
            "震中侧重实时震情与可执行避险（动态+静态）；震后侧重安全评估、政策与恢复（静态+动态混合）。"
            "系统将阶段作为调度先验，而非让人工切换模式。",
            "2.6.2  阶段分类器与紧急度",
            "core/phase_classifier.py 基于关键词分层打分判定震前/震中/震后/通用，"
            "并计算 urgency（「现在」「刚才」等）与 need_dynamic 标志，输出 PhaseResult 供调度器使用。",
            "2.6.3  三信号调度策略",
            "Scheduler 读取 static_confidence（RAG/图谱探测）、dynamic_availability（USGS 探测）"
            "与 urgency，按阶段调用 _schedule_pre/_during/_post，并在信号策略中处理动态不可用降级与可靠性提示。",
        ],
    ),
    (
        165,
        [
            "3.4  实验流程与可复现性",
            "3.4.1  服务启动与配置快照",
            "实验前执行 docker compose up -d neo4j，配置 .env 与 config.py 快照，"
            "记录 RAG_USE_MEMORY_RAG、各 SCHEDULER_* 阈值与 META_PAYLOAD_ENABLED 状态。",
            "3.4.2  批量请求与结果归档",
            "从 /api/eval-set 拉取 60 题，按基线循环：修改配置→重启 python app.py→"
            "脚本批量 POST /api/query（含 history 空列表）→保存 JSON（response、meta.phase、"
            "schedule_reasoning、media_resources）。",
            "3.4.3  人工评测与分阶段汇总",
            "标注者按第 3.1.2 节规则打分，并按震前/震中/震后聚合表 7-3 与案例截图，保证与实现 meta 字段可交叉验证。",
        ],
    ),
    (
        247,
        [
            "6.4  第三层输出增强与多模态",
            "6.4.1  OutputEnricher",
            "services/output_enricher.py 在生成后附加避难所推荐（shelter_service，优先高德 POI）、"
            "震中静态地图（amap_client 代理）、余震时序图（aftershock_chart）与政策 RSS 摘要（policy_rss）。",
            "6.4.2  多模态输入与资源匹配",
            "文本路径由 MultimodalOutput 匹配 emergency SVG 示意图；图片路径由 QwenVLHandler "
            "对上传图片与文本联合推理。LAYER3_INJECT_PROMPT 默认为 False，媒体 primarily 在前端展示。",
        ],
    ),
]


def apply_patch() -> None:
    if not THESIS.exists():
        raise FileNotFoundError(THESIS)

    backup = THESIS.with_name(
        f"华东师范大学硕士论文.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
    )
    shutil.copy2(THESIS, backup)
    print(f"已备份: {backup}")

    doc = Document(str(THESIS))

    for idx in sorted(INSERTIONS, key=lambda x: x[0], reverse=True):
        anchor, lines = idx
        if anchor >= len(doc.paragraphs):
            continue
        insert_block_after(doc.paragraphs[anchor], lines)

    for idx, text in REPLACEMENTS.items():
        if idx < len(doc.paragraphs):
            set_para_text(doc.paragraphs[idx], text)

    doc.save(str(THESIS))
    print(f"已更新: {THESIS}")


if __name__ == "__main__":
    apply_patch()
