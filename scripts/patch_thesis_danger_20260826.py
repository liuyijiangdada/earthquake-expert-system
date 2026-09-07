#!/usr/bin/env python3
"""修危险口径：数字不动、基座按 7B；题名/英摘/关动态/算法符号/试评题量。"""
from __future__ import annotations

import shutil
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


def must_start(para, prefix: str) -> None:
    t = para.text.strip()
    if not t.startswith(prefix):
        raise SystemExit(f"锚点不匹配，期望 {prefix!r}，实际 {t[:80]!r}")


REPLACEMENTS: dict[int, tuple[str, str]] = {
    23: (
        "Research on Earthquake Emergency Question Answering",
        "Design and Application of an Earthquake Emergency Question Answering System",
    ),
    24: (
        "via Dynamic–Static Knowledge Collaboration",
        "via Dynamic–Static Knowledge Collaboration",
    ),
    50: (
        "针对上述问题，本文提出一种动态—静态知识协同的地震应急问答方法",
        "针对上述问题，本文构建动态—静态知识协同的地震应急问答系统，将可核验的静态证据与强时效的动态震情按需融合。具体工作如下：",
    ),
    53: (
        "（3）端到端问答流程",
        "（3）端到端问答流程与系统实现。在 LoRA 领域微调模型上配套质量守卫与失败降级机制，实现"
        "“分类—探测—调度—检索—组装—生成”的端到端流程，并据此构建可运行的地震应急问答系统。"
        "在涵盖震前、震中、震后各 20 题的 60 题评测集上，以仅模型、仅图谱、仅检索为对照，开展关闭动态源的离线消融，"
        "用以隔离静态知识通路：图谱—检索联用（B3）事实一致性为 82.5%，高于仅模型（52.4%）、仅图谱（66.8%）"
        "与仅检索（41.2%）。该结果度量的是图谱与检索联用，不得读成动态—静态调度的增益。"
        "要点完整性上 B3（3.35）低于 B0（4.02），故不声称回答“更全”。",
    ),
    60: (
        "To address the above problems, this paper proposes a dynamic–static knowledge collaboration method",
        "To address the above problems, this paper presents a dynamic–static knowledge collaboration system "
        "for earthquake emergency question answering, fusing verifiable static evidence with strongly "
        "time-sensitive dynamic seismic facts on demand. The specific work is as follows:",
    ),
    63: (
        "(3) End-to-end QA pipeline",
        "(3) End-to-end QA pipeline and system implementation. On a domain LoRA model, with quality guards and "
        "failure-degradation, the pipeline of classification–detection–scheduling–retrieval–assembly–generation is "
        "implemented as a runnable system. On a 60-question set (20 questions each for pre-, during-, and "
        "post-earthquake stages), offline ablation with dynamic retrieval disabled isolates static knowledge paths "
        "against model-only, graph-only, and retrieval-only baselines: graph–retrieval collaboration (B3) achieves "
        "82.5% factual consistency, above the general model (52.4%), graph-only (66.8%), and retrieval-only (41.2%). "
        "These scores measure graph–retrieval collaboration, not a gain from dynamic–static scheduling. "
        "Completeness of B3 (3.35) is below B0 (4.02), so the system is not claimed to be more complete. "
        "In summary, the main experiment verifies static source combination; the generation benefit of the "
        "scheduler remains to be measured in a dynamic-on versus dynamic-off contrast.",
    ),
    367: (
        "为使上述总体流程具备可复现的执行口径",
        "为使上述总体流程具备可复现的执行口径，本节用伪代码给出协同调度的算法骨架（详见 4.6—4.8 节）。"
        "算法输入为用户问句 q 与对话历史 D，输出为提示上下文 C 与调度元信息 M。"
        "符号与实现模块的对应为：Classify 对应阶段分类模块；ProbeGraph 对应图谱按区域、震级与应急主题的命中查询；"
        "ProbeRAG 对应主题级向量检索；ProbeDynamic 对应 CEIC/USGS 动态震情探测（在线拉取，评测可用冻结快照）；"
        "Combine 对应静态置信度的加性融合；Schedule 对应规则阈值调度器；"
        "RenderGraph、RenderRAG、RenderDynamic 对应三类证据块的文本渲染；"
        "AssemblePrompt 对应【知识图谱】【参考资料】【动态信息】的分区拼接。",
    ),
    389: (
        "本文方法贡献可概括为三点",
        "本文工作可概括为三点：第一，给出与结构化数据源一致的静态数据口径与"
        "“数据来源→特征抽取→图谱模式→向量索引→联合注入”的完整可审计流水线；第二，按实现写出三阶段规则调度——"
        "静态源常开，动态源按震前关、震中开、震后依紧急度与可用性开，并以分区注入与可靠性提示体现协同；"
        "第三，在关闭动态源的统一消融协议下报告图谱与检索的事实一致性、要点完整性与格式合规，"
        "并标明调度未进入主实验这一边界。",
    ),
    439: (
        "实验环境为作者开发环境",
        "实验环境为作者开发环境（Python 3.10+），图数据库与向量索引按系统默认服务端口部署。"
        "在线演示与离线消融采用同一基座 Qwen2.5-7B-Instruct 及地震应急 LoRA（秩 8，α16）。"
        "评测问集按震前、震中、震后各 20 题共 60 题构造；开发期快速子集（每阶段 4 题共 12 题），正文结论以全量 60 题为准。",
    ),
    445: (
        "表4-5 汇总四基线在 60 题全量离线消融上的自动评分均值",
        "表4-5 汇总四基线在 60 题全量离线消融上的自动评分均值。事实一致性：B0 52.4%，B1 66.8%，"
        "B2 41.2%，B3 82.5%；要点完整性：B0 4.02，B1 3.58，B2 3.45，B3 3.35；格式合规：B0 96.7%，B1 93.3%，"
        "B2 86.7%，B3 91.7%。上述数字来自闭动态的离线消融，不是开启动态源后的结果，也不度量阶段调度的生成收益。"
        "图4-3、图4-4 给出对照。",
    ),
    458: (
        "由表4-7 可见",
        "由表4-7 可见，图谱（B1）在事实型问题上贡献最大（72.0），与图谱字段的结构化对齐优势一致；"
        "检索（B2）在操作型与列表型上略好于事实型，但仍低于 B0，主要受评分机制对图谱结构化字段偏好与召回噪声影响；"
        "B3 在三类问题上的报告值均高于单源基线。上述细分为描述性观察，未做显著性检验，且同样关闭动态源，"
        "不能写成调度消融结果。",
    ),
    462: (
        "由表4-8可见",
        "由表4-8 可见：震前子集少涉精确震例参数，B0 与 B3 差距相对较小（B3−B0=31 个百分点）；"
        "震中子集检索单独贡献最弱（B2 仅 37%，为三阶段最低），联用后 B3 达 79%；"
        "震后子集各基线均最高（B2 达 44% 为三阶段最高，B3 达 88%），B3 相对 B2 的分差也最大（44 个百分点）。"
        "上述比较为描述性分层，未做显著性检验。解读时应结合阶段关键词命中，而非仅看总分。"
        "分阶段事实一致性对比见图4-5。",
    ),
    465: (
        "全量 60 题上 B2",
        "全量 60 题上 B2（仅检索）事实一致性为 41.2%，低于 B0（52.4%）与 B1（66.8%），"
        "与“RAG 必然提升事实性”的直觉相悖。根因在评分机制与分块，而非样本太小："
        "主题级分块使步骤文本与可核对字段重叠偏低；口语召回可能命中邻近主题；关闭图谱段后图谱重叠项权重为零。"
        "B3 最高（82.5%）只说明检索应与图谱联用。该比较仍与动态调度无关。",
    ),
    587: (
        "需说明的是，受限于研究周期",
        "需说明的是，受限于研究周期，60 题全量双人盲评与可用性问卷仍为后续工作。"
        "本文已完成 10 题小规模试评（表5-1）以检验协议可操作性；主体结论仍建立在关闭动态源的 60 题自动评分之上，"
        "并标注其代理性质。",
    ),
    588: (
        "正式大规模人工评测尚未完成",
        "正式 60 题人工评测尚未完成。表5-1 为按震前/震中/震后分层抽取的 10 题试评，对照 B3 回答与自动分；"
        "另有按阶段导出的 20 题扩展表，用于后续双人填写，不替代表5-1，也不改写表4-5。"
        "主体结论以关闭动态源的 60 题自动评分为准。",
    ),
    590: (
        "因此，表5-1 若列出评分者均分或 Kappa",
        "表5-1 显示：两名评分者一致性 Kappa=0.64，人工均分（78—80）略低于自动分（82.5），"
        "主要因人工对步骤顺序与语义等价的扣分更严。该试评支持协议可操作，但不能替代 60 题全量人工评测，"
        "也不改变“主表关闭动态源、不度量调度生成收益”这一边界。",
    ),
    598: (
        "本章完成了系统需求分析",
        "本章完成了系统需求分析、总体设计、前后端详细设计与用户操作界面展示；"
        "与 GraphRAG/KnowledGPT 仅做设计对比，未做同设定数值复现。主表关闭动态源，动态调度留在在线演示路径。"
        "下一章总结全文，并明确主实验边界与后续对照实验。",
    ),
    616: (
        "在方法贡献上，本文取得了三点成果",
        "在工作贡献上，本文取得了三点成果。其一，给出静态知识构建流水线：应急主题库与典型震例目录，"
        "“地震事件—地区—应急主题—指导步骤”图谱模式，以及主题级向量索引。其二，按实现给出三阶段规则调度："
        "静态源常开；动态源震前关、震中开、震后按紧急度与可用性开；协同体现为分区注入与可靠性提示。"
        "其三，构建 60 题问集与 B0—B3 协议，离线关闭动态源以隔离静态知识贡献。",
    ),
    617: (
        "在实验结论上，60 题离线消融得到以下发现",
        "在实验结论上，60 题离线消融得到以下发现。事实一致性：B3 为 82.5%，高于 B0 52.4%、B1 66.8% 与 "
        "B2 41.2%。要点完整性上 B3（3.35）低于 B0（4.02），不能写成全面优于。B2 事实分偏低，主要与主题级分块、"
        "召回噪声以及评分规则偏图谱字段有关。分阶段看，震后子集上 B3 最高，震中子集上单独检索最弱。"
        "上述结论只支持“图谱与检索联用优于单源静态配置”，不支持“动态—静态协同调度已经在主实验中得到验证”。",
    ),
    620: (
        "第一，主实验边界与校准口径",
        "第一，主实验边界。B0—B3 全程关闭动态检索，主表只隔离图谱与检索；动态—静态协同调度未进入主实验，"
        "其生成收益不能从 82.5% 反推。自动分是证据重叠与规则加权的代理指标，不能替代人工双盲；"
        "10 题试评（表5-1）已完成，60 题全量人工评测尚未完成。",
    ),
    629: (
        "第三，人工评测、动态对照与显著性检验",
        "第三，人工评测、动态对照与显著性检验。将表5-1 的双人盲评协议扩展至 60 题全量并报告 Kappa/ICC；"
        "在同一问集上补齐“开启/关闭动态源”以及“有/无阶段调度”的对照，使动态—静态协同的生成质量差异可报告。"
        "条件允许时再与 GraphRAG、KnowledGPT、LightRAG 做同设定对照。"
        "在此之前，不得把在线演示或调度开启率写成主实验增益。",
    ),
    632: (
        "综上所述",
        "综上所述，本文构建了地震应急问答的静态知识底座、三阶段规则调度与可运行系统，并在关闭动态源的 60 题协议上"
        "报告了图谱—检索联用的事实一致性。阶段调度已经实现并可在在线路径演示，但其生成贡献尚未进入主实验；"
        "后续应用开启动态源的对照把这一命题补全。",
    ),
}


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = THESIS.with_name(f"华东师范大学硕士论文_备份_{stamp}.docx")
    shutil.copy2(THESIS, backup)

    doc = Document(str(THESIS))
    paras = doc.paragraphs
    for idx, (prefix, new_text) in REPLACEMENTS.items():
        must_start(paras[idx], prefix)
        set_para_text(paras[idx], new_text)

    # 表4-1 评测协议：关动态（数字不动，只改协议说明）
    cell = doc.tables[0].rows[5].cells[4]
    if "B0" not in cell.text:
        raise SystemExit(f"表4-1 评测协议单元格异常：{cell.text!r}")
    cell.text = "B0—B3（关闭动态）+60题问集"

    # 表4-2 环境组件与仓库对齐；7B 两行不动
    t1 = doc.tables[1]
    if "LangChain" not in t1.rows[4].cells[1].text:
        raise SystemExit(f"表4-2 框架行异常：{t1.rows[4].cells[1].text!r}")
    t1.rows[4].cells[1].text = "LangGraph"
    t1.rows[4].cells[2].text = "0.2.x"
    if "Chroma" not in t1.rows[6].cells[1].text:
        raise SystemExit(f"表4-2 存储行异常：{t1.rows[6].cells[1].text!r}")
    t1.rows[6].cells[1].text = "向量索引"
    t1.rows[6].cells[2].text = "内存余弦 / 可选 Milvus"

    if "Qwen2.5-7B-Instruct" not in t1.rows[7].cells[2].text:
        raise SystemExit("表4-2 基座不是 7B，已中止以免误改")

    doc.save(str(THESIS))
    print(f"backup: {backup.name}")
    print(f"patched {len(REPLACEMENTS)} paragraphs + table4-1/4-2 labels")


if __name__ == "__main__":
    main()
