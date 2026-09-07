#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重排论文第三～五章正文，使小节与大章标题一致。

目标：
  第三章 面向地震应急的多源静态知识构建
    ← 原 4.3–4.4（图谱+向量），并补 3.1/3.2/3.5
  第四章 动态—静态协同的地震应急问答方法与评估
    ← 原第3章总体方法(3.1–3.7) + 原4.1–4.2 + 原4.5–4.9 + 原5.4.1–5.4.4
  第五章 多源知识驱动的地震应急问答系统
    ← 原5.1–5.3 + 5.4.5–5.4.6 + 5.5
"""
from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from copy import deepcopy

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "华东师范大学硕士论文.docx"


def T(p) -> str:
    return (p.text or "").strip()


def set_text(p, text: str) -> None:
    if p.runs:
        p.runs[0].text = text
        for r in p.runs[1:]:
            r.text = ""
    else:
        p.add_run(text)


def is_toc(p) -> bool:
    name = p.style.name if p.style else ""
    return bool(name) and name.lower().startswith("toc")


def body_start(doc: Document) -> int:
    for i, p in enumerate(doc.paragraphs):
        if is_toc(p):
            continue
        if T(p).startswith("第一章"):
            return i
    raise SystemExit("无正文第一章")


def heading_indices(doc: Document, start: int) -> list[tuple[int, str]]:
    out = []
    for i in range(start, len(doc.paragraphs)):
        t = T(doc.paragraphs[i])
        if not t or len(t) > 100 or is_toc(doc.paragraphs[i]):
            continue
        if t.startswith(("第", "1.", "2.", "3.", "4.", "5.", "6.", "附录", "参考文献", "致谢")):
            out.append((i, t))
        elif t in ("相关技术与理论基础", "面向地震应急的多源静态知识构建"):
            out.append((i, t))
    return out


def find_title(heads: list[tuple[int, str]], pred) -> tuple[int, str]:
    for i, t in heads:
        if pred(t):
            return i, t
    raise KeyError(pred)


def range_to(heads, start_title_pred, end_title_pred) -> tuple[int, int]:
    si, _ = find_title(heads, start_title_pred)
    ei, _ = find_title(heads, end_title_pred)
    return si, ei


def elems(doc, a, b):
    return [doc.paragraphs[i]._element for i in range(a, b)]


def insert_before(anchor_el, elements):
    """按 elements 原有顺序插到 anchor 前。"""
    for el in elements:
        anchor_el.addprevious(el)


def make_para(doc: Document, text: str, style_like=None):
    """创建段落；返回可移动的 XML element（已从文末挂起，调用方 addprevious）。"""
    p = doc.add_paragraph(text)
    el = p._element
    if style_like is not None:
        src = style_like._element.pPr
        if src is not None:
            # 替换或插入 pPr
            old = el.pPr
            if old is not None:
                el.remove(old)
            el.insert(0, deepcopy(src))
    return el


def main():
    if not DOC.exists():
        raise SystemExit(DOC)
    bak = DOC.with_name(f"{DOC.stem}.backup_ch345_{datetime.now():%Y%m%d_%H%M%S}{DOC.suffix}")
    shutil.copy2(DOC, bak)
    print("备份:", bak)

    doc = Document(str(DOC))
    bi = body_start(doc)
    heads = heading_indices(doc, bi)

    i_ch3, _ = find_title(heads, lambda t: "多源静态知识构建" in t or t.startswith("第三章"))
    i_ch4, _ = find_title(heads, lambda t: t.startswith("第四章"))
    i_ch5, _ = find_title(heads, lambda t: t.startswith("第五章"))
    i_ch6, _ = find_title(heads, lambda t: t.startswith("第六章"))

    i_31, _ = find_title(heads, lambda t: t.startswith("3.1"))
    i_37_end = find_title(heads, lambda t: t.startswith("3.7"))[0]
    # 3.7 块结束 = 第四章标题
    i_41, _ = find_title(heads, lambda t: t.startswith("4.1"))
    i_42, _ = find_title(heads, lambda t: t.startswith("4.2"))
    i_43, _ = find_title(heads, lambda t: t.startswith("4.3"))
    i_44, _ = find_title(heads, lambda t: t.startswith("4.4"))
    i_45, _ = find_title(heads, lambda t: t.startswith("4.5"))
    # 4.9 块结束 = 第五章
    i_51, _ = find_title(heads, lambda t: t.startswith("5.1"))
    i_54, _ = find_title(heads, lambda t: t.startswith("5.4") and "实验" in t)
    i_541, _ = find_title(heads, lambda t: t.startswith("5.4.1"))
    i_545, _ = find_title(heads, lambda t: t.startswith("5.4.5"))
    i_55, _ = find_title(heads, lambda t: t.startswith("5.5"))

    # 块 [start, end)
    ch3_title = elems(doc, i_ch3, i_ch3 + 1)
    method = elems(doc, i_31, i_ch4)          # 原错误第3章 3.1–3.7
    b41 = elems(doc, i_41, i_42)
    b42 = elems(doc, i_42, i_43)
    b43 = elems(doc, i_43, i_44)
    b44 = elems(doc, i_44, i_45)
    b45_49 = elems(doc, i_45, i_ch5)
    ch4_title = elems(doc, i_ch4, i_ch4 + 1)
    ch5_title = elems(doc, i_ch5, i_ch5 + 1)
    b51_53 = elems(doc, i_51, i_54)
    b541_544 = elems(doc, i_541, i_545)
    b545_546 = elems(doc, i_545, i_55)
    b55 = elems(doc, i_55, i_ch6)

    body = doc.element.body
    ch6_el = doc.paragraphs[i_ch6]._element

    # 移除 Ch3 标题到 Ch6 前全部
    to_remove = elems(doc, i_ch3, i_ch6)
    for el in to_remove:
        body.remove(el)

    # 第三章导读段落
    style_ref = doc.paragraphs[min(bi + 5, len(doc.paragraphs) - 1)]
    intro_31_h = make_para(doc, "3.1  研究动机与贡献", style_ref)
    intro_31_b = make_para(
        doc,
        "本章承接全文对静态证据层的需求，聚焦可离线固化的多源静态知识构建："
        "以知识图谱承载可核验的震例与应急主题结构，以主题级向量索引覆盖口语化问句召回，"
        "二者互补为后续动态—静态协同问答提供可核对的证据底座。本章不展开实时源启闭与阶段调度。",
        style_ref,
    )
    intro_32_h = make_para(doc, "3.2  问题定义", style_ref)
    intro_32_b = make_para(
        doc,
        "设输入为应急主题库与典型震例目录，输出为可联合服务问答的静态证据层："
        "图谱库（支持结构化查询）与向量索引（支持语义 Top-K 召回）。"
        "在阶段标签约束下，静态层应产出图谱摘要与检索摘要并在提示中分区呈现，"
        "以满足字段刚性、步骤完整性、口语覆盖与可复现四项约束。",
        style_ref,
    )
    end_35_h = make_para(doc, "3.5  本章小结", style_ref)
    end_35_b = make_para(
        doc,
        "本章完成了面向地震应急的多源静态知识构建：给出图谱模式与导入口径，"
        "并建立主题级向量索引及其与图谱的互补注入规则，形成可核验的静态证据层。"
        "下一章在此基础上展开动态—静态协同问答方法，并完成实验评估。",
        style_ref,
    )

    new_order = []
    new_order += ch3_title
    new_order += [intro_31_h, intro_31_b, intro_32_h, intro_32_b]
    new_order += b43
    new_order += b44
    new_order += [end_35_h, end_35_b]
    new_order += ch4_title
    new_order += method
    new_order += b41
    new_order += b42
    new_order += b45_49
    new_order += b541_544
    new_order += ch5_title
    new_order += b51_53
    new_order += b545_546
    new_order += b55

    insert_before(ch6_el, new_order)
    # 文末可能残留 make_para 产生的空壳？make_para 的 element 已从末尾 remove 了吗？
    # add_paragraph 先挂到末尾，我们只把 el 用 addprevious 挪走了，不应残留重复。
    # 但 make_para 没有从 body 末尾 remove 就 addprevious——addprevious 会移动节点，OK。

    # —— 按文档顺序重命名标题 ——
    state = None
    for p in doc.paragraphs[body_start(doc) :]:
        t = T(p)
        if not t or len(t) > 100:
            continue
        if "多源静态知识构建" in t:
            set_text(p, "第三章  面向地震应急的多源静态知识构建")
            state = "ch3"
            continue
        if t.startswith("第四章"):
            set_text(p, "第四章  动态—静态协同的地震应急问答方法与评估")
            state = "ch4"
            ch4_phase = "method"  # method → detail → eval
            continue
        if t.startswith("第五章"):
            set_text(p, "第五章  多源知识驱动的地震应急问答系统")
            state = "ch5"
            continue
        if t.startswith("第六章"):
            state = "done"
            continue

        if state == "ch3":
            m = re.match(r"^4\.3(\.\d+)?\s+(.*)$", t)
            if m:
                set_text(p, f"3.3{m.group(1) or ''}  {m.group(2)}")
                continue
            m = re.match(r"^4\.4(\.\d+)?\s+(.*)$", t)
            if m:
                set_text(p, f"3.4{m.group(1) or ''}  {m.group(2)}")
                continue

        if state == "ch4":
            # 原 3.x 总体方法 → 4.1–4.7
            m = re.match(r"^3\.([1-7])(\.\d+)?\s+(.*)$", t)
            if m:
                n = int(m.group(1))
                set_text(p, f"4.{n}{m.group(2) or ''}  {m.group(3)}")
                if n == 7:
                    ch4_phase = "detail"
                continue
            # 原 4.1/4.2
            if re.match(r"^4\.1\s+", t) and "研究动机" in t:
                set_text(p, "4.8  " + t.split("  ", 1)[-1])
                continue
            if re.match(r"^4\.2\s+", t):
                set_text(p, "4.9  " + t.split("  ", 1)[-1])
                continue
            # 原 4.5–4.9 → 4.10–4.14
            m = re.match(r"^4\.([5-9])(\.\d+)?\s+(.*)$", t)
            if m:
                old = int(m.group(1))
                new = old + 5  # 5→10 ... 9→14
                set_text(p, f"4.{new}{m.group(2) or ''}  {m.group(3)}")
                continue
            # 实验 5.4.1–5.4.4 → 4.15–4.18
            m = re.match(r"^5\.4\.([1-4])\s+(.*)$", t)
            if m:
                new = 14 + int(m.group(1))  # 1→15
                set_text(p, f"4.{new}  {m.group(2)}")
                continue

        if state == "ch5":
            if t.startswith("5.4") and "实验" in t:
                set_text(p, "5.4  系统可用性与性能评测")
                continue
            m = re.match(r"^5\.4\.5\s+(.*)$", t)
            if m:
                set_text(p, f"5.4.1  {m.group(1)}")
                continue
            m = re.match(r"^5\.4\.6\s+(.*)$", t)
            if m:
                set_text(p, f"5.4.2  {m.group(1)}")
                continue

    # 交叉引用与 1.4
    for p in doc.paragraphs[body_start(doc) :]:
        t = p.text or ""
        nt = t
        reps = [
            ("详见第四章4.3—4.4节", "详见第三章3.3—3.4节"),
            ("详见第四章4.3—4.4", "详见第三章3.3—3.4"),
            ("详见第四章4.9节", "详见第四章动态知识接入相关节"),
            ("用于第五章的数值主表", "用于第四章的数值主表"),
            ("第五章将结合机制讨论", "第四章将结合机制讨论"),
            ("第四章4.3—4.4节", "第三章3.3—3.4节"),
            ("第四章4.9节", "第四章动态知识接入相关节"),
            ("静态知识层（知识图谱构建与主题级向量检索，详见第四章4.3—4.4节）",
             "静态知识层（知识图谱构建与主题级向量检索，详见第三章3.3—3.4节）"),
            ("动态知识层（CEIC/USGS 接入与融合，详见第四章4.9节）",
             "动态知识层（CEIC/USGS 接入与融合，详见第四章相关节）"),
        ]
        for a, b in reps:
            nt = nt.replace(a, b)
        if nt != t:
            set_text(p, nt)

    for p in doc.paragraphs[body_start(doc) :]:
        t = T(p)
        if "全文共六章" in t and ("第三章" in t or "组织结构" in t or "环环相扣" in t or "逻辑闭环" in t):
            set_text(
                p,
                "全文共六章。第一章为绪论；第二章为相关技术与理论基础；"
                "第三章阐述面向地震应急的多源静态知识构建（知识图谱与主题级向量检索）；"
                "第四章阐述动态—静态协同的地震应急问答方法与评估（总体框架、协同调度、动态接入与60题实验）；"
                "第五章给出多源知识驱动的地震应急问答系统；第六章总结与展望。"
                "第三章→第四章→第五章环环相扣：先构建可核验静态证据，再完成协同方法与评估，最后落地为可运行系统。",
            )
            break

    # 文前目录：按 toc 样式顺序改标题（保留页码）
    toc_titles = [
        "第一章  绪论",
        "1.1  研究背景与意义",
        "1.1.1  地震应急场景下的信息需求与问答痛点",
        "1.1.2  大模型应用中的事实性与可核对性挑战",
        "1.2  国内外研究现状",
        "1.2.1  国内地震应急知识管理与信息服务",
        "1.2.2  知识图谱与向量检索在问答中的应用",
        "1.2.3  开放域 RAG 与 KG 结合方法",
        "1.2.4  现有不足与本文切入点",
        "1.3  研究目标与内容",
        "1.4  论文组织结构",
        "第二章 相关技术与理论基础",
        "2.1  知识图谱构建技术",
        "2.2  向量检索与RAG",
        "2.3  动态数据接入与实时处理",
        "2.4  问答系统评价指标",
        "2.5  本章小结",
        "第三章 面向地震应急的多源静态知识构建",
        "3.1  研究动机与贡献",
        "3.2  问题定义",
        "3.3  地震应急知识图谱构建",
        "3.3.1  数据来源与原始格式",
        "3.3.2  特征抽取与预处理",
        "3.3.3  图谱模式设计",
        "3.4  基于向量的应急文档检索",
        "3.4.1  主题级分块及适用性",
        "3.4.2  向量化与索引",
        "3.4.3  检索与上下文增强",
        "3.5  本章小结",
        "第四章  动态—静态协同的地震应急问答方法与评估",
        "4.1  研究思路与总体框架",
        "4.2  核心概念与操作性定义",
        "4.3  动态—静态协同问答总体流程",
        "4.4  协同调度算法概述",
        "4.5  与已有方法的差异定位",
        "4.6  方法贡献与章节组织",
        "4.7  本章小结",
        "4.8  协同调度方法的研究动机",
        "4.9  问题定义",
        "4.10  协同调度方法的研究动机",  # placeholder overwritten below from actual
    ]
    # 用正文实际标题生成文前目录更稳
    bi2 = body_start(doc)
    real = []
    for i, t in heading_indices(doc, bi2):
        if t.startswith("参考文献"):
            break
        if t.startswith(("第", "1.", "2.", "3.", "4.", "5.", "6.")):
            real.append(t)
    toc_ps = [p for p in doc.paragraphs if is_toc(p)]
    # 截到参考文献 toc 为止
    for j, p in enumerate(toc_ps):
        if j < len(real):
            raw = p.text or ""
            title = real[j]
            if "\t" in raw:
                set_text(p, f"{title}\t{raw.split(chr(9))[-1]}")
            else:
                m = re.match(r"^(.*?)(\s+\d+\s*)$", raw)
                set_text(p, f"{title}{m.group(2)}" if m else title)
        else:
            # 多余 toc 行清空（章节变少时）
            if T(p).startswith(("3.", "4.", "5.", "第")):
                set_text(p, "")

    # 若 real 更长，不强行插入 toc 行（避免破坏域）；提示用户更新目录
    print(f"正文标题数={len(real)}, 文前toc行={len(toc_ps)}")

    # 同步内容模块章名
    mod = ROOT / "scripts/thesis_rewrite_content_6ch.py"
    if mod.exists():
        s = mod.read_text(encoding="utf-8")
        s2 = s
        for a, b in [
            ('"第三章 静态知识构建与检索方法"', '"第三章 面向地震应急的多源静态知识构建"'),
            ('"第四章 动态—静态知识协同调度方法"', '"第四章 动态—静态协同的地震应急问答方法与评估"'),
            ('"第五章 系统实现与实验分析"', '"第五章 多源知识驱动的地震应急问答系统"'),
            ('"第三章  静态知识构建与检索方法"', '"第三章  面向地震应急的多源静态知识构建"'),
            ('"第四章  动态—静态知识协同调度方法"', '"第四章  动态—静态协同的地震应急问答方法与评估"'),
            ('"第五章  系统实现与实验分析"', '"第五章  多源知识驱动的地震应急问答系统"'),
        ]:
            s2 = s2.replace(a, b)
        if s2 != s:
            mod.write_text(s2, encoding="utf-8")
            print("已同步内容模块章标题")

    doc.save(str(DOC))
    print("已写入", DOC)

    print("\n正文第三～五章标题：")
    state = None
    for i, t in heading_indices(doc, body_start(doc)):
        if t.startswith("第三") or "多源静态" in t:
            state = "p"
        if state == "p":
            print(" ", t)
        if t.startswith("第六"):
            break


if __name__ == "__main__":
    main()
