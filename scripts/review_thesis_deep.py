#!/usr/bin/env python3
"""Deep honesty review of thesis docx."""
import re
import json
from docx import Document

DOCX = "/Users/xiaoxiaoqingnian/Desktop/biyelunwen/华东师范大学硕士论文.docx"

OVERCLAIM_PATTERNS = [
    (r"显著", "显著"),
    (r"验证了有效", "验证了有效"),
    (r"验证.*有效", "验证有效(正则)"),
    (r"不确定性感知", "不确定性感知"),
    (r"隔离静态与动态", "隔离静态与动态"),
    (r"开启动态.*提升|动态.*提升.*事实", "开启动态提升"),
    (r"\b4B\b|4B模型|4B 模型", "4B"),
    (r"1\.8B|1\.8 B|Qwen1\.5-1\.8B", "1.8B"),
    (r"7B|Qwen2\.5-7B|7B-Instruct", "7B"),
    (r"LangChain", "LangChain"),
    (r"并行探测", "并行探测"),
    (r"16题|16 题", "16题"),
    (r"91\.2%|91\.2％", "91.2%"),
    (r"人工评测已完成|人工评估已完成|大规模人工评测.*完成", "人工评测已完成"),
    (r"Kappa\s*[=＝]\s*0\.64|Kappa.*0\.64|κ\s*=\s*0\.64", "Kappa=0.64"),
    (r"P50|P95|延迟.*P50", "P50"),
    (r"方法研究", "方法研究"),
    (r"快准全", "快准全"),
    (r"全面显著|显著优于|显著提升", "显著/显著提升"),
    (r"动态—静态协同.*验证|协同调度.*验证|调度.*已验证", "调度已验证"),
    (r"LightRAG", "LightRAG"),
    (r"GraphRAG.*复现|KnowledGPT.*复现|同设定.*数值", "同设定复现"),
    (r"GB/T\s*7714|GB7714", "GB/T7714"),
    (r"显著性检验|t检验|p\s*[<＜]\s*0\.05", "显著性检验"),
    (r"B4|B5|B3-ns", "B4/B5"),
    (r"双盲.*完成|双人.*完成.*评测", "人工双盲完成"),
]

def para_context(paras, idx, window=0):
    p = paras[idx]
    return {
        "idx": idx,
        "style": p.style.name if p.style else "",
        "text": p.text.strip()[:350],
    }

def find_ref_start(paras):
    for i, p in enumerate(paras):
        t = p.text.strip()
        if re.match(r"^(参考文献|References)$", t):
            return i
    return len(paras)

def extract_title_abstract(paras):
    results = {"cn_title": [], "en_title": [], "cn_abstract": [], "en_abstract": [], 
               "cn_keywords": [], "en_keywords": [], "innovation": [], "toc": []}
    state = None
    for i, p in enumerate(paras[:250]):
        t = p.text.strip()
        s = p.style.name if p.style else ""
        if not t:
            continue
        if i < 25 and len(t) > 8 and ("地震" in t or "Earthquake" in t):
            if "基于" in t or "Design" in t or "Application" in t:
                if re.search(r"[\u4e00-\u9fff]", t):
                    results["cn_title"].append({"idx": i, "text": t, "style": s})
                else:
                    results["en_title"].append({"idx": i, "text": t, "style": s})
        if t in ("摘要",) or t.startswith("摘  要"):
            state = "cn_abs"
            continue
        if t in ("Abstract", "ABSTRACT", "英文摘要"):
            state = "en_abs"
            continue
        if t.startswith("关键词") or t.startswith("Key words") or t.startswith("Keywords"):
            if "Key" in t or "key" in t:
                state = "en_kw"
                results["en_keywords"].append({"idx": i, "text": t})
            else:
                state = "cn_kw"
                results["cn_keywords"].append({"idx": i, "text": t})
            continue
        if t in ("目录", "目  录", "CONTENTS"):
            state = "toc"
            continue
        if "创新点" in t or t == "本文创新点" or t == "主要创新点":
            results["innovation"].append({"idx": i, "text": t, "style": s})
            state = "innovation"
            continue
        if state == "cn_abs" and not t.startswith("关键词"):
            results["cn_abstract"].append({"idx": i, "text": t[:300]})
        elif state == "en_abs" and not t.startswith("Key"):
            results["en_abstract"].append({"idx": i, "text": t[:300]})
        elif state == "innovation" and len(t) > 5:
            results["innovation"].append({"idx": i, "text": t[:300], "style": s})
        elif state == "toc" and len(t) > 3:
            results["toc"].append({"idx": i, "text": t[:120]})
        if re.match(r"^第[一二三四五六七八九十\d]+章", t):
            state = None
    return results

def main():
    doc = Document(DOCX)
    paras = doc.paragraphs
    ref_start = find_ref_start(paras)

    # Outline
    outline = []
    for i, p in enumerate(paras):
        t = p.text.strip()
        if not t:
            continue
        s = p.style.name if p.style else ""
        is_h = False
        if any(x in s for x in ["Heading", "标题", "heading"]) or re.match(r"^[1-9]$", s or ""):
            is_h = True
        elif re.match(r"^第[一二三四五六七八九十\d]+章", t):
            is_h = True
        elif re.match(r"^\d+(\.\d+)*\s+[\u4e00-\u9fff]", t) and len(t) < 100:
            is_h = True
        if is_h:
            outline.append({"idx": i, "style": s, "text": t})

    chapters = [o for o in outline if re.match(r"^第[一二三四五六七八九十\d]+章", o["text"])]

    # Char count
    cn = 0
    for i, p in enumerate(paras):
        if i >= ref_start:
            break
        cn += sum(1 for c in p.text if "\u4e00" <= c <= "\u9fff")

    # Overclaim scan (exclude refs)
    issues = []
    for i, p in enumerate(paras):
        if i >= ref_start:
            continue
        t = p.text
        for pat, label in OVERCLAIM_PATTERNS:
            if re.search(pat, t, re.I):
                # skip if it's explicitly negating the claim
                neg = any(x in t for x in ["尚未", "未完成", "不纳入", "占位", "不能", "不得", "未进入", "未做", "未补齐", "不是"])
                issues.append({
                    "label": label,
                    "idx": i,
                    "negated_context": neg,
                    "text": t.strip()[:280],
                    "style": p.style.name if p.style else "",
                })

    meta = extract_title_abstract(paras)

    # All tables dump for 5-1 and env
    table_info = []
    for ti, tbl in enumerate(doc.tables):
        rows = [[c.text.strip().replace("\n", " | ") for c in r.cells] for r in tbl.rows]
        flat = " ".join(" ".join(r) for r in rows)
        table_info.append({
            "index": ti,
            "nrows": len(rows),
            "preview": rows[:8],
            "flat_snip": flat[:600],
            "has_kappa": "Kappa" in flat or "kappa" in flat or "0.64" in flat,
            "has_qwen": "Qwen" in flat or "qwen" in flat,
            "has_langchain": "LangChain" in flat or "langchain" in flat.lower(),
            "has_b_scores": "82.5" in flat or "52.4" in flat,
            "caption_nearby": None,
        })

    # Find paragraphs mentioning 表5-1 / 表 5-1
    t51_refs = []
    for i, p in enumerate(paras):
        if "表5-1" in p.text or "表 5-1" in p.text:
            t51_refs.append(para_context(paras, i))

    # 不纳入结论 mentions
    exclude_refs = [para_context(paras, i) for i, p in enumerate(paras) if "不纳入" in p.text]

    # English title from first pages
    en_title_full = ""
    for item in meta["en_title"]:
        en_title_full = item["text"]

    out = {
        "cn_chars_body": cn,
        "approx_pages": round(cn / 800, 1),
        "ref_start_idx": ref_start,
        "chapter_count": len(chapters),
        "chapters": chapters,
        "outline": outline,
        "meta": meta,
        "overclaim_issues": issues,
        "table_info": table_info,
        "t51_text_refs": t51_refs,
        "exclude_refs": exclude_refs,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
