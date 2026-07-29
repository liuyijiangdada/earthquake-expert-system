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


def test_abstract_mentions_60_and_honesty():
    from scripts.thesis_rewrite_content_6ch import ABSTRACT_CN
    assert "60" in ABSTRACT_CN
    assert "事实" in ABSTRACT_CN
    assert "完整" in ABSTRACT_CN  # 承认完整性未必全面领先


def test_ch1_defines_static_dynamic():
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


def test_ch4_collaboration_subjects_and_scheduler():
    ch4 = _chapter_blob("第四章")
    assert "静态知识" in ch4 and "动态知识" in ch4
    assert "震前" in ch4 and "震中" in ch4 and "震后" in ch4
    assert "阈值" in ch4 or "调度" in ch4
    assert "伪代码" in ch4 or "算法" in ch4 or "procedure" in ch4.lower() or "输入：" in ch4
    assert "CEIC" in ch4 or "台网" in ch4
    assert "USGS" in ch4
    assert "冲突" in ch4 or "降级" in ch4
