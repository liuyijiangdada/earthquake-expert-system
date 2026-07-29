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
