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
