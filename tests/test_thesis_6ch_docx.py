# tests/test_thesis_6ch_docx.py
from pathlib import Path
import subprocess
import sys
from docx import Document
from docx.shared import RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "华东师范大学硕士论文_6章修改版.docx"
THESIS = ROOT / "华东师范大学硕士论文.docx"
SUMMARY = ROOT / "data/eval/table_6_3_summary_60.json"
RED = RGBColor(0xFF, 0x00, 0x00)


def test_generate_and_validate_docx():
    assert THESIS.exists()
    cmd = [
        sys.executable,
        str(ROOT / "scripts/rewrite_thesis_6ch.py"),
        "--thesis",
        str(THESIS),
        "--output",
        str(OUT),
        "--fill-eval",
        str(SUMMARY),
        "--no-backup",
    ]
    subprocess.check_call(cmd, cwd=str(ROOT))
    assert OUT.exists()
    # 底稿仍在
    assert THESIS.exists()
    doc = Document(str(OUT))
    texts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    blob = "\n".join(texts)
    assert "第一章" in blob and "第六章" in blob
    assert "第七章" not in blob
    toc_i = next(i for i, t in enumerate(texts) if t == "目录")
    toc_end = next(
        i for i, t in enumerate(texts) if t.startswith("第一章") or t in {"图目录", "表目录"}
    )
    assert "第七章" not in "\n".join(texts[toc_i:toc_end])
    assert "82.5" in blob  # FACT_B3 filled
    assert "{{FACT_B3}}" not in blob
    # 致谢存在且其后不应再塞章节
    assert any(t == "致谢" for t in texts)
    red_paras = 0
    for p in doc.paragraphs:
        for r in p.runs:
            if r.font.color and r.font.color.rgb == RED:
                red_paras += 1
                break
    assert red_paras >= 50
