#!/usr/bin/env python3
"""按媛媛论文样式重建主目录、图目录、表目录。"""
from __future__ import annotations

import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

from docx import Document
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from lxml import etree

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCX = ROOT / "华东师范大学硕士论文.docx"

CAPTION_VERBS = re.compile(r"给出|展示|说明|如图|如表|所示|描述|为图|为表|是图|是表")


def is_caption(text: str, kind: str) -> bool:
    t = text.strip()
    if not re.match(rf"^{kind}\s*\d+-\d+\s+\S", t):
        return False
    if kind == "图" and len(t) > 60:
        return False
    if kind == "表" and len(t) > 80:
        return False
    if CAPTION_VERBS.search(t[:20]):
        return False
    return True


def parse_label(text: str, kind: str):
    m = re.match(rf"^({kind}\s*\d+-\d+)\s+(.+)$", text.strip())
    if not m:
        return None
    label = re.sub(r"\s+", "", m.group(1))
    nums = re.search(r"(\d+)-(\d+)", label)
    if not nums:
        return None
    return int(nums.group(1)), int(nums.group(2)), label, m.group(2).strip()
