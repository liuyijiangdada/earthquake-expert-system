#!/usr/bin/env python3
"""项目级共享常量（省名列表等），供 KG 注入与 Neo4j 解析共用。"""

from __future__ import annotations

from typing import List, Optional

# 与 kg/neo4j_kg 区域推断、问句子串匹配保持一致；较长名称放前面避免误匹配
CHINA_REGION_NAMES: List[str] = [
    "四川", "云南", "青海", "西藏", "新疆", "甘肃", "河北", "台湾", "广东", "辽宁",
    "北京", "上海", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北",
    "湖南", "广西", "海南", "重庆", "贵州", "陕西", "吉林", "黑龙江", "内蒙古",
    "宁夏", "香港", "澳门",
]


def match_region_in_text(text: str) -> Optional[str]:
    """返回问句中命中的第一个省级名称，无命中则 None。"""
    if not text:
        return None
    for region in CHINA_REGION_NAMES:
        if region in text:
            return region
    return None
