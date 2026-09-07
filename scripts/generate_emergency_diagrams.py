#!/usr/bin/env python3
"""生成应急示意图 PNG（避免 SVG 在 img 标签中中文乱码）。

优先使用 matplotlib；若未安装则回退到 Pillow。
"""

from __future__ import annotations

import os
from typing import List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "static", "emergency")

Diagram = Tuple[str, str, List[str], str, str]

DIAGRAMS: List[Diagram] = [
    (
        "indoor_shelter.png",
        "室内避险示意图",
        ["伏地 · 遮挡 · 手抓牢", "就近躲到桌下或坚固家具旁", "保护头颈，远离窗户和易倒物品", "震动停止后再有序撤离"],
        "#e8f4fc",
        "#2563eb",
    ),
    (
        "outdoor_safety.png",
        "室外避险要点",
        ["迅速前往开阔地带", "远离建筑物、围墙和广告牌", "避开电线杆、路灯和树木", "驾车时应减速停车并留在车内"],
        "#ecfdf5",
        "#059669",
    ),
    (
        "emergency_kit.png",
        "家庭应急包物资清单",
        ["· 饮用水与压缩食品", "· 手电筒、电池、口哨", "· 急救包、常用药品", "· 保暖衣物、雨具、充电宝", "· 证件复印件与联系人信息"],
        "#fff7ed",
        "#ea580c",
    ),
    (
        "building_damage.png",
        "震后房屋安全评估要点",
        ["观察墙体、梁柱是否出现裂缝或倾斜", "检查门窗是否变形、地面是否隆起", "闻到燃气异味应立即关阀并撤离", "疑似危房时勿擅自进入，等待鉴定"],
        "#fef2f2",
        "#dc2626",
    ),
    (
        "aftershock.png",
        "余震与次生灾害防范",
        ["主震后仍可能有较强余震", "警惕滑坡、泥石流、堰塞湖", "远离受损建筑、边坡和河道", "持续关注官方预警与安置指引"],
        "#faf5ff",
        "#7c3aed",
    ),
    (
        "earthquake_drill.png",
        "学校地震演练疏散示意图",
        ["1. 听到信号后迅速就地避险", "2. 按指定路线有序撤离至操场", "3. 清点人数，勿拥挤、勿返回教室", "4. 听从老师指挥，等待通知"],
        "#eff6ff",
        "#2563eb",
    ),
    (
        "highrise_escape.png",
        "高层建筑地震避险要点",
        ["就地选择小开间或坚固家具旁避险", "不要使用电梯，不要跳楼", "撤离时走楼梯，扶好扶手", "到室外后远离玻璃幕墙和悬挂物"],
        "#f0f9ff",
        "#0284c7",
    ),
]


def _hex_to_rgb(color: str) -> Tuple[int, int, int]:
    c = color.lstrip("#")
    return tuple(int(c[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _load_chinese_font(size: int):
    from PIL import ImageFont

    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=size, index=0)
            except OSError:
                continue
    return ImageFont.load_default()


def render_with_pillow(filename: str, title: str, lines: List[str], bg: str, accent: str) -> None:
    from PIL import Image, ImageDraw

    width, height = 768, 432
    img = Image.new("RGB", (width, height), _hex_to_rgb(bg))
    draw = ImageDraw.Draw(img)
    margin = 36
    draw.rounded_rectangle(
        [margin, margin, width - margin, height - margin],
        radius=18,
        fill=(255, 255, 255),
        outline=_hex_to_rgb(accent),
        width=3,
    )
    title_font = _load_chinese_font(28)
    body_font = _load_chinese_font(20)
    draw.text((width // 2, 88), title, fill=_hex_to_rgb(accent), font=title_font, anchor="mm")
    y = 150
    step = 42 if len(lines) <= 4 else 36
    for line in lines:
        draw.text((width // 2, y), line, fill=(51, 65, 85), font=body_font, anchor="mm")
        y += step
    out_path = os.path.join(OUT_DIR, filename)
    img.save(out_path, format="PNG", optimize=True)
    print(f"Wrote {out_path}")


def render_with_matplotlib(filename: str, title: str, lines: List[str], bg: str, accent: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.patches import FancyBboxPatch

    rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Heiti SC",
        "STHeiti",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=120)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    card = FancyBboxPatch(
        (0.06, 0.1),
        0.88,
        0.8,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=2,
        edgecolor=accent,
        facecolor="white",
    )
    ax.add_patch(card)
    ax.text(0.5, 0.78, title, ha="center", va="center", fontsize=16, color=accent, fontweight="bold")
    y = 0.66
    step = 0.11 if len(lines) <= 4 else 0.09
    for line in lines:
        ax.text(0.5, y, line, ha="center", va="center", fontsize=11, color="#334155")
        y -= step
    out_path = os.path.join(OUT_DIR, filename)
    fig.savefig(out_path, bbox_inches="tight", facecolor=bg)
    plt.close(fig)
    print(f"Wrote {out_path}")


def render_diagram(filename: str, title: str, lines: List[str], bg: str, accent: str) -> None:
    try:
        import matplotlib  # noqa: F401

        render_with_matplotlib(filename, title, lines, bg, accent)
    except ImportError:
        render_with_pillow(filename, title, lines, bg, accent)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for item in DIAGRAMS:
        render_diagram(*item)


if __name__ == "__main__":
    main()
