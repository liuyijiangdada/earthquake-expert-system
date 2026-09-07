#!/usr/bin/env python3
"""生成「RAG 分块与向量化流程」超清横向流程图（PNG + SVG）。"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "superpowers" / "architecture"
PNG_PATH = OUT_DIR / "fig-4-2-rag-pipeline.png"
SVG_PATH = OUT_DIR / "fig-4-2-rag-pipeline.svg"
STATIC_PNG = ROOT / "static" / "fig-4-2-rag-pipeline.png"
LEGACY_PNG = OUT_DIR / "fig-5-1-rag-pipeline.png"

W, H = 1280, 340
SCALE = 4  # 5120×1360 物理像素，超清输出

BOX = 132
GAP = 36
MARGIN_X = 56

COLORS = {
    "white": (255, 255, 255),
    "border": (226, 232, 240),
    "box_stroke": (51, 65, 85),
    "title": (15, 23, 42),
    "sub": (100, 116, 139),
    "arrow": (71, 85, 105),
}

STEPS = [
    {"lines": ["JSON", "文件"], "fill": (96, 165, 250)},
    {"lines": ["Topic 级", "分块"], "fill": (20, 184, 166)},
    {"lines": ["Sentence-", "Transformer", "编码"], "fill": (168, 85, 247)},
    {"lines": ["向量", "矩阵"], "fill": (249, 115, 22)},
    {"lines": ["相似度", "计算"], "fill": (234, 88, 12)},
    {"lines": ["Top-K", "选择"], "fill": (34, 197, 94)},
]

ROW_CY = 210
TITLE_Y = 48
SUB_Y = 76


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size, index=1 if bold else 0)
            except OSError:
                return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def step_centers() -> list[float]:
    n = len(STEPS)
    total = n * BOX + (n - 1) * GAP
    x0 = (W - total) / 2 + BOX / 2
    return [x0 + i * (BOX + GAP) for i in range(n)]


def box_rect(cx: float) -> tuple[int, int, int, int]:
    return int(cx - BOX / 2), int(ROW_CY - BOX / 2), int(cx + BOX / 2), int(ROW_CY + BOX / 2)


def draw_round_rect(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_arrow(draw, p1: tuple[float, float], p2: tuple[float, float], scale: int) -> None:
    color = COLORS["arrow"]
    draw.line([p1, p2], fill=color, width=2 * scale)
    ang = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    sz = 10 * scale
    bx, by = p2
    draw.polygon(
        [
            (bx, by),
            (bx - sz * math.cos(ang - 0.42), by - sz * math.sin(ang - 0.42)),
            (bx - sz * math.cos(ang + 0.42), by - sz * math.sin(ang + 0.42)),
        ],
        fill=color,
    )


def draw_step_box(
    draw,
    cx: float,
    step: dict,
    font,
    scale: int,
) -> tuple[int, int, int, int]:
    box = box_rect(cx)
    sbox = tuple(v * scale for v in box)
    fill = step["fill"]
    draw_round_rect(
        draw,
        sbox,
        14 * scale,
        fill,
        outline=COLORS["box_stroke"],
        width=2 * scale,
    )
    lines = step["lines"]
    line_h = 20 * scale
    total_h = line_h * len(lines)
    y0 = ROW_CY * scale - total_h / 2 + line_h / 2
    for i, line in enumerate(lines):
        draw.text(
            (cx * scale, y0 + i * line_h),
            line,
            fill=COLORS["white"],
            font=font,
            anchor="mm",
        )
    return box


def render_frame(scale: int) -> Image.Image:
    sw, sh = W * scale, H * scale
    img = Image.new("RGB", (sw, sh), COLORS["white"])
    draw = ImageDraw.Draw(img)

    fonts = {
        "title": load_font(26 * scale, bold=True),
        "sub": load_font(14 * scale),
        "node": load_font(15 * scale, bold=True),
    }

    m = 22 * scale
    draw_round_rect(draw, (m, m, sw - m, sh - m), 14 * scale, COLORS["white"], outline=COLORS["border"], width=scale)

    draw.text((sw // 2, TITLE_Y * scale), "主题级向量检索流程", fill=COLORS["title"], font=fonts["title"], anchor="mm")
    draw.text(
        (sw // 2, SUB_Y * scale),
        "应急知识 JSON · Topic 级分块 · BGE 句向量 · 余弦 Top-K",
        fill=COLORS["sub"],
        font=fonts["sub"],
        anchor="mm",
    )

    centers = step_centers()
    boxes = [draw_step_box(draw, cx, step, fonts["node"], scale) for cx, step in zip(centers, STEPS)]

    for i in range(len(boxes) - 1):
        y = ROW_CY * scale
        x1 = boxes[i][2] * scale + 6 * scale
        x2 = boxes[i + 1][0] * scale - 6 * scale
        draw_arrow(draw, (x1, y), (x2, y), scale)

    return img


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def generate_svg() -> Path:
    centers = step_centers()
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<defs>",
        '<marker id="arr" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">'
        '<path d="M0,0 L10,4 L0,8 Z" fill="#475569"/></marker>',
        "</defs>",
        f'<rect x="22" y="22" width="{W-44}" height="{H-44}" rx="14" fill="#fff" stroke="#e2e8f0"/>',
        f'<text x="{W/2:.1f}" y="{TITLE_Y}" text-anchor="middle" '
        f'font-family="PingFang SC,Microsoft YaHei,sans-serif" font-size="26" font-weight="700" fill="#0f172a">'
        "主题级向量检索流程</text>",
        f'<text x="{W/2:.1f}" y="{SUB_Y}" text-anchor="middle" '
        f'font-family="PingFang SC,Microsoft YaHei,sans-serif" font-size="14" fill="#64748b">'
        "应急知识 JSON · Topic 级分块 · BGE 句向量 · 余弦 Top-K</text>",
    ]

    arrow = 'stroke="#475569" stroke-width="2" fill="none" marker-end="url(#arr)"'
    for cx, step in zip(centers, STEPS):
        x, y = cx - BOX / 2, ROW_CY - BOX / 2
        fill = _hex(step["fill"])
        lines.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{BOX}" height="{BOX}" rx="14" '
            f'fill="{fill}" stroke="#334155" stroke-width="2"/>'
        )
        n = len(step["lines"])
        line_h = 20
        y0 = ROW_CY - (n - 1) * line_h / 2
        for i, text in enumerate(step["lines"]):
            esc = text.replace("&", "&amp;")
            lines.append(
                f'<text x="{cx:.1f}" y="{y0 + i * line_h:.1f}" text-anchor="middle" '
                f'font-family="PingFang SC,Microsoft YaHei,sans-serif" font-size="15" font-weight="700" fill="#fff">{esc}</text>'
            )

    for i in range(len(centers) - 1):
        x1 = centers[i] + BOX / 2 + 6
        x2 = centers[i + 1] - BOX / 2 - 6
        lines.append(f'<path d="M {x1:.1f} {ROW_CY} L {x2:.1f} {ROW_CY}" {arrow}/>')

    lines.append("</svg>")
    SVG_PATH.write_text("\n".join(lines), encoding="utf-8")
    return SVG_PATH


def generate_png() -> Path:
    img = render_frame(SCALE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(PNG_PATH, format="PNG", dpi=(300, 300), optimize=True)
    STATIC_PNG.parent.mkdir(parents=True, exist_ok=True)
    img.save(STATIC_PNG, format="PNG", dpi=(300, 300), optimize=True)
    if LEGACY_PNG.parent.exists():
        img.save(LEGACY_PNG, format="PNG", dpi=(300, 300), optimize=True)
    return PNG_PATH


def main() -> None:
    png = generate_png()
    svg = generate_svg()
    print(f"PNG: {png}  ({png.stat().st_size // 1024} KB, {W * SCALE}×{H * SCALE}px)")
    print(f"SVG: {svg}")
    print(f"Static copy: {STATIC_PNG}")


if __name__ == "__main__":
    main()
