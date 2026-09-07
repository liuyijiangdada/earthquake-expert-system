#!/usr/bin/env python3
"""生成「图 5-2 问答数据流」超清流程图（PNG + SVG）。"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "superpowers" / "architecture"
PNG_PATH = OUT_DIR / "fig-5-2-query-flow.png"
SVG_PATH = OUT_DIR / "fig-5-2-query-flow.svg"
STATIC_PNG = ROOT / "static" / "fig-5-2-query-flow.png"
LEGACY_PNG = OUT_DIR / "fig-2-1-2-query-flow.png"
LEGACY_SVG = OUT_DIR / "fig-2-1-2-query-flow.svg"

W, H = 800, 860
SCALE = 5

COLORS = {
    "white": (255, 255, 255),
    "border": (226, 232, 240),
    "title": (15, 23, 42),
    "text": (30, 41, 59),
    "sub": (71, 85, 105),
    "arrow": (100, 116, 139),
    "node_fill": (235, 244, 255),
    "node_stroke": (43, 108, 176),
    "kg_fill": (230, 255, 250),
    "kg_stroke": (39, 103, 73),
    "rag_fill": (254, 252, 191),
    "rag_stroke": (183, 121, 31),
    "llm_fill": (250, 245, 255),
    "llm_stroke": (85, 60, 154),
}

TITLE = "问答请求数据流（query_type: llm）"

# (x1, y1, x2, y2, style, [(text, size_key), ...])
# size_key: lbl | sml | mono
NODES: list[tuple] = [
    (260, 56, 540, 100, "node", [("用户输入自然语言问题", "lbl")]),
    (220, 112, 580, 164, "node", [("POST /api/query", "lbl"), ("query_type = llm，params.input", "sml")]),
    (260, 178, 540, 218, "node", [("generate_response（app.py）", "lbl")]),
    (
        40,
        238,
        260,
        318,
        "kg",
        [
            ("知识图谱分支", "lbl"),
            ("省名 / 震级规则 / 应急语境", "sml"),
            ("Neo4j 查询得到图谱上下文", "sml"),
        ],
    ),
    (
        290,
        238,
        510,
        318,
        "rag",
        [
            ("向量检索分支", "lbl"),
            ("_build_rag_section", "mono"),
            ("余弦 Top-K 得到参考资料", "sml"),
        ],
    ),
    (
        540,
        238,
        760,
        318,
        "node",
        [("配置开关", "lbl"), ("KG_CONTEXT", "mono"), ("RAG_ENABLED", "mono")],
    ),
    (
        160,
        342,
        640,
        410,
        "node",
        [
            ("拼装 user 提示", "lbl"),
            ("【知识图谱】【参考资料】+ 规则 + 【问题】（问题在末尾）", "sml"),
            ("system / user 模板与微调格式一致", "sml"),
        ],
    ),
    (
        220,
        430,
        580,
        486,
        "llm",
        [
            ("tokenizer（truncation_side = left）", "lbl"),
            ("max_length = LLM_INPUT_MAX_TOKENS", "mono"),
        ],
    ),
    (
        220,
        500,
        580,
        556,
        "llm",
        [
            ("model.generate", "lbl"),
            ("temperature、top_p、repetition_penalty 等", "sml"),
        ],
    ),
    (240, 570, 560, 614, "node", [("decode 新生成 token 为文本", "lbl")]),
    (230, 630, 570, 674, "node", [("JSON { response } 返回前端", "lbl")]),
]

FOOTNOTES = [
    "另：query_type = kg 仅查 Neo4j，不经大模型（列表与筛选）",
    "POST /api/update-data：update_from_realtime_data() 刷新目录",
]


def load_font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        for path in (
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/Supplemental/Courier New.ttf",
        ):
            if Path(path).exists():
                return ImageFont.truetype(path, size=size)
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size, index=1 if bold else 0)
            except OSError:
                return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def style_colors(style: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if style == "kg":
        return COLORS["kg_fill"], COLORS["kg_stroke"]
    if style == "rag":
        return COLORS["rag_fill"], COLORS["rag_stroke"]
    if style == "llm":
        return COLORS["llm_fill"], COLORS["llm_stroke"]
    return COLORS["node_fill"], COLORS["node_stroke"]


def box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def draw_round_rect(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_arrow(draw, p1, p2, scale: int) -> None:
    color = COLORS["arrow"]
    draw.line([p1, p2], fill=color, width=2 * scale)
    ang = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    sz = 9 * scale
    bx, by = p2
    draw.polygon(
        [
            (bx, by),
            (bx - sz * math.cos(ang - 0.42), by - sz * math.sin(ang - 0.42)),
            (bx - sz * math.cos(ang + 0.42), by - sz * math.sin(ang + 0.42)),
        ],
        fill=color,
    )


def draw_polyline(draw, pts: list[tuple[float, float]], scale: int) -> None:
    for i in range(len(pts) - 1):
        draw_arrow(draw, pts[i], pts[i + 1], scale)


def line_layout(n: int, box: tuple[int, int, int, int]) -> list[float]:
    x1, y1, x2, y2 = box
    if n == 1:
        return [(y1 + y2) / 2]
    if n == 2:
        return [y1 + (y2 - y1) * 0.38, y1 + (y2 - y1) * 0.72]
    return [y1 + (y2 - y1) * 0.28, y1 + (y2 - y1) * 0.52, y1 + (y2 - y1) * 0.76]


def draw_node(
    draw,
    box: tuple[int, int, int, int],
    style: str,
    lines: list[tuple[str, str]],
    fonts: dict,
    scale: int,
) -> None:
    x1, y1, x2, y2 = box
    sbox = tuple(v * scale for v in box)
    fill, stroke = style_colors(style)
    draw_round_rect(draw, sbox, 10 * scale, fill, outline=stroke, width=2 * scale)

    ys = line_layout(len(lines), box)
    cx = (x1 + x2) / 2 * scale
    for (text, kind), ly in zip(lines, ys):
        font = fonts[kind]
        color = COLORS["text"] if kind == "lbl" else COLORS["sub"]
        draw.text((cx, ly * scale), text, fill=color, font=font, anchor="mm")


def render_frame(scale: int) -> Image.Image:
    sw, sh = W * scale, H * scale
    img = Image.new("RGB", (sw, sh), COLORS["white"])
    draw = ImageDraw.Draw(img)

    fonts = {
        "lbl": load_font(15 * scale, bold=True),
        "sml": load_font(12 * scale),
        "mono": load_font(12 * scale, mono=True),
    }

    m = 20 * scale
    draw_round_rect(draw, (m, m, sw - m, sh - m), 12 * scale, COLORS["white"], outline=COLORS["border"], width=scale)
    draw.text((sw // 2, 40 * scale), TITLE, fill=COLORS["title"], font=load_font(22 * scale, bold=True), anchor="mm")

    boxes = [n[:4] for n in NODES]
    for x1, y1, x2, y2, style, node_lines in NODES:
        draw_node(draw, (x1, y1, x2, y2), style, node_lines, fonts, scale)

    s = scale
    cx = W / 2 * s

    # 顶部链
    draw_polyline(
        draw,
        [(cx, boxes[0][3] * s), (cx, boxes[1][1] * s)],
        scale,
    )
    draw_polyline(draw, [(cx, boxes[1][3] * s), (cx, boxes[2][1] * s)], scale)
    split_y = (boxes[2][3] + 8) * s
    draw_polyline(
        draw,
        [
            (cx, boxes[2][3] * s),
            (cx, split_y),
            (150 * s, split_y),
            (150 * s, boxes[3][1] * s),
        ],
        scale,
    )
    draw_polyline(draw, [(cx, split_y), (cx, boxes[4][1] * s)], scale)
    draw_polyline(
        draw,
        [
            (cx, split_y),
            (650 * s, split_y),
            (650 * s, boxes[5][1] * s),
        ],
        scale,
    )

    join_y = (boxes[3][3] + 14) * s
    draw_polyline(
        draw,
        [
            (150 * s, boxes[3][3] * s),
            (150 * s, join_y),
            (cx, join_y),
            (cx, boxes[6][1] * s),
        ],
        scale,
    )
    draw_polyline(draw, [(cx, boxes[4][3] * s), (cx, join_y)], scale)
    draw_polyline(
        draw,
        [
            (650 * s, boxes[5][3] * s),
            (650 * s, join_y),
        ],
        scale,
    )

    for i in range(6, len(boxes) - 1):
        draw_polyline(
            draw,
            [(cx, boxes[i][3] * s), (cx, boxes[i + 1][1] * s)],
            scale,
        )

    foot_font = load_font(12 * scale)
    for i, note in enumerate(FOOTNOTES):
        draw.text((cx, (720 + i * 24) * s), note, fill=COLORS["sub"], font=foot_font, anchor="mm")

    return img


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _text_anchor_lines(box, lines, kind: str) -> str:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    ys = line_layout(len(lines), box)
    parts = []
    for (text, tk), ly in zip(lines, ys):
        ff = "Menlo,Consolas,monospace" if tk == "mono" else "PingFang SC,Microsoft YaHei,sans-serif"
        fw = "700" if tk == "lbl" else "400"
        fs = 15 if tk == "lbl" else 12
        fill = "#1e293b" if tk == "lbl" else "#475569"
        parts.append(
            f'<text x="{cx:.1f}" y="{ly + 4:.1f}" text-anchor="middle" font-family="{ff}" '
            f'font-size="{fs}" font-weight="{fw}" fill="{fill}">{_esc(text)}</text>'
        )
    return "\n".join(parts)


def generate_svg() -> Path:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<defs>",
        '<marker id="arr" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">'
        '<path d="M0,0 L10,4 L0,8 Z" fill="#64748b"/></marker>',
        "</defs>",
        f'<rect x="20" y="20" width="{W-40}" height="{H-40}" rx="12" fill="#fff" stroke="#e2e8f0"/>',
        f'<text x="{W/2:.1f}" y="40" text-anchor="middle" font-family="PingFang SC,Microsoft YaHei,sans-serif" '
        f'font-size="22" font-weight="700" fill="#0f172a">{_esc(TITLE)}</text>',
    ]

    boxes = []
    for x1, y1, x2, y2, style, node_lines in NODES:
        boxes.append((x1, y1, x2, y2))
        fill, stroke = style_colors(style)
        lines.append(
            f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" rx="10" '
            f'fill="{_hex(fill)}" stroke="{_hex(stroke)}" stroke-width="2"/>'
        )
        lines.append(_text_anchor_lines((x1, y1, x2, y2), node_lines, style))

    arrow = 'stroke="#64748b" stroke-width="2" fill="none" marker-end="url(#arr)"'
    cx = W / 2
    split_y = boxes[2][3] + 8
    join_y = boxes[3][3] + 14
    lines.extend(
        [
            f'<path d="M {cx} {boxes[0][3]} L {cx} {boxes[1][1]}" {arrow}/>',
            f'<path d="M {cx} {boxes[1][3]} L {cx} {boxes[2][1]}" {arrow}/>',
            f'<path d="M {cx} {boxes[2][3]} L {cx} {split_y} L 150 {split_y} L 150 {boxes[3][1]}" {arrow}/>',
            f'<path d="M {cx} {split_y} L {cx} {boxes[4][1]}" {arrow}/>',
            f'<path d="M {cx} {split_y} L 650 {split_y} L 650 {boxes[5][1]}" {arrow}/>',
            f'<path d="M 150 {boxes[3][3]} L 150 {join_y} L {cx} {join_y} L {cx} {boxes[6][1]}" {arrow}/>',
            f'<path d="M {cx} {boxes[4][3]} L {cx} {join_y}" {arrow}/>',
            f'<path d="M 650 {boxes[5][3]} L 650 {join_y}" {arrow}/>',
        ]
    )
    for i in range(6, len(boxes) - 1):
        lines.append(f'<path d="M {cx} {boxes[i][3]} L {cx} {boxes[i+1][1]}" {arrow}/>')

    for i, note in enumerate(FOOTNOTES):
        lines.append(
            f'<text x="{cx:.1f}" y="{720 + i * 24}" text-anchor="middle" '
            f'font-family="PingFang SC,Microsoft YaHei,sans-serif" font-size="12" fill="#64748b">{_esc(note)}</text>'
        )
    lines.append("</svg>")
    text = "\n".join(lines)
    SVG_PATH.write_text(text, encoding="utf-8")
    LEGACY_SVG.write_text(text, encoding="utf-8")
    return SVG_PATH


def generate_png() -> Path:
    img = render_frame(SCALE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(PNG_PATH, format="PNG", dpi=(300, 300), optimize=True)
    img.save(LEGACY_PNG, format="PNG", dpi=(300, 300), optimize=True)
    STATIC_PNG.parent.mkdir(parents=True, exist_ok=True)
    img.save(STATIC_PNG, format="PNG", dpi=(300, 300), optimize=True)
    return PNG_PATH


def main() -> None:
    png = generate_png()
    svg = generate_svg()
    print(f"PNG: {png}  ({png.stat().st_size // 1024} KB, {W * SCALE}×{H * SCALE}px)")
    print(f"SVG: {svg}")
    print(f"Static copy: {STATIC_PNG}")


if __name__ == "__main__":
    main()
