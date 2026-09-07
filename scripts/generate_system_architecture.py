#!/usr/bin/env python3
"""生成「图 5-1 系统总体逻辑架构」超清架构图（PNG + SVG，与当前代码一致）。"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "superpowers" / "architecture"
PNG_PATH = OUT_DIR / "fig-5-1-system-architecture.png"
SVG_PATH = OUT_DIR / "fig-5-1-system-architecture.svg"
STATIC_PNG = ROOT / "static" / "fig-5-1-system-architecture.png"
# 兼容旧文件名
LEGACY_PNG = OUT_DIR / "fig-2-1-1-system-architecture.png"
LEGACY_SVG = OUT_DIR / "fig-2-1-1-system-architecture.svg"

W, H = 1080, 640
SCALE = 5
CX = W / 2

MARGIN_X = 40
COMP_Y1, COMP_Y2 = 286, 424
COMP_W = (W - 2 * MARGIN_X - 3 * 16) // 4  # 238
COMP_GAP = 16


def _comp_box(index: int) -> tuple[int, int, int, int]:
    x1 = MARGIN_X + index * (COMP_W + COMP_GAP)
    return x1, COMP_Y1, x1 + COMP_W, COMP_Y2

COLORS = {
    "white": (255, 255, 255),
    "border": (226, 232, 240),
    "title": (15, 23, 42),
    "text": (30, 41, 59),
    "sub": (71, 85, 105),
    "arrow": (100, 116, 139),
    "dash": (148, 163, 184),
    "node_fill": (235, 244, 255),
    "node_stroke": (43, 108, 176),
    "kg_fill": (230, 255, 250),
    "kg_stroke": (39, 103, 73),
    "rag_fill": (254, 252, 191),
    "rag_stroke": (183, 121, 31),
    "dyn_fill": (224, 242, 254),
    "dyn_stroke": (2, 132, 199),
    "llm_fill": (250, 245, 255),
    "llm_stroke": (85, 60, 154),
    "cfg_fill": (255, 247, 237),
    "cfg_stroke": (194, 65, 12),
}

TITLE = "系统总体逻辑架构"

# (x1, y1, x2, y2, style, lines[(text, kind)])
# kind: lbl | sml | mono
LAYERS: list[tuple] = [
    (
        300,
        58,
        780,
        132,
        "node",
        [
            ("展示层：Vue 3 SPA + Android 客户端", "lbl"),
            ("frontend/ → static/spa/ · disaster_app/android/", "mono"),
            ("对话 · 地震列表 · 图文问答 · 阶段标签 · 数据更新", "sml"),
        ],
    ),
    (
        130,
        158,
        950,
        262,
        "node",
        [
            ("应用服务层：Flask（app.py）", "lbl"),
            ("POST /api/query · /api/multimodal-query · /api/update-data · /api/eval-set", "mono"),
            ("core/：PhaseClassifier · Scheduler → context_builder 组装提示", "sml"),
            ("response_guard 质量兜底 · output_enricher 第三层增强", "sml"),
        ],
    ),
    (
        *_comp_box(0),
        "kg",
        [
            ("Neo4j 知识图谱", "lbl"),
            ("kg/neo4j_kg.py", "mono"),
            ("地震事件 · 区域 · 13 应急主题", "sml"),
        ],
    ),
    (
        *_comp_box(1),
        "rag",
        [
            ("RAG 向量检索", "lbl"),
            ("rag/emergency_rag.py", "mono"),
            ("BGE · topic 分块 · Milvus/内存 Top-K", "sml"),
        ],
    ),
    (
        *_comp_box(2),
        "dyn",
        [
            ("动态震情", "lbl"),
            ("core/earthquake_feed.py", "mono"),
            ("CEIC + USGS 统一目录 · TTL 缓存", "sml"),
        ],
    ),
    (
        *_comp_box(3),
        "llm",
        [
            ("本地大模型", "lbl"),
            ("transformers + peft（LoRA）", "sml"),
            ("Qwen1.5-1.8B + LoRA", "sml"),
            ("Qwen2-VL 懒加载", "sml"),
        ],
    ),
    (
        360,
        456,
        720,
        512,
        "cfg",
        [
            ("config/config.py", "lbl"),
            ("Neo4j · RAG/LLM · 调度与动态源开关 · 解码参数", "sml"),
        ],
    ),
]

COMPONENT_IDX = (2, 3, 4, 5)
APP_BOX = LAYERS[1][:4]
PRES_BOX = LAYERS[0][:4]
CFG_BOX = LAYERS[6][:4]


def load_font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        for path in (
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/Supplemental/Courier New.ttf",
        ):
            if Path(path).exists():
                return ImageFont.truetype(path, size=size)
    for path in (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
    ):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size, index=1 if bold else 0)
            except OSError:
                return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def has_cjk(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text)


def pick_font(text: str, kind: str, fonts: dict) -> ImageFont.FreeTypeFont:
    if kind == "mono" and not has_cjk(text):
        return fonts["mono"]
    if kind == "lbl":
        return fonts["lbl"]
    return fonts["sml"]


def style_colors(style: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    key = style if style != "cfg" else "cfg"
    if key == "cfg":
        return COLORS["cfg_fill"], COLORS["cfg_stroke"]
    if key == "kg":
        return COLORS["kg_fill"], COLORS["kg_stroke"]
    if key == "rag":
        return COLORS["rag_fill"], COLORS["rag_stroke"]
    if key == "dyn":
        return COLORS["dyn_fill"], COLORS["dyn_stroke"]
    if key == "llm":
        return COLORS["llm_fill"], COLORS["llm_stroke"]
    return COLORS["node_fill"], COLORS["node_stroke"]


def box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def line_layout(n: int, box: tuple[int, int, int, int]) -> list[float]:
    x1, y1, x2, y2 = box
    if n == 1:
        return [(y1 + y2) / 2]
    if n == 2:
        return [y1 + (y2 - y1) * 0.38, y1 + (y2 - y1) * 0.68]
    if n == 3:
        return [y1 + (y2 - y1) * 0.30, y1 + (y2 - y1) * 0.52, y1 + (y2 - y1) * 0.74]
    if n == 4:
        return [y1 + (y2 - y1) * 0.22, y1 + (y2 - y1) * 0.40, y1 + (y2 - y1) * 0.58, y1 + (y2 - y1) * 0.76]


def draw_round_rect(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_arrow(draw, p1, p2, scale: int, dashed: bool = False) -> None:
    color = COLORS["dash"] if dashed else COLORS["arrow"]
    if dashed:
        ax, ay = p1
        bx, by = p2
        dist = math.hypot(bx - ax, by - ay)
        if dist < 1:
            return
        ux, uy = (bx - ax) / dist, (by - ay) / dist
        pos, on = 0.0, True
        while pos < dist:
            seg = min((8 * scale if on else 6 * scale), dist - pos)
            if on:
                draw.line(
                    [(ax + ux * pos, ay + uy * pos), (ax + ux * (pos + seg), ay + uy * (pos + seg))],
                    fill=color,
                    width=max(1, scale),
                )
            pos += seg
            on = not on
    else:
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


def draw_polyline(draw, pts: list[tuple[float, float]], scale: int, dashed: bool = False) -> None:
    for i in range(len(pts) - 1):
        draw_arrow(draw, pts[i], pts[i + 1], scale, dashed=dashed)


def draw_layer_box(
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
        font = pick_font(text, kind, fonts)
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
    draw.text((sw // 2, 38 * scale), TITLE, fill=COLORS["title"], font=load_font(22 * scale, bold=True), anchor="mm")

    boxes = [layer[:4] for layer in LAYERS]
    for x1, y1, x2, y2, style, lines in LAYERS:
        draw_layer_box(draw, (x1, y1, x2, y2), style, lines, fonts, scale)

    s = scale
    cx = CX * s
    pres, app, cfg = PRES_BOX, APP_BOX, CFG_BOX

    draw_polyline(draw, [(cx, pres[3] * s), (cx, (pres[3] + 10) * s), (cx, app[1] * s)], scale)
    draw.text(((cx + 12 * s), (pres[3] + 22) * s), "HTTP / JSON", fill=COLORS["sub"], font=fonts["sml"], anchor="lm")

    split_y = (app[3] + 12) * s
    comp_boxes = [boxes[i] for i in COMPONENT_IDX]

    draw_polyline(draw, [(cx, app[3] * s), (cx, split_y)], scale)
    for b in comp_boxes:
        bx = box_center(b)[0] * s
        draw_polyline(draw, [(cx, split_y), (bx, split_y), (bx, b[1] * s)], scale)

    cfg_cx, cfg_top = box_center(cfg)[0] * s, cfg[1] * s
    for b in comp_boxes:
        comp_cx, _ = box_center(b)
        draw_polyline(
            draw,
            [
                (comp_cx * s, b[3] * s),
                (comp_cx * s, (b[3] + 10) * s),
                (cfg_cx, (b[3] + 10) * s),
                (cfg_cx, cfg_top),
            ],
            scale,
            dashed=True,
        )

    return img


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _svg_text_block(box, lines: list[tuple[str, str]]) -> str:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    ys = line_layout(len(lines), box)
    parts = []
    for (text, kind), ly in zip(lines, ys):
        if kind == "mono" and not has_cjk(text):
            ff = "Menlo,Consolas,monospace"
        else:
            ff = "PingFang SC,Microsoft YaHei,sans-serif"
        fw = "700" if kind == "lbl" else "400"
        fs = 15 if kind == "lbl" else 12
        fill = "#1e293b" if kind == "lbl" else "#475569"
        parts.append(
            f'<text x="{cx:.1f}" y="{ly + 4:.1f}" text-anchor="middle" font-family="{ff}" '
            f'font-size="{fs}" font-weight="{fw}" fill="{fill}">{_esc(text)}</text>'
        )
    return "\n".join(parts)


def generate_svg() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<defs>",
        '<marker id="arr" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">'
        '<path d="M0,0 L10,4 L0,8 Z" fill="#64748b"/></marker>',
        '<marker id="arrd" markerWidth="8" markerHeight="7" refX="7" refY="3.5" orient="auto">'
        '<path d="M0,0 L8,3.5 L0,7 Z" fill="#94a3b8"/></marker>',
        "</defs>",
        f'<rect x="20" y="20" width="{W-40}" height="{H-40}" rx="12" fill="#fff" stroke="#e2e8f0"/>',
        f'<text x="{CX:.1f}" y="38" text-anchor="middle" font-family="PingFang SC,Microsoft YaHei,sans-serif" '
        f'font-size="22" font-weight="700" fill="#0f172a">{_esc(TITLE)}</text>',
    ]

    boxes = []
    for x1, y1, x2, y2, style, node_lines in LAYERS:
        boxes.append((x1, y1, x2, y2))
        fill, stroke = style_colors(style)
        lines.append(
            f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" rx="10" '
            f'fill="{_hex(fill)}" stroke="{_hex(stroke)}" stroke-width="2"/>'
        )
        lines.append(_svg_text_block((x1, y1, x2, y2), node_lines))

    pres, app, cfg = boxes[0], boxes[1], boxes[6]
    comp_boxes = [boxes[i] for i in COMPONENT_IDX]
    split_y = app[3] + 12
    solid = 'stroke="#64748b" stroke-width="2" fill="none" marker-end="url(#arr)"'
    dashed = 'stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="6 4" fill="none" marker-end="url(#arrd)"'

    lines.append(f'<path d="M {CX} {pres[3]} L {CX} {app[1]}" {solid}/>')
    lines.append(
        f'<text x="{CX+12:.1f}" y="{pres[3]+22}" font-family="PingFang SC,Microsoft YaHei,sans-serif" '
        f'font-size="12" fill="#64748b">HTTP / JSON</text>'
    )
    lines.append(f'<path d="M {CX} {app[3]} L {CX} {split_y}" {solid}/>')
    for b in comp_boxes:
        bx = (b[0] + b[2]) / 2
        lines.append(f'<path d="M {CX} {split_y} L {bx} {split_y} L {bx} {b[1]}" {solid}/>')

    cfg_cx = (cfg[0] + cfg[2]) / 2
    merge_y = comp_boxes[0][3] + 10
    for b in comp_boxes:
        bx = (b[0] + b[2]) / 2
        lines.append(f'<path d="M {bx} {b[3]} L {bx} {merge_y} L {cfg_cx} {merge_y} L {cfg_cx} {cfg[1]}" {dashed}/>')

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
