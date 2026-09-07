#!/usr/bin/env python3
"""生成「图 5-3 技术栈与模块映射」超清表格图（PNG + SVG）。"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "superpowers" / "architecture"
PNG_PATH = OUT_DIR / "fig-5-3-tech-stack.png"
SVG_PATH = OUT_DIR / "fig-5-3-tech-stack.svg"
STATIC_PNG = ROOT / "static" / "fig-5-3-tech-stack.png"
LEGACY_PNG = OUT_DIR / "fig-2-1-3-tech-stack.png"
LEGACY_SVG = OUT_DIR / "fig-2-1-3-tech-stack.svg"

W, H = 960, 420
SCALE = 5

MARGIN = 40
TABLE_X = MARGIN
TABLE_W = W - 2 * MARGIN
COL_W = (160, 340, TABLE_W - 160 - 340)  # 模块 | 代码/技术 | 职责
HDR_H, ROW_H = 44, 48

COLORS = {
    "white": (255, 255, 255),
    "border": (226, 232, 240),
    "grid": (226, 232, 240),
    "title": (15, 23, 42),
    "hdr_bg": (44, 82, 130),
    "hdr_text": (255, 255, 255),
    "row_odd": (247, 250, 252),
    "row_even": (255, 255, 255),
    "cell": (30, 41, 59),
    "code": (30, 58, 138),
}

ROWS = [
    ("模块", "主要代码 / 技术", "职责"),
    ("前端", "static/index.html", "对话、地震列表、筛选、触发数据更新"),
    ("应用网关", "app.py（Flask）", "路由、组装 KG/RAG 上下文、调用 LLM"),
    ("知识图谱", "kg/neo4j_kg.py + Neo4j", "地震事件、区域、应急主题与处置步骤"),
    ("向量检索", "rag/emergency_rag.py", "应急知识 JSON 分块、嵌入、余弦 Top-K"),
    ("大模型", "transformers + peft（LoRA）", "本地基座权重 + 微调适配器推理"),
    ("配置", "config/config.py", "Neo4j、路径、RAG/LLM 开关与解码参数"),
]

TITLE = "技术栈与模块映射"
TABLE_TOP = 72


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


def has_cjk(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text)


def pick_code_font(text: str, fonts: dict) -> ImageFont.FreeTypeFont:
    """纯 ASCII 用等宽；含中文/全角符号时用 PingFang，避免括号显示为方框。"""
    return fonts["code_cjk"] if has_cjk(text) else fonts["code_mono"]


def col_x(col: int) -> int:
    return TABLE_X + sum(COL_W[:col])


def draw_round_rect(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def table_height() -> int:
    return HDR_H + (len(ROWS) - 1) * ROW_H


def render_frame(scale: int) -> Image.Image:
    sw, sh = W * scale, H * scale
    img = Image.new("RGB", (sw, sh), COLORS["white"])
    draw = ImageDraw.Draw(img)

    fonts = {
        "title": load_font(22 * scale, bold=True),
        "hdr": load_font(15 * scale, bold=True),
        "cell": load_font(14 * scale),
        "cell_b": load_font(14 * scale, bold=True),
        "code_mono": load_font(13 * scale, mono=True),
        "code_cjk": load_font(13 * scale),
    }

    m = 20 * scale
    draw_round_rect(draw, (m, m, sw - m, sh - m), 12 * scale, COLORS["white"], outline=COLORS["border"], width=scale)

    draw.text((sw // 2, 46 * scale), TITLE, fill=COLORS["title"], font=fonts["title"], anchor="mm")

    tx, ty = TABLE_X * scale, TABLE_TOP * scale
    tw, th = TABLE_W * scale, table_height() * scale
    draw_round_rect(draw, (tx, ty, tx + tw, ty + th), 8 * scale, COLORS["white"], outline=COLORS["grid"], width=scale)

    # 表头
    hx, hy = tx, ty
    draw.rectangle((hx, hy, hx + tw, hy + HDR_H * scale), fill=COLORS["hdr_bg"])
    for ci, label in enumerate(ROWS[0]):
        cx = col_x(ci) * scale
        cw = COL_W[ci] * scale
        draw.text(
            (cx + cw / 2, hy + HDR_H * scale / 2),
            label,
            fill=COLORS["hdr_text"],
            font=fonts["hdr"],
            anchor="mm",
        )

    # 数据行
    y = ty + HDR_H * scale
    for ri, (mod, code, duty) in enumerate(ROWS[1:], start=1):
        bg = COLORS["row_odd"] if ri % 2 == 1 else COLORS["row_even"]
        draw.rectangle((hx, y, hx + tw, y + ROW_H * scale), fill=bg)

        # 列分隔线
        for ci in range(1, 3):
            lx = col_x(ci) * scale
            draw.line([(lx, y), (lx, y + ROW_H * scale)], fill=COLORS["grid"], width=scale)

        cy = y + ROW_H * scale / 2
        draw.text(
            (col_x(0) * scale + COL_W[0] * scale / 2, cy),
            mod,
            fill=COLORS["cell"],
            font=fonts["cell_b"],
            anchor="mm",
        )
        draw.text(
            (col_x(1) * scale + COL_W[1] * scale / 2, cy),
            code,
            fill=COLORS["code"],
            font=pick_code_font(code, fonts),
            anchor="mm",
        )
        draw.text(
            (col_x(2) * scale + 16 * scale, cy),
            duty,
            fill=COLORS["cell"],
            font=fonts["cell"],
            anchor="lm",
        )
        y += ROW_H * scale

    # 外框与行线
    draw.line([(hx, ty + HDR_H * scale), (hx + tw, ty + HDR_H * scale)], fill=COLORS["grid"], width=2 * scale)
    for ri in range(1, len(ROWS)):
        ly = ty + (HDR_H + ri * ROW_H) * scale
        draw.line([(hx, ly), (hx + tw, ly)], fill=COLORS["grid"], width=scale)

    return img


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def generate_svg() -> Path:
    th = table_height()
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect x="20" y="20" width="{W-40}" height="{H-40}" rx="12" fill="#fff" stroke="#e2e8f0"/>',
        f'<text x="{W/2:.1f}" y="46" text-anchor="middle" font-family="PingFang SC,Microsoft YaHei,sans-serif" '
        f'font-size="22" font-weight="700" fill="#0f172a">{_esc(TITLE)}</text>',
        f'<rect x="{TABLE_X}" y="{TABLE_TOP}" width="{TABLE_W}" height="{th}" rx="8" fill="#fff" stroke="#e2e8f0"/>',
        f'<rect x="{TABLE_X}" y="{TABLE_TOP}" width="{TABLE_W}" height="{HDR_H}" fill="{_hex(COLORS["hdr_bg"])}"/>',
    ]

    for ci, label in enumerate(ROWS[0]):
        cx = col_x(ci) + COL_W[ci] / 2
        lines.append(
            f'<text x="{cx:.1f}" y="{TABLE_TOP + HDR_H/2 + 5:.1f}" text-anchor="middle" '
            f'font-family="PingFang SC,Microsoft YaHei,sans-serif" font-size="15" font-weight="700" fill="#fff">'
            f"{_esc(label)}</text>"
        )

    for ri, (mod, code, duty) in enumerate(ROWS[1:], start=1):
        y = TABLE_TOP + HDR_H + (ri - 1) * ROW_H
        bg = _hex(COLORS["row_odd"] if ri % 2 == 1 else COLORS["row_even"])
        lines.append(f'<rect x="{TABLE_X}" y="{y}" width="{TABLE_W}" height="{ROW_H}" fill="{bg}"/>')
        for ci in range(1, 3):
            lx = col_x(ci)
            lines.append(f'<line x1="{lx}" y1="{y}" x2="{lx}" y2="{y + ROW_H}" stroke="#e2e8f0"/>')
        cy = y + ROW_H / 2 + 5
        lines.append(
            f'<text x="{col_x(0) + COL_W[0]/2:.1f}" y="{cy:.1f}" text-anchor="middle" '
            f'font-family="PingFang SC,Microsoft YaHei,sans-serif" font-size="14" font-weight="700" fill="#1e293b">'
            f"{_esc(mod)}</text>"
        )
        code_ff = (
            "PingFang SC,Microsoft YaHei,sans-serif"
            if has_cjk(code)
            else "Menlo,Consolas,monospace"
        )
        lines.append(
            f'<text x="{col_x(1) + COL_W[1]/2:.1f}" y="{cy:.1f}" text-anchor="middle" '
            f'font-family="{code_ff}" font-size="13" fill="#1e3a8a">{_esc(code)}</text>'
        )
        lines.append(
            f'<text x="{col_x(2) + 16:.1f}" y="{cy:.1f}" '
            f'font-family="PingFang SC,Microsoft YaHei,sans-serif" font-size="14" fill="#1e293b">{_esc(duty)}</text>'
        )

    ty = TABLE_TOP
    lines.append(f'<line x1="{TABLE_X}" y1="{ty + HDR_H}" x2="{TABLE_X + TABLE_W}" y2="{ty + HDR_H}" stroke="#e2e8f0" stroke-width="2"/>')
    for ri in range(1, len(ROWS)):
        ly = TABLE_TOP + HDR_H + ri * ROW_H
        lines.append(f'<line x1="{TABLE_X}" y1="{ly}" x2="{TABLE_X + TABLE_W}" y2="{ly}" stroke="#e2e8f0"/>')

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
