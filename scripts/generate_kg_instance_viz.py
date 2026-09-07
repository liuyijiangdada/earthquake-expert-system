#!/usr/bin/env python3
"""生成全中文知识图谱实例子图（高清 PNG + 矢量 SVG）。"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "superpowers" / "architecture"
PNG_PATH = OUT_DIR / "fig-4-2-kg-instance.png"
SVG_PATH = OUT_DIR / "fig-4-2-kg-instance.svg"

# 逻辑画布尺寸（论文常用 3:2 比例）
W, H = 1200, 800
SCALE = 3  # 超采样倍率，输出 3600×2400 高清图

TYPE_COLORS = {
    "地震": {"fill": (219, 234, 254), "stroke": (37, 99, 235), "text": (30, 58, 138), "hdr": (37, 99, 235)},
    "区域": {"fill": (204, 251, 241), "stroke": (13, 148, 136), "text": (15, 94, 89), "hdr": (13, 148, 136)},
    "主题": {"fill": (237, 233, 254), "stroke": (124, 58, 237), "text": (76, 29, 149), "hdr": (124, 58, 237)},
    "步骤": {"fill": (255, 237, 213), "stroke": (234, 88, 12), "text": (154, 52, 18), "hdr": (234, 88, 12)},
}

# 节点：中心坐标 + 宽高 + 单行或双行标题 + 可选副标题
NODES: dict[str, dict] = {
    "eq_wc": {"title": "汶川特大地震", "sub": "8.0 级", "type": "地震", "cx": 170, "cy": 175, "w": 168, "h": 68},
    "eq_ls": {"title": "芦山地震", "sub": "7.0 级", "type": "地震", "cx": 170, "cy": 315, "w": 148, "h": 64},
    "eq_ld": {"title": "鲁甸地震", "sub": "6.5 级", "type": "地震", "cx": 170, "cy": 455, "w": 148, "h": 64},
    "rg_sc": {"title": "四川", "sub": "", "type": "区域", "cx": 420, "cy": 245, "w": 112, "h": 56},
    "rg_yn": {"title": "云南", "sub": "", "type": "区域", "cx": 420, "cy": 455, "w": 112, "h": 56},
    "tp_indoor": {"title": "室内避险", "sub": "震中", "type": "主题", "cx": 670, "cy": 155, "w": 132, "h": 60},
    "tp_after": {"title": "余震防范", "sub": "震后", "type": "主题", "cx": 670, "cy": 315, "w": 132, "h": 60},
    "tp_kit": {"title": "家庭应急准备", "sub": "震前", "type": "主题", "cx": 670, "cy": 475, "w": 148, "h": 60},
    "st1": {"title": "伏地遮挡护头", "sub": "", "type": "步骤", "cx": 960, "cy": 115, "w": 148, "h": 52},
    "st2": {"title": "远离门窗吊灯", "sub": "", "type": "步骤", "cx": 960, "cy": 195, "w": 148, "h": 52},
    "st3": {"title": "勿返危楼取物", "sub": "", "type": "步骤", "cx": 960, "cy": 315, "w": 148, "h": 52},
    "st4": {"title": "准备应急包", "sub": "", "type": "步骤", "cx": 960, "cy": 445, "w": 132, "h": 52},
    "st5": {"title": "定期防震演练", "sub": "", "type": "步骤", "cx": 960, "cy": 525, "w": 148, "h": 52},
}

COLUMNS = [
    (170, "地震事件", TYPE_COLORS["地震"]["hdr"]),
    (420, "行政区域", TYPE_COLORS["区域"]["hdr"]),
    (670, "应急主题", TYPE_COLORS["主题"]["hdr"]),
    (960, "处置步骤", TYPE_COLORS["步骤"]["hdr"]),
]

EDGES = [
    ("eq_wc", "rg_sc", "发生于", (13, 148, 136), False, 0.0),
    ("eq_ls", "rg_sc", "发生于", (13, 148, 136), False, 0.0),
    ("eq_ld", "rg_yn", "发生于", (13, 148, 136), False, 0.0),
    ("eq_wc", "tp_after", "建议主题", (124, 58, 237), True, -0.12),
    ("eq_wc", "tp_kit", "建议主题", (124, 58, 237), True, 0.18),
    ("eq_ls", "tp_indoor", "建议主题", (124, 58, 237), True, 0.0),
    ("eq_ld", "tp_after", "建议主题", (124, 58, 237), True, 0.15),
    ("tp_indoor", "st1", "包含步骤", (234, 88, 12), False, -0.08),
    ("tp_indoor", "st2", "包含步骤", (234, 88, 12), False, 0.08),
    ("tp_after", "st3", "包含步骤", (234, 88, 12), False, 0.0),
    ("tp_kit", "st4", "包含步骤", (234, 88, 12), False, -0.06),
    ("tp_kit", "st5", "包含步骤", (234, 88, 12), False, 0.06),
    ("tp_indoor", "tp_after", "关联主题", (124, 58, 237), True, 0.35),
    ("tp_kit", "tp_indoor", "关联主题", (124, 58, 237), True, -0.25),
]


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


def node_box(nid: str) -> tuple[float, float, float, float]:
    n = NODES[nid]
    x1 = n["cx"] - n["w"] / 2
    y1 = n["cy"] - n["h"] / 2
    return x1, y1, x1 + n["w"], y1 + n["h"]


def node_center(nid: str) -> tuple[float, float]:
    n = NODES[nid]
    return float(n["cx"]), float(n["cy"])


def rect_anchor(nid: str, toward: tuple[float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = node_box(nid)
    cx, cy = node_center(nid)
    tx, ty = toward
    dx, dy = tx - cx, ty - cy
    if abs(dx) > abs(dy):
        return (x2 if dx > 0 else x1, cy)
    return (cx, y2 if dy > 0 else y1)


def draw_round_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    radius: int,
    fill,
    outline=None,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_node_card(draw: ImageDraw.ImageDraw, nid: str, fonts: dict, scale: int) -> None:
    n = NODES[nid]
    c = TYPE_COLORS[n["type"]]
    x1, y1, x2, y2 = node_box(nid)
    pad = 2 * scale
    draw_round_rect(
        draw,
        (x1 - pad, y1 - pad, x2 + pad, y2 + pad),
        12 * scale,
        (248, 250, 252),
    )
    draw_round_rect(draw, (x1, y1, x2, y2), 10 * scale, c["fill"], outline=c["stroke"], width=2 * scale)
    # 顶栏色条
    bar_h = 6 * scale
    draw_round_rect(draw, (x1, y1, x2, y1 + bar_h + 4 * scale), 10 * scale, c["hdr"])
    draw.rectangle((x1, y1 + bar_h, x2, y1 + bar_h + 4 * scale), fill=c["hdr"])

    cx, cy = n["cx"], n["cy"]
    if n["sub"]:
        draw.text((cx, cy - 8 * scale), n["title"], fill=c["text"], font=fonts["node"], anchor="mm")
        draw.text((cx, cy + 14 * scale), n["sub"], fill=c["text"], font=fonts["sub"], anchor="mm")
    else:
        draw.text((cx, cy + 2 * scale), n["title"], fill=c["text"], font=fonts["node"], anchor="mm")


def draw_edge(
    draw: ImageDraw.ImageDraw,
    n1: str,
    n2: str,
    label: str,
    color: tuple[int, int, int],
    dashed: bool,
    curve: float,
    fonts: dict,
    scale: int,
) -> None:
    c2 = node_center(n2)
    p1 = rect_anchor(n1, c2)
    c1 = node_center(n1)
    p2 = rect_anchor(n2, c1)

    mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    dist = max(math.hypot(dx, dy), 1.0)
    nx, ny = -dy / dist, dx / dist
    cx, cy = mx + nx * dist * curve, my + ny * dist * curve

    steps = 32
    pts: list[tuple[float, float]] = []
    for i in range(steps + 1):
        t = i / steps
        x = (1 - t) ** 2 * p1[0] + 2 * (1 - t) * t * cx + t**2 * p2[0]
        y = (1 - t) ** 2 * p1[1] + 2 * (1 - t) * t * cy + t**2 * p2[1]
        pts.append((x, y))

    lw = 2 * scale
    if dashed:
        seg = 8 * scale
        gap = 6 * scale
        for i in range(len(pts) - 1):
            ax, ay = pts[i]
            bx, by = pts[i + 1]
            seg_len = math.hypot(bx - ax, by - ay)
            if seg_len < 1:
                continue
            ux, uy = (bx - ax) / seg_len, (by - ay) / seg_len
            pos = 0.0
            on = True
            while pos < seg_len:
                ln = min(seg if on else gap, seg_len - pos)
                if on:
                    sx, sy = ax + ux * pos, ay + uy * pos
                    ex, ey = ax + ux * (pos + ln), ay + uy * (pos + ln)
                    draw.line([(sx, sy), (ex, ey)], fill=color, width=lw)
                pos += ln
                on = not on
    else:
        draw.line(pts, fill=color, width=lw)

    ax, ay = pts[-2]
    bx, by = pts[-1]
    ang = math.atan2(by - ay, bx - ax)
    sz = 10 * scale
    draw.polygon(
        [
            (bx, by),
            (bx - sz * math.cos(ang - 0.42), by - sz * math.sin(ang - 0.42)),
            (bx - sz * math.cos(ang + 0.42), by - sz * math.sin(ang + 0.42)),
        ],
        fill=color,
    )

    tw = draw.textlength(label, font=fonts["rel"])
    pad_x, pad_y = 10 * scale, 6 * scale
    lx, ly = cx, cy
    draw_round_rect(
        draw,
        (lx - tw / 2 - pad_x, ly - 12 * scale, lx + tw / 2 + pad_x, ly + 12 * scale),
        10 * scale,
        (255, 255, 255),
        outline=color,
        width=max(1, scale),
    )
    draw.text((lx, ly), label, fill=color, font=fonts["rel"], anchor="mm")


def render_frame(scale: int) -> Image.Image:
    sw, sh = W * scale, H * scale
    img = Image.new("RGB", (sw, sh), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    fonts = {
        "title": load_font(24 * scale, bold=True),
        "subtitle": load_font(14 * scale),
        "col": load_font(15 * scale, bold=True),
        "node": load_font(16 * scale, bold=True),
        "sub": load_font(13 * scale),
        "rel": load_font(13 * scale, bold=True),
        "legend": load_font(13 * scale),
        "legend_b": load_font(13 * scale, bold=True),
    }

    m = 28 * scale
    draw_round_rect(draw, (m, m, sw - m, sh - m), 16 * scale, (255, 255, 255), outline=(226, 232, 240), width=scale)

    draw.text((sw // 2, 52 * scale), "知识图谱实例子图", fill=(15, 23, 42), font=fonts["title"], anchor="mm")
    draw.text(
        (sw // 2, 78 * scale),
        "地震事件 → 行政区域 → 应急主题 → 处置步骤（数据来自本地知识目录）",
        fill=(100, 116, 139),
        font=fonts["subtitle"],
        anchor="mm",
    )

    # 列标题
    for cx, label, color in COLUMNS:
        tw = draw.textlength(label, font=fonts["col"])
        bx = cx * scale
        draw_round_rect(
            draw,
            (bx - tw / 2 - 14 * scale, 98 * scale, bx + tw / 2 + 14 * scale, 126 * scale),
            8 * scale,
            color,
        )
        draw.text((bx, 112 * scale), label, fill=(255, 255, 255), font=fonts["col"], anchor="mm")

    # 列分隔虚线
    for x in (295, 545, 815):
        y0, y1 = 138 * scale, 690 * scale
        pos = y0
        while pos < y1:
            draw.line([(x * scale, pos), (x * scale, min(pos + 10 * scale, y1))], fill=(226, 232, 240), width=scale)
            pos += 16 * scale

    # 缩放坐标系：临时把 draw 逻辑映射到 scale
    class ScaledDraw:
        def __init__(self, base: ImageDraw.ImageDraw, s: int):
            self.base = base
            self.s = s

        def _scale_nodes(self):
            scaled = {}
            for k, n in NODES.items():
                scaled[k] = {**n, "cx": n["cx"] * s, "cy": n["cy"] * s, "w": n["w"] * s, "h": n["h"] * s}
            return scaled

    # 直接在 scaled 坐标下绘制
    global NODES
    orig = NODES
    NODES = {k: {**v, "cx": v["cx"] * scale, "cy": v["cy"] * scale, "w": v["w"] * scale, "h": v["h"] * scale} for k, v in orig.items()}

    for n1, n2, label, color, dashed, curve in EDGES:
        draw_edge(draw, n1, n2, label, color, dashed, curve, fonts, scale)
    for nid in NODES:
        draw_node_card(draw, nid, fonts, scale)

    NODES = orig

    # 图例
    ly0, ly1 = 718 * scale, 768 * scale
    draw_round_rect(draw, (48 * scale, ly0, sw - 48 * scale, ly1), 10 * scale, (248, 250, 252), outline=(226, 232, 240), width=scale)
    draw.text((68 * scale, (ly0 + ly1) / 2), "图例", fill=(15, 23, 42), font=fonts["legend_b"], anchor="lm")
    x = 120 * scale
    for name, key in [("地震事件", "地震"), ("行政区域", "区域"), ("应急主题", "主题"), ("处置步骤", "步骤")]:
        c = TYPE_COLORS[key]
        draw_round_rect(draw, (x, ly0 + 16 * scale, x + 22 * scale, ly0 + 34 * scale), 4 * scale, c["fill"], outline=c["stroke"], width=scale)
        draw.text((x + 30 * scale, (ly0 + ly1) / 2), name, fill=(51, 65, 85), font=fonts["legend"], anchor="lm")
        x += 130 * scale
    draw.line([(680 * scale, (ly0 + ly1) / 2), (720 * scale, (ly0 + ly1) / 2)], fill=(13, 148, 136), width=2 * scale)
    draw.text((730 * scale, (ly0 + ly1) / 2), "实线：结构关系", fill=(51, 65, 85), font=fonts["legend"], anchor="lm")
    draw.line([(880 * scale, (ly0 + ly1) / 2), (920 * scale, (ly0 + ly1) / 2)], fill=(124, 58, 237), width=2 * scale)
    draw.text((930 * scale, (ly0 + ly1) / 2), "虚线：推荐 / 关联", fill=(51, 65, 85), font=fonts["legend"], anchor="lm")

    return img


def generate_png() -> Path:
    img = render_frame(SCALE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(PNG_PATH, format="PNG", dpi=(300, 300), optimize=True)
    return PNG_PATH


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def generate_svg() -> Path:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<defs>",
        '<filter id="shadow"><feDropShadow dx="0" dy="2" stdDeviation="4" flood-color="#64748b" flood-opacity="0.12"/></filter>',
        '<marker id="arr" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="#64748b"/></marker>',
        "</defs>",
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<rect x="28" y="28" width="{W-56}" height="{H-56}" rx="16" fill="#fff" stroke="#e2e8f0" filter="url(#shadow)"/>',
        '<text x="600" y="52" text-anchor="middle" font-family="PingFang SC,Microsoft YaHei,SimHei,sans-serif" font-size="24" font-weight="700" fill="#0f172a">知识图谱实例子图</text>',
        '<text x="600" y="78" text-anchor="middle" font-family="PingFang SC,Microsoft YaHei,SimHei,sans-serif" font-size="14" fill="#64748b">地震事件 → 行政区域 → 应急主题 → 处置步骤（数据来自本地知识目录）</text>',
    ]
    for cx, label, color in COLUMNS:
        lines.append(
            f'<rect x="{cx-52}" y="98" width="104" height="28" rx="8" fill="{_hex(color)}"/>'
        )
        lines.append(
            f'<text x="{cx}" y="117" text-anchor="middle" font-family="PingFang SC,Microsoft YaHei,sans-serif" font-size="15" font-weight="700" fill="#fff">{label}</text>'
        )

    for n1, n2, label, color, dashed, curve in EDGES:
        c2 = node_center(n2)
        p1 = rect_anchor(n1, c2)
        c1 = node_center(n1)
        p2 = rect_anchor(n2, c1)
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        dist = max(math.hypot(dx, dy), 1.0)
        nx, ny = -dy / dist, dx / dist
        cx, cy = mx + nx * dist * curve, my + ny * dist * curve
        hc = _hex(color)
        dash = ' stroke-dasharray="8 5"' if dashed else ""
        lines.append(
            f'<path d="M {p1[0]:.1f} {p1[1]:.1f} Q {cx:.1f} {cy:.1f} {p2[0]:.1f} {p2[1]:.1f}" fill="none" stroke="{hc}" stroke-width="2.5"{dash} marker-end="url(#arr)"/>'
        )
        tw = len(label) * 14 + 20
        lines.append(f'<rect x="{cx-tw/2:.1f}" y="{cy-13:.1f}" width="{tw:.1f}" height="26" rx="10" fill="#fff" stroke="{hc}"/>')
        lines.append(
            f'<text x="{cx:.1f}" y="{cy+5:.1f}" text-anchor="middle" font-family="PingFang SC,Microsoft YaHei,sans-serif" font-size="13" font-weight="700" fill="{hc}">{label}</text>'
        )

    for nid, n in NODES.items():
        c = TYPE_COLORS[n["type"]]
        x1, y1, x2, y2 = node_box(nid)
        lines.append(
            f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{n["w"]:.1f}" height="{n["h"]:.1f}" rx="10" fill="{_hex(c["fill"])}" stroke="{_hex(c["stroke"])}" stroke-width="2" filter="url(#shadow)"/>'
        )
        lines.append(f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{n["w"]:.1f}" height="8" rx="10" fill="{_hex(c["hdr"])}"/>')
        if n["sub"]:
            lines.append(
                f'<text x="{n["cx"]:.1f}" y="{n["cy"]-6:.1f}" text-anchor="middle" font-family="PingFang SC,Microsoft YaHei,sans-serif" font-size="16" font-weight="700" fill="{_hex(c["text"])}">{n["title"]}</text>'
            )
            lines.append(
                f'<text x="{n["cx"]:.1f}" y="{n["cy"]+16:.1f}" text-anchor="middle" font-family="PingFang SC,Microsoft YaHei,sans-serif" font-size="13" fill="{_hex(c["text"])}">{n["sub"]}</text>'
            )
        else:
            lines.append(
                f'<text x="{n["cx"]:.1f}" y="{n["cy"]+5:.1f}" text-anchor="middle" font-family="PingFang SC,Microsoft YaHei,sans-serif" font-size="16" font-weight="700" fill="{_hex(c["text"])}">{n["title"]}</text>'
            )

    lines.append("</svg>")
    SVG_PATH.write_text("\n".join(lines), encoding="utf-8")
    return SVG_PATH


def main() -> None:
    png = generate_png()
    svg = generate_svg()
    static = ROOT / "static" / "fig-4-2-kg-instance.png"
    static.write_bytes(png.read_bytes())
    print(f"OK PNG ({W * SCALE}×{H * SCALE}, 300dpi): {png}")
    print(f"OK SVG ({W}×{H}): {svg}")
    print(f"OK copy: {static}")


if __name__ == "__main__":
    main()
