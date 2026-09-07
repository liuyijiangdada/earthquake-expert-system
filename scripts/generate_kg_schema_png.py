#!/usr/bin/env python3
"""生成全中文知识图谱 ER 模式图（高清 PNG + 矢量 SVG）。"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "superpowers" / "architecture"
PNG_PATH = OUT_DIR / "fig-4-1-kg-schema.png"
SVG_PATH = OUT_DIR / "fig-4-1-kg-schema.svg"

W, H = 1200, 760
SCALE = 3

# 图例区域（实体卡片底边应在其上方留 LEGEND_GAP）
LEGEND_Y0 = 590
LEGEND_Y1 = 730
LEGEND_GAP = 28

COLORS = {
    "white": (255, 255, 255),
    "border": (226, 232, 240),
    "title": (15, 23, 42),
    "sub": (100, 116, 139),
    "text": (51, 65, 85),
    "muted": (148, 163, 184),
    "eq": (37, 99, 235),
    "eq_hdr": (37, 99, 235),
    "eq_fill": (219, 234, 254),
    "eq_text": (30, 58, 138),
    "rg": (13, 148, 136),
    "rg_hdr": (13, 148, 136),
    "rg_fill": (204, 251, 241),
    "rg_text": (15, 94, 89),
    "tp": (124, 58, 237),
    "tp_hdr": (124, 58, 237),
    "tp_fill": (237, 233, 254),
    "tp_text": (76, 29, 149),
    "st": (234, 88, 12),
    "st_hdr": (234, 88, 12),
    "st_fill": (255, 237, 213),
    "st_text": (154, 52, 18),
}

# 实体卡片：(x1, y1, x2, y2)
ENTITIES = {
    "eq": {
        "title": "地震事件",
        "box": (80, 175, 360, 385),
        "hdr": "eq_hdr",
        "fill": "eq_fill",
        "stroke": "eq",
        "text": "eq_text",
        "attrs": [
            "事件编号　唯一标识",
            "事件名称　发震地点描述",
            "发震时间　日期时间",
            "震级 / 深度　浮点数",
            "震中位置　经纬度坐标",
        ],
        "note": "来源：真实地震事件目录",
    },
    "rg": {
        "title": "行政区域",
        "box": (80, 432, 360, 558),
        "hdr": "rg_hdr",
        "fill": "rg_fill",
        "stroke": "rg",
        "text": "rg_text",
        "attrs": ["区域名称　唯一（如：四川、云南）"],
        "note": "按省区筛选与聚合",
    },
    "tp": {
        "title": "应急主题",
        "box": (800, 200, 1060, 440),
        "hdr": "tp_hdr",
        "fill": "tp_fill",
        "stroke": "tp",
        "text": "tp_text",
        "attrs": [
            "主题编号　唯一标识",
            "主题标题　知识主题名称",
            "知识类别　避险 / 处置 / 恢复",
            "阶段标签　震前 / 震中 / 震后",
            "知识时效　半衰期（供调度参考）",
        ],
        "note": "当前共 13 个主题，对应检索分块",
    },
}

EDGES = [
    ("eq", "rg", "发生于", "rg", False, 0.0, "bottom", "top"),
    ("eq", "tp", "建议主题", "tp", True, -0.06, "right", "left"),
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


def c(key: str) -> tuple[int, int, int]:
    return COLORS[key]


def box_anchor(box: tuple[int, int, int, int], side: str) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if side == "top":
        return cx, y1
    if side == "bottom":
        return cx, y2
    if side == "left":
        return x1, cy
    return x2, cy


def draw_round_rect(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_entity_card(draw, ent: dict, fonts: dict, scale: int) -> None:
    x1, y1, x2, y2 = ent["box"]
    hdr = c(ent["hdr"])
    fill = c(ent["fill"])
    stroke = c(ent["stroke"])
    text_c = c(ent["text"])

    pad = 3 * scale
    draw_round_rect(draw, (x1 - pad, y1 - pad, x2 + pad, y2 + pad), 14 * scale, (248, 250, 252))
    draw_round_rect(draw, (x1, y1, x2, y2), 12 * scale, fill, outline=stroke, width=2 * scale)

    if ent.get("compact"):
        # 紧凑卡片：左侧色条 + 标题，无顶栏大色块
        draw.rectangle((x1, y1 + 4 * scale, x1 + 5 * scale, y2 - 4 * scale), fill=stroke)
        draw.text((x1 + 16 * scale, y1 + 18 * scale), ent["title"], fill=stroke, font=fonts["card_title"], anchor="lm")
        y = y1 + 44 * scale
        for line in ent["attrs"]:
            draw.text((x1 + 16 * scale, y), line, fill=text_c, font=fonts["attr"])
            y += 22 * scale
        return

    bar_h = 42 * scale
    draw_round_rect(draw, (x1, y1, x2, y1 + bar_h), 12 * scale, hdr)
    draw.rectangle((x1, y1 + bar_h - 8 * scale, x2, y1 + bar_h), fill=hdr)
    draw.text(((x1 + x2) / 2, y1 + 22 * scale), ent["title"], fill=COLORS["white"], font=fonts["card_title"], anchor="mm")
    draw.line([(x1 + 16 * scale, y1 + bar_h + 8 * scale), (x2 - 16 * scale, y1 + bar_h + 8 * scale)], fill=c("border"), width=scale)

    y = y1 + bar_h + 28 * scale
    for line in ent["attrs"]:
        draw.text((x1 + 18 * scale, y), line, fill=text_c, font=fonts["attr"])
        y += 24 * scale
    if ent.get("note"):
        draw.text((x1 + 18 * scale, y2 - 22 * scale), ent["note"], fill=c("muted"), font=fonts["note"])


def draw_bezier_edge(
    draw,
    p1: tuple[float, float],
    p2: tuple[float, float],
    label: str,
    color: tuple[int, int, int],
    dashed: bool,
    curve: float,
    fonts: dict,
    scale: int,
    label_sub: str = "",
) -> None:
    mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    dist = max(math.hypot(dx, dy), 1.0)
    nx, ny = -dy / dist, dx / dist
    cx, cy = mx + nx * dist * curve, my + ny * dist * curve

    steps = 36
    pts = []
    for i in range(steps + 1):
        t = i / steps
        x = (1 - t) ** 2 * p1[0] + 2 * (1 - t) * t * cx + t**2 * p2[0]
        y = (1 - t) ** 2 * p1[1] + 2 * (1 - t) * t * cy + t**2 * p2[1]
        pts.append((x, y))

    lw = 2 * scale
    if dashed:
        seg, gap = 10 * scale, 7 * scale
        for i in range(len(pts) - 1):
            ax, ay = pts[i]
            bx, by = pts[i + 1]
            seg_len = math.hypot(bx - ax, by - ay)
            if seg_len < 1:
                continue
            ux, uy = (bx - ax) / seg_len, (by - ay) / seg_len
            pos, on = 0.0, True
            while pos < seg_len:
                ln = min(seg if on else gap, seg_len - pos)
                if on:
                    draw.line(
                        [(ax + ux * pos, ay + uy * pos), (ax + ux * (pos + ln), ay + uy * (pos + ln))],
                        fill=color,
                        width=lw,
                    )
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
    pad_x = 12 * scale
    lx, ly = cx, cy - (8 * scale if label_sub else 0)
    draw_round_rect(
        draw,
        (lx - tw / 2 - pad_x, ly - 13 * scale, lx + tw / 2 + pad_x, ly + 13 * scale),
        10 * scale,
        COLORS["white"],
        outline=color,
        width=max(1, scale),
    )
    draw.text((lx, ly), label, fill=color, font=fonts["rel"], anchor="mm")
    if label_sub:
        draw.text((lx, ly + 22 * scale), label_sub, fill=c("muted"), font=fonts["note"], anchor="mm")


def draw_self_loop_top(
    draw,
    box: tuple[float, float, float, float],
    color: tuple[int, int, int],
    fonts: dict,
    scale: int,
) -> None:
    """在实体卡片上方绘制自环，与卡片顶栏保持明显间距。"""
    x1, y1, x2, _y2 = box
    mid_x = (x1 + x2) / 2
    inset = 36 * scale
    left, right = x1 + inset, x2 - inset

    # 弧线整体在卡片顶边之上：峰顶距卡片顶边 gap，底端距顶边 stub
    gap = 42 * scale       # 峰顶到卡片顶边的距离
    stub = 10 * scale      # 从顶边向上引出的短竖线
    peak_y = y1 - gap
    base_y = y1 - stub

    # 右 → 上 → 左 的拱形（控制点抬高，避免压到卡片标题栏）
    ctrl_y = peak_y - 8 * scale
    path_pts = []
    for i in range(41):
        t = i / 40
        # 三次贝塞尔：right,base → right,ctrl → left,ctrl → left,base
        u = 1 - t
        px = (
            u**3 * right
            + 3 * u**2 * t * right
            + 3 * u * t**2 * left
            + t**3 * left
        )
        py = (
            u**3 * base_y
            + 3 * u**2 * t * ctrl_y
            + 3 * u * t**2 * ctrl_y
            + t**3 * base_y
        )
        path_pts.append((px, py))

    # 竖线引脚（与卡片顶边衔接）
    for sx in (right, left):
        draw.line([(sx, y1), (sx, base_y)], fill=color, width=2 * scale)

    for i in range(len(path_pts) - 1):
        ax, ay = path_pts[i]
        bx, by = path_pts[i + 1]
        seg_len = math.hypot(bx - ax, by - ay)
        if seg_len < 1:
            continue
        ux, uy = (bx - ax) / seg_len, (by - ay) / seg_len
        pos, on = 0.0, True
        while pos < seg_len:
            ln = min((10 * scale if on else 7 * scale), seg_len - pos)
            if on:
                draw.line(
                    [(ax + ux * pos, ay + uy * pos), (ax + ux * (pos + ln), ay + uy * (pos + ln))],
                    fill=color,
                    width=2 * scale,
                )
            pos += ln
            on = not on

    # 箭头指向左引脚
    ax, ay = path_pts[-2]
    bx, by = path_pts[-1]
    ang = math.atan2(by - ay, bx - ax)
    sz = 9 * scale
    draw.polygon(
        [
            (bx, by),
            (bx - sz * math.cos(ang - 0.42), by - sz * math.sin(ang - 0.42)),
            (bx - sz * math.cos(ang + 0.42), by - sz * math.sin(ang + 0.42)),
        ],
        fill=color,
    )

    label = "关联主题"
    tw = draw.textlength(label, font=fonts["rel"])
    lx, ly = mid_x, peak_y - 14 * scale
    draw_round_rect(
        draw,
        (lx - tw / 2 - 10 * scale, ly - 12 * scale, lx + tw / 2 + 10 * scale, ly + 12 * scale),
        10 * scale,
        COLORS["white"],
        outline=color,
        width=max(1, scale),
    )
    draw.text((lx, ly), label, fill=color, font=fonts["rel"], anchor="mm")


def draw_legend(draw, sw: int, scale: int, fonts: dict) -> None:
    """规范两行图例：第一行节点类型，第二行关系类型。"""
    lx, rx = 56 * scale, sw - 56 * scale
    ly0, ly1 = LEGEND_Y0 * scale, LEGEND_Y1 * scale
    draw_round_rect(draw, (lx, ly0, rx, ly1), 12 * scale, (248, 250, 252), outline=c("border"), width=scale)

    # 标题区
    draw.text((76 * scale, ly0 + 28 * scale), "图例", fill=c("title"), font=fonts["legend_b"], anchor="lm")
    draw.line([(120 * scale, ly0 + 16 * scale), (120 * scale, ly1 - 16 * scale)], fill=c("border"), width=scale)

    # 第一行：节点类型
    row1_y = ly0 + 36 * scale
    draw.text((140 * scale, row1_y), "节点类型", fill=c("muted"), font=fonts["legend_label"], anchor="lm")
    items = [("地震事件", "eq"), ("行政区域", "rg"), ("应急主题", "tp")]
    col_x = [320, 560, 800]
    for (name, fk), cx in zip(items, col_x):
        ent = ENTITIES[fk]
        swatch = 22 * scale
        sx = cx * scale - 70 * scale
        draw_round_rect(
            draw,
            (sx, row1_y - 12 * scale, sx + swatch, row1_y + 10 * scale),
            5 * scale,
            c(ent["fill"]),
            outline=c(ent["stroke"]),
            width=scale,
        )
        draw.text((sx + swatch + 8 * scale, row1_y), name, fill=c("text"), font=fonts["legend"], anchor="lm")

    # 分隔线
    div_y = ly0 + 72 * scale
    draw.line([(140 * scale, div_y), (rx - 40 * scale, div_y)], fill=c("border"), width=scale)

    # 第二行：关系类型（固定两列）
    row2_y = ly0 + 108 * scale
    draw.text((140 * scale, row2_y), "关系类型", fill=c("muted"), font=fonts["legend_label"], anchor="lm")
    for x0, color, dashed, desc in [
        (240 * scale, c("rg"), False, "实线箭头：一对多（发生于）"),
        (620 * scale, c("tp"), True, "虚线箭头：推荐 / 关联（建议主题、关联主题）"),
    ]:
        x1_line = x0 + 44 * scale
        if dashed:
            sx = x0
            while sx < x1_line:
                draw.line([(sx, row2_y), (min(sx + 8 * scale, x1_line), row2_y)], fill=color, width=2 * scale)
                sx += 12 * scale
        else:
            draw.line([(x0, row2_y), (x1_line, row2_y)], fill=color, width=2 * scale)
        draw.polygon(
            [(x1_line, row2_y), (x1_line - 8 * scale, row2_y - 4 * scale), (x1_line - 8 * scale, row2_y + 4 * scale)],
            fill=color,
        )
        draw.text((x1_line + 12 * scale, row2_y), desc, fill=c("text"), font=fonts["legend"], anchor="lm")


def render_frame(scale: int) -> Image.Image:
    sw, sh = W * scale, H * scale
    img = Image.new("RGB", (sw, sh), COLORS["white"])
    draw = ImageDraw.Draw(img)

    fonts = {
        "title": load_font(24 * scale, bold=True),
        "sub": load_font(14 * scale),
        "card_title": load_font(17 * scale, bold=True),
        "attr": load_font(14 * scale),
        "note": load_font(12 * scale),
        "rel": load_font(14 * scale, bold=True),
        "legend_b": load_font(15 * scale, bold=True),
        "legend_label": load_font(12 * scale, bold=True),
        "legend": load_font(13 * scale),
        "zone": load_font(13 * scale, bold=True),
    }

    m = 28 * scale
    draw_round_rect(draw, (m, m, sw - m, sh - m), 16 * scale, COLORS["white"], outline=c("border"), width=scale)

    draw.text((sw // 2, 54 * scale), "知识图谱实体关系模式", fill=c("title"), font=fonts["title"], anchor="mm")
    draw.text(
        (sw // 2, 80 * scale),
        "图数据库 · 地震事件与应急知识结构化建模",
        fill=c("sub"),
        font=fonts["sub"],
        anchor="mm",
    )

    # 左右分区标题（顶部横条，不占用右侧边距）
    # 右侧不再单独放「应急知识」标签，避免与自环「关联主题」拥挤
    for cx, label, color in [(220, "静态事实", c("eq"))]:
        tw = draw.textlength(label, font=fonts["zone"])
        draw_round_rect(
            draw,
            (cx * scale - tw / 2 - 14 * scale, 118 * scale, cx * scale + tw / 2 + 14 * scale, 142 * scale),
            8 * scale,
            color,
        )
        draw.text((cx * scale, 130 * scale), label, fill=COLORS["white"], font=fonts["zone"], anchor="mm")

    scaled_entities = {
        k: {**v, "box": tuple(co * scale for co in v["box"])}
        for k, v in ENTITIES.items()
    }

    for fk, tk, label, ck, dashed, curve, fs, ts in EDGES:
        p1 = box_anchor(scaled_entities[fk]["box"], fs)
        p2 = box_anchor(scaled_entities[tk]["box"], ts)
        sub = "按震级 / 区域推荐" if label == "建议主题" else ""
        draw_bezier_edge(draw, p1, p2, label, c(ck), dashed, curve, fonts, scale, sub)

    # 先画自环，再画应急主题卡片，避免弧线与标题栏重叠
    draw_self_loop_top(draw, scaled_entities["tp"]["box"], c("tp"), fonts, scale)
    for key, ent in scaled_entities.items():
        draw_entity_card(draw, ent, fonts, scale)
    draw_legend(draw, sw, scale, fonts)

    return img


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
        f'<text x="{W//2}" y="54" text-anchor="middle" font-family="PingFang SC,Microsoft YaHei,SimHei,sans-serif" font-size="24" font-weight="700" fill="#0f172a">知识图谱实体关系模式</text>',
        f'<text x="{W//2}" y="80" text-anchor="middle" font-family="PingFang SC,Microsoft YaHei,sans-serif" font-size="14" fill="#64748b">图数据库 · 地震事件与应急知识结构化建模</text>',
    ]

    for eid, ent in ENTITIES.items():
        x1, y1, x2, y2 = ent["box"]
        hdr, fill, stroke, text_c = c(ent["hdr"]), c(ent["fill"]), c(ent["stroke"]), c(ent["text"])
        lines.append(
            f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" rx="12" fill="{_hex(fill)}" stroke="{_hex(stroke)}" stroke-width="2" filter="url(#shadow)"/>'
        )
        if ent.get("compact"):
            lines.append(f'<rect x="{x1}" y="{y1+4}" width="5" height="{y2-y1-8}" fill="{_hex(stroke)}"/>')
            lines.append(
                f'<text x="{x1+16}" y="{y1+22}" font-family="PingFang SC,Microsoft YaHei,sans-serif" font-size="17" font-weight="700" fill="{_hex(stroke)}">{ent["title"]}</text>'
            )
            y = y1 + 48
            for attr in ent["attrs"]:
                lines.append(
                    f'<text x="{x1+16}" y="{y}" font-family="PingFang SC,Microsoft YaHei,sans-serif" font-size="14" fill="{_hex(text_c)}">{attr}</text>'
                )
                y += 22
            continue
        lines.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="42" rx="12" fill="{_hex(hdr)}"/>')
        lines.append(
            f'<text x="{(x1+x2)/2}" y="{y1+26}" text-anchor="middle" font-family="PingFang SC,Microsoft YaHei,sans-serif" font-size="17" font-weight="700" fill="#fff">{ent["title"]}</text>'
        )
        y = y1 + 68
        for attr in ent["attrs"]:
            lines.append(
                f'<text x="{x1+18}" y="{y}" font-family="PingFang SC,Microsoft YaHei,sans-serif" font-size="14" fill="{_hex(text_c)}">{attr}</text>'
            )
            y += 24
        if ent.get("note"):
            lines.append(
                f'<text x="{x1+18}" y="{y2-14}" font-family="PingFang SC,Microsoft YaHei,sans-serif" font-size="12" fill="#94a3b8">{ent["note"]}</text>'
            )

    for fk, tk, label, ck, dashed, curve, fs, ts in EDGES:
        p1 = box_anchor(ENTITIES[fk]["box"], fs)
        p2 = box_anchor(ENTITIES[tk]["box"], ts)
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        dist = max(math.hypot(dx, dy), 1.0)
        nx, ny = -dy / dist, dx / dist
        cx, cy = mx + nx * dist * curve, my + ny * dist * curve
        hc = _hex(c(ck))
        dash = ' stroke-dasharray="10 7"' if dashed else ""
        lines.append(
            f'<path d="M {p1[0]:.1f} {p1[1]:.1f} Q {cx:.1f} {cy:.1f} {p2[0]:.1f} {p2[1]:.1f}" fill="none" stroke="{hc}" stroke-width="2.5"{dash} marker-end="url(#arr)"/>'
        )
        tw = len(label) * 14 + 24
        lines.append(f'<rect x="{cx-tw/2:.1f}" y="{cy-14:.1f}" width="{tw:.1f}" height="28" rx="10" fill="#fff" stroke="{hc}"/>')
        lines.append(
            f'<text x="{cx:.1f}" y="{cy+5:.1f}" text-anchor="middle" font-family="PingFang SC,Microsoft YaHei,sans-serif" font-size="14" font-weight="700" fill="{hc}">{label}</text>'
        )

    # 自环（卡片上方，与顶栏分离）
    x1, y1, x2, y2 = ENTITIES["tp"]["box"]
    mid_x = (x1 + x2) / 2
    inset = 36
    left, right = x1 + inset, x2 - inset
    gap, stub = 42, 10
    peak_y = y1 - gap
    base_y = y1 - stub
    ctrl_y = peak_y - 8
    tp_c = _hex(c("tp"))
    lines.append(
        f'<path d="M {right} {y1} L {right} {base_y} C {right} {ctrl_y} {left} {ctrl_y} {left} {base_y} L {left} {y1}" '
        f'fill="none" stroke="{tp_c}" stroke-width="2.5" stroke-dasharray="8 5"/>'
    )
    lines.append(
        f'<rect x="{mid_x - 42}" y="{peak_y - 30}" width="84" height="24" rx="10" fill="#fff" stroke="{tp_c}"/>'
    )
    lines.append(
        f'<text x="{mid_x}" y="{peak_y - 14}" text-anchor="middle" font-family="PingFang SC,Microsoft YaHei,sans-serif" '
        f'font-size="14" font-weight="700" fill="{tp_c}">关联主题</text>'
    )

    # 图例
    ly0, ly1 = LEGEND_Y0, LEGEND_Y1
    lines.append(f'<rect x="56" y="{ly0}" width="{W-112}" height="{ly1-ly0}" rx="12" fill="#f8fafc" stroke="#e2e8f0"/>')
    lines.append(f'<text x="76" y="{ly0+28}" font-family="PingFang SC,Microsoft YaHei,sans-serif" font-size="15" font-weight="700" fill="#0f172a">图例</text>')
    lines.append(f'<line x1="120" y1="{ly0+16}" x2="120" y2="{ly1-16}" stroke="#e2e8f0"/>')
    lines.append(f'<text x="140" y="{ly0+40}" font-family="PingFang SC,Microsoft YaHei,sans-serif" font-size="12" font-weight="700" fill="#94a3b8">节点类型</text>')
    for name, fk, col in [("地震事件", "eq", 320), ("行政区域", "rg", 560), ("应急主题", "tp", 800)]:
        ent = ENTITIES[fk]
        lines.append(f'<rect x="{col}" y="{ly0+28}" width="22" height="18" rx="5" fill="{_hex(c(ent["fill"]))}" stroke="{_hex(c(ent["stroke"]))}" stroke-width="1.5"/>')
        lines.append(f'<text x="{col+30}" y="{ly0+42}" font-family="PingFang SC,Microsoft YaHei,sans-serif" font-size="13" fill="#334155">{name}</text>')
    lines.append(f'<line x1="140" y1="{ly0+72}" x2="{W-96}" y2="{ly0+72}" stroke="#e2e8f0"/>')
    lines.append(f'<text x="140" y="{ly0+112}" font-family="PingFang SC,Microsoft YaHei,sans-serif" font-size="12" font-weight="700" fill="#94a3b8">关系类型</text>')
    lines.append(f'<line x1="240" y1="{ly0+108}" x2="284" y2="{ly0+108}" stroke="{_hex(c("rg"))}" stroke-width="2.5"/>')
    lines.append(f'<text x="292" y="{ly0+112}" font-family="PingFang SC,Microsoft YaHei,sans-serif" font-size="13" fill="#334155">实线箭头：一对多（发生于）</text>')
    lines.append(f'<line x1="580" y1="{ly0+108}" x2="624" y2="{ly0+108}" stroke="{_hex(c("tp"))}" stroke-width="2.5" stroke-dasharray="8 5"/>')
    lines.append(f'<text x="632" y="{ly0+112}" font-family="PingFang SC,Microsoft YaHei,sans-serif" font-size="13" fill="#334155">虚线箭头：推荐 / 关联（建议主题、关联主题）</text>')

    lines.append("</svg>")
    SVG_PATH.write_text("\n".join(lines), encoding="utf-8")
    return SVG_PATH


def generate_png() -> Path:
    img = render_frame(SCALE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(PNG_PATH, format="PNG", dpi=(300, 300), optimize=True)
    return PNG_PATH


def main() -> None:
    png = generate_png()
    svg = generate_svg()
    static = ROOT / "static" / "fig-4-1-kg-schema.png"
    static.write_bytes(png.read_bytes())
    print(f"OK PNG ({W * SCALE}×{H * SCALE}, 300dpi): {png}")
    print(f"OK SVG ({W}×{H}): {svg}")
    print(f"OK copy: {static}")


if __name__ == "__main__":
    main()
