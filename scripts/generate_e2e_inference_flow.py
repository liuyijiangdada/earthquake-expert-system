#!/usr/bin/env python3
"""生成「端到端协同推理流程」高清流程图（PNG + SVG）。

设计目标：分阶段分区、左右泳道对比清晰、去掉与泳道标题重复的节点。
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "superpowers" / "architecture"
PNG_PATH = OUT_DIR / "fig-6-1-e2e-inference-flow.png"
SVG_PATH = OUT_DIR / "fig-6-1-e2e-inference-flow.svg"
STATIC_PNG = ROOT / "static" / "fig-6-1-e2e-inference-flow.png"

W, H = 1080, 1120
SCALE = 3

BOX_W, BOX_H = 260, 50
GAP_Y = 18
CX, LX, RX = 540, 270, 810

COLORS = {
    "white": (255, 255, 255),
    "bg": (248, 250, 252),
    "border": (226, 232, 240),
    "title": (15, 23, 42),
    "sub": (100, 116, 139),
    "stage": (71, 85, 105),
    "arrow": (148, 163, 184),
    "in_fill": (37, 99, 235),
    "kg_fill": (8, 145, 178),
    "kg_bg": (236, 254, 255),
    "rag_fill": (147, 51, 234),
    "rag_bg": (250, 245, 255),
    "merge_bg": (255, 247, 237),
    "m1_fill": (234, 88, 12),
    "m2_fill": (220, 38, 38),
    "m3_fill": (153, 27, 27),
    "llm_fill": (76, 29, 149),
    "out_fill": (22, 163, 74),
    "lane_kg": (14, 116, 144),
    "lane_rag": (126, 34, 206),
    "hint": (100, 116, 139),
}

# 泳道标题已说明分支名，节点只写步骤
LEFT_NODES = [
    ("地区/震级/应急触发", "规则命中才查库"),
    ("Neo4j 子图查询", "Cypher 结构化检索"),
    ("图谱上下文 Φ", "可核对事实与步骤"),
]
RIGHT_NODES = [
    ("句向量编码", "bge-small-zh"),
    ("余弦相似度 Top-K", "默认 K=5"),
    ("参考资料片段 Ψ", "科普/规程补充"),
]
MERGE_NODES = [
    ("① 提示模板拼装", "图谱 + 检索 + 规则 + 问题", "m1_fill"),
    ("② 对话模板套入", "对齐微调 chat 格式", "m2_fill"),
    ("③ 左侧截断保尾", "超长时保留末尾问题", "m3_fill"),
    ("④ 本地 LLM 推理", "Qwen1.5-1.8B + LoRA", "llm_fill"),
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


def box_rect(cx: float, cy: float, w: float = BOX_W, h: float = BOX_H):
    return int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)


def draw_round_rect(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_arrow(draw, p1, p2, scale, color=None):
    color = color or c("arrow")
    draw.line([p1, p2], fill=color, width=max(2, 2 * scale // 2 + 1))
    ang = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    sz = 8 * scale
    bx, by = p2
    draw.polygon(
        [
            (bx, by),
            (bx - sz * math.cos(ang - 0.4), by - sz * math.sin(ang - 0.4)),
            (bx - sz * math.cos(ang + 0.4), by - sz * math.sin(ang + 0.4)),
        ],
        fill=color,
    )


def draw_node(
    draw,
    cx,
    cy,
    title: str,
    fill,
    font_title,
    scale,
    *,
    subtitle: str | None = None,
    font_sub=None,
    w=BOX_W,
    h=BOX_H,
):
    box = box_rect(cx, cy, w, h)
    sbox = tuple(v * scale for v in box)
    draw_round_rect(draw, sbox, 10 * scale, fill)
    if subtitle and font_sub:
        draw.text(
            (cx * scale, (cy - 8) * scale),
            title,
            fill=(255, 255, 255),
            font=font_title,
            anchor="mm",
        )
        draw.text(
            (cx * scale, (cy + 12) * scale),
            subtitle,
            fill=(226, 232, 240),
            font=font_sub,
            anchor="mm",
        )
    else:
        draw.text(
            (cx * scale, cy * scale),
            title,
            fill=(255, 255, 255),
            font=font_title,
            anchor="mm",
        )
    return box


def render_frame(scale: int) -> Image.Image:
    sw, sh = W * scale, H * scale
    img = Image.new("RGB", (sw, sh), c("bg"))
    draw = ImageDraw.Draw(img)

    fonts = {
        "title": load_font(26 * scale, bold=True),
        "sub": load_font(14 * scale),
        "stage": load_font(13 * scale, bold=True),
        "node": load_font(15 * scale, bold=True),
        "node_sm": load_font(14 * scale, bold=True),
        "hint": load_font(11 * scale),
        "lane": load_font(15 * scale, bold=True),
    }

    # 外框
    m = 20 * scale
    draw_round_rect(
        draw, (m, m, sw - m, sh - m), 16 * scale, c("white"), outline=c("border"), width=scale
    )

    # 标题
    draw.text(
        (sw // 2, 48 * scale),
        "端到端协同推理流程",
        fill=c("title"),
        font=fonts["title"],
        anchor="mm",
    )
    draw.text(
        (sw // 2, 76 * scale),
        "问句并行走图谱与检索两路 → 拼装提示 → 本地大模型生成",
        fill=c("sub"),
        font=fonts["sub"],
        anchor="mm",
    )

    # ---- 阶段① 输入 ----
    y_stage1 = 108
    draw.text(
        (60 * scale, y_stage1 * scale),
        "① 输入",
        fill=c("stage"),
        font=fonts["stage"],
        anchor="lm",
    )
    in_cy = 148
    input_box = draw_node(
        draw, CX, in_cy, "用户问题 q", c("in_fill"), fonts["node"], scale
    )

    # ---- 阶段② 双路检索（泳道面板）----
    y_stage2 = 198
    draw.text(
        (60 * scale, y_stage2 * scale),
        "② 双路检索（并行）",
        fill=c("stage"),
        font=fonts["stage"],
        anchor="lm",
    )

    lane_top = 218
    lane_bottom = 470
    # 左泳道底
    draw_round_rect(
        draw,
        (
            (LX - BOX_W / 2 - 18) * scale,
            lane_top * scale,
            (LX + BOX_W / 2 + 18) * scale,
            lane_bottom * scale,
        ),
        12 * scale,
        c("kg_bg"),
        outline=(165, 243, 252),
        width=scale,
    )
    # 右泳道底
    draw_round_rect(
        draw,
        (
            (RX - BOX_W / 2 - 18) * scale,
            lane_top * scale,
            (RX + BOX_W / 2 + 18) * scale,
            lane_bottom * scale,
        ),
        12 * scale,
        c("rag_bg"),
        outline=(233, 213, 255),
        width=scale,
    )

    draw.text(
        (LX * scale, (lane_top + 22) * scale),
        "知识图谱分支 Φ",
        fill=c("lane_kg"),
        font=fonts["lane"],
        anchor="mm",
    )
    draw.text(
        (LX * scale, (lane_top + 42) * scale),
        "结构化 · 可核对",
        fill=c("hint"),
        font=fonts["hint"],
        anchor="mm",
    )
    draw.text(
        (RX * scale, (lane_top + 22) * scale),
        "向量检索分支 Ψ",
        fill=c("lane_rag"),
        font=fonts["lane"],
        anchor="mm",
    )
    draw.text(
        (RX * scale, (lane_top + 42) * scale),
        "语义召回 · 表述补充",
        fill=c("hint"),
        font=fonts["hint"],
        anchor="mm",
    )

    node0_y = lane_top + 78
    left_boxes, right_boxes = [], []
    for i, ((lt, ls), (rt, rs)) in enumerate(zip(LEFT_NODES, RIGHT_NODES)):
        cy = node0_y + i * (BOX_H + GAP_Y) + BOX_H / 2
        left_boxes.append(
            draw_node(
                draw,
                LX,
                cy,
                lt,
                c("kg_fill"),
                fonts["node_sm"],
                scale,
                subtitle=ls,
                font_sub=fonts["hint"],
                h=54,
            )
        )
        right_boxes.append(
            draw_node(
                draw,
                RX,
                cy,
                rt,
                c("rag_fill"),
                fonts["node_sm"],
                scale,
                subtitle=rs,
                font_sub=fonts["hint"],
                h=54,
            )
        )

    # 输入 → 分叉
    split_y = (input_box[3] + lane_top) / 2 + 8
    draw_arrow(
        draw,
        (CX * scale, input_box[3] * scale),
        (CX * scale, split_y * scale),
        scale,
    )
    # 横线分叉
    draw.line(
        [(LX * scale, split_y * scale), (RX * scale, split_y * scale)],
        fill=c("arrow"),
        width=2 * scale,
    )
    draw_arrow(
        draw,
        (LX * scale, split_y * scale),
        (LX * scale, left_boxes[0][1] * scale),
        scale,
        color=c("lane_kg"),
    )
    draw_arrow(
        draw,
        (RX * scale, split_y * scale),
        (RX * scale, right_boxes[0][1] * scale),
        scale,
        color=c("lane_rag"),
    )
    draw.text(
        ((CX + 70) * scale, (split_y - 12) * scale),
        "并行",
        fill=c("hint"),
        font=fonts["hint"],
        anchor="mm",
    )

    for boxes, col, color in (
        (left_boxes, LX, c("lane_kg")),
        (right_boxes, RX, c("lane_rag")),
    ):
        for i in range(len(boxes) - 1):
            draw_arrow(
                draw,
                (col * scale, boxes[i][3] * scale),
                (col * scale, boxes[i + 1][1] * scale),
                scale,
                color=color,
            )

    # ---- 阶段③ 融合 ----
    y_stage3 = 500
    draw.text(
        (60 * scale, y_stage3 * scale),
        "③ 提示融合",
        fill=c("stage"),
        font=fonts["stage"],
        anchor="lm",
    )

    merge_panel_top = 518
    merge_panel_bot = 900
    draw_round_rect(
        draw,
        (
            (CX - BOX_W / 2 - 28) * scale,
            merge_panel_top * scale,
            (CX + BOX_W / 2 + 28) * scale,
            merge_panel_bot * scale,
        ),
        12 * scale,
        c("merge_bg"),
        outline=(254, 215, 170),
        width=scale,
    )

    merge0_y = merge_panel_top + 40
    merge_boxes = []
    for i, (title, sub, fk) in enumerate(MERGE_NODES):
        cy = merge0_y + i * (58 + 14) + 29
        merge_boxes.append(
            draw_node(
                draw,
                CX,
                cy,
                title,
                c(fk),
                fonts["node_sm"],
                scale,
                subtitle=sub,
                font_sub=fonts["hint"],
                h=58,
            )
        )

    # 双路汇合
    join_y = merge_panel_top + 8
    draw_arrow(
        draw,
        (LX * scale, left_boxes[-1][3] * scale),
        (LX * scale, join_y * scale),
        scale,
        color=c("lane_kg"),
    )
    draw_arrow(
        draw,
        (RX * scale, right_boxes[-1][3] * scale),
        (RX * scale, join_y * scale),
        scale,
        color=c("lane_rag"),
    )
    draw.line(
        [(LX * scale, join_y * scale), (RX * scale, join_y * scale)],
        fill=c("arrow"),
        width=2 * scale,
    )
    draw_arrow(
        draw,
        (CX * scale, join_y * scale),
        (CX * scale, merge_boxes[0][1] * scale),
        scale,
    )
    draw.text(
        ((CX + 78) * scale, (join_y - 10) * scale),
        "汇合",
        fill=c("hint"),
        font=fonts["hint"],
        anchor="mm",
    )

    for i in range(len(merge_boxes) - 1):
        draw_arrow(
            draw,
            (CX * scale, merge_boxes[i][3] * scale),
            (CX * scale, merge_boxes[i + 1][1] * scale),
            scale,
        )

    # ---- 阶段④ 输出 ----
    y_stage4 = 930
    draw.text(
        (60 * scale, y_stage4 * scale),
        "④ 输出",
        fill=c("stage"),
        font=fonts["stage"],
        anchor="lm",
    )
    out_cy = 980
    out_box = draw_node(draw, CX, out_cy, "回答 a", c("out_fill"), fonts["node"], scale)
    draw_arrow(
        draw,
        (CX * scale, merge_boxes[-1][3] * scale),
        (CX * scale, out_box[1] * scale),
        scale,
    )

    # 图例
    draw.text(
        (sw // 2, 1065 * scale),
        "青=图谱事实通路　　紫=向量检索通路　　橙红=提示处理　　绿=最终回答",
        fill=c("hint"),
        font=fonts["hint"],
        anchor="mm",
    )

    return img


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def generate_svg() -> Path:
    """简化 SVG：与 PNG 同结构，便于论文插图替换。"""
    # 直接导出 PNG 为主；SVG 用栅格友好的简化版路径说明
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#f8fafc"/>',
        f'<rect x="20" y="20" width="{W-40}" height="{H-40}" rx="16" fill="#fff" stroke="#e2e8f0"/>',
        f'<text x="{CX}" y="52" text-anchor="middle" font-family="PingFang SC,Microsoft YaHei,sans-serif" '
        f'font-size="26" font-weight="700" fill="#0f172a">端到端协同推理流程</text>',
        f'<text x="{CX}" y="78" text-anchor="middle" font-family="PingFang SC,Microsoft YaHei,sans-serif" '
        f'font-size="14" fill="#64748b">问句并行走图谱与检索两路 → 拼装提示 → 本地大模型生成</text>',
        '<text x="60" y="200" font-family="PingFang SC,Microsoft YaHei,sans-serif" font-size="13" '
        'font-weight="700" fill="#475569">详见同名 PNG（推荐论文插图使用 PNG）</text>',
        "</svg>",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SVG_PATH.write_text("\n".join(lines), encoding="utf-8")
    return SVG_PATH


def generate_png() -> Path:
    img = render_frame(SCALE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(PNG_PATH, format="PNG", dpi=(300, 300), optimize=True)
    STATIC_PNG.parent.mkdir(parents=True, exist_ok=True)
    img.save(STATIC_PNG, format="PNG", dpi=(300, 300), optimize=True)
    return PNG_PATH


def main() -> None:
    png = generate_png()
    svg = generate_svg()
    print(f"PNG: {png}  ({png.stat().st_size // 1024} KB)")
    print(f"SVG: {svg}")
    print(f"Static copy: {STATIC_PNG}")


if __name__ == "__main__":
    main()
