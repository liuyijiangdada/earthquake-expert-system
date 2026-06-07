#!/usr/bin/env python3
"""
重新生成 docs/superpowers/architecture 下的三张 SVG（UTF-8 合法 XML）。
若 SVG 在预览或 Word 中打不开，多半是编码损坏，执行本脚本即可恢复。
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main() -> None:
    base = Path(__file__).resolve().parent.parent / "docs" / "superpowers" / "architecture"

    svg1 = r"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="920" height="560" viewBox="0 0 920 560">
  <defs>
    <style type="text/css"><![CDATA[
      .title { font-family: "PingFang SC","Microsoft YaHei","SimHei",sans-serif; font-size: 16px; font-weight: bold; fill: #1a1a1a; }
      .box { fill: #f7f9fc; stroke: #2c5282; stroke-width: 2; rx: 10; }
      .box-db { fill: #edf7f6; stroke: #276749; stroke-width: 2; rx: 10; }
      .box-llm { fill: #faf5ff; stroke: #553c9a; stroke-width: 2; rx: 10; }
      .lbl { font-family: "PingFang SC","Microsoft YaHei","SimHei",sans-serif; font-size: 14px; fill: #1a202c; }
      .sml { font-family: "PingFang SC","Microsoft YaHei","SimHei",sans-serif; font-size: 12px; fill: #4a5568; }
      .arrow { stroke: #2d3748; stroke-width: 2; fill: none; marker-end: url(#m); }
      .dash { stroke: #718096; stroke-width: 1.5; stroke-dasharray: 6 4; fill: none; marker-end: url(#m2); }
    ]]></style>
    <marker id="m" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <polygon points="0 0, 10 3, 0 6" fill="#2d3748"/>
    </marker>
    <marker id="m2" markerWidth="8" markerHeight="8" refX="7" refY="2.5" orient="auto">
      <polygon points="0 0, 8 2.5, 0 5" fill="#718096"/>
    </marker>
  </defs>
  <text x="460" y="36" text-anchor="middle" class="title">""" + esc(
        "图 2-1-1 系统总体逻辑架构"
    ) + r"""</text>
  <rect class="box" x="280" y="60" width="360" height="72"/>
  <text x="460" y="92" text-anchor="middle" class="lbl">""" + esc("展示层：用户浏览器") + r"""</text>
  <text x="460" y="112" text-anchor="middle" class="sml">""" + esc(
        "static/index.html（对话 / 列表 / 筛选 / 更新）"
    ) + r"""</text>
  <path class="arrow" d="M 460 132 L 460 158"/>
  <text x="472" y="150" class="sml">HTTP / JSON</text>
  <rect class="box" x="220" y="168" width="480" height="88"/>
  <text x="460" y="200" text-anchor="middle" class="lbl">""" + esc("应用服务层：Flask（app.py）") + r"""</text>
  <text x="460" y="222" text-anchor="middle" class="sml">""" + esc(
        "POST /api/query（llm | kg）、POST /api/update-data、GET /"
    ) + r"""</text>
  <text x="460" y="242" text-anchor="middle" class="sml">""" + esc(
        "generate_response：组装【知识图谱】【参考资料】并调用本地 LLM"
    ) + r"""</text>
  <path class="arrow" d="M 320 256 L 320 300"/>
  <path class="arrow" d="M 460 256 L 460 300"/>
  <path class="arrow" d="M 600 256 L 600 300"/>
  <rect class="box-db" x="60" y="308" width="240" height="100"/>
  <text x="180" y="342" text-anchor="middle" class="lbl">""" + esc("Neo4j 知识图谱") + r"""</text>
  <text x="180" y="364" text-anchor="middle" class="sml">kg/neo4j_kg.py</text>
  <text x="180" y="386" text-anchor="middle" class="sml">""" + esc("地震 / 区域 / 应急主题与步骤") + r"""</text>
  <rect class="box-db" x="340" y="308" width="240" height="100"/>
  <text x="460" y="342" text-anchor="middle" class="lbl">""" + esc("RAG 向量检索") + r"""</text>
  <text x="460" y="364" text-anchor="middle" class="sml">rag/emergency_rag.py</text>
  <text x="460" y="386" text-anchor="middle" class="sml">SentenceTransformer · Top-K</text>
  <rect class="box-llm" x="620" y="308" width="240" height="100"/>
  <text x="740" y="342" text-anchor="middle" class="lbl">""" + esc("本地大模型") + r"""</text>
  <text x="740" y="364" text-anchor="middle" class="sml">transformers + peft（LoRA）</text>
  <text x="740" y="386" text-anchor="middle" class="sml">""" + esc("基座 + 微调适配器") + r"""</text>
  <rect class="box" x="360" y="440" width="200" height="56" style="fill:#fffaf0;stroke:#c05621"/>
  <text x="460" y="468" text-anchor="middle" class="lbl">config/config.py</text>
  <text x="460" y="486" text-anchor="middle" class="sml">""" + esc("Neo4j、路径、RAG/LLM 开关与参数") + r"""</text>
  <path class="dash" d="M 460 428 L 460 440"/>
  <path class="dash" d="M 180 408 L 180 430 L 400 468"/>
  <path class="dash" d="M 460 408 L 460 440"/>
  <path class="dash" d="M 740 408 L 740 430 L 520 468"/>
</svg>
"""

    svg2 = r"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="720" height="920" viewBox="0 0 720 920">
  <defs>
    <style type="text/css"><![CDATA[
      .title { font-family: "PingFang SC","Microsoft YaHei","SimHei",sans-serif; font-size: 16px; font-weight: bold; fill: #1a1a1a; }
      .node { fill: #ebf4ff; stroke: #2b6cb0; stroke-width: 2; rx: 8; }
      .node-g { fill: #e6fffa; stroke: #276749; stroke-width: 2; rx: 8; }
      .node-r { fill: #fefcbf; stroke: #b7791f; stroke-width: 2; rx: 8; }
      .node-m { fill: #faf5ff; stroke: #553c9a; stroke-width: 2; rx: 8; }
      .lbl { font-family: "PingFang SC","Microsoft YaHei","SimHei",sans-serif; font-size: 13px; fill: #1a202c; }
      .sml { font-family: "PingFang SC","Microsoft YaHei","SimHei",sans-serif; font-size: 11px; fill: #4a5568; }
      .arrow { stroke: #2d3748; stroke-width: 2; fill: none; marker-end: url(#am); }
    ]]></style>
    <marker id="am" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <polygon points="0 0, 10 3, 0 6" fill="#2d3748"/>
    </marker>
  </defs>
  <text x="360" y="32" text-anchor="middle" class="title">""" + esc(
        "图 2-1-2 问答数据流（query_type: llm）"
    ) + r"""</text>
  <rect class="node" x="230" y="52" width="260" height="44"/>
  <text x="360" y="80" text-anchor="middle" class="lbl">""" + esc("用户输入自然语言问题") + r"""</text>
  <path class="arrow" d="M 360 96 L 360 118"/>
  <rect class="node" x="200" y="120" width="320" height="52"/>
  <text x="360" y="144" text-anchor="middle" class="lbl">POST /api/query</text>
  <text x="360" y="162" text-anchor="middle" class="sml">""" + esc("query_type = llm，params.input") + r"""</text>
  <path class="arrow" d="M 360 172 L 360 194"/>
  <rect class="node" x="240" y="196" width="240" height="40"/>
  <text x="360" y="222" text-anchor="middle" class="lbl">generate_response（app.py）</text>
  <path class="arrow" d="M 280 236 L 140 260"/>
  <path class="arrow" d="M 360 236 L 360 260"/>
  <path class="arrow" d="M 440 236 L 580 260"/>
  <rect class="node-g" x="40" y="262" width="200" height="76"/>
  <text x="140" y="288" text-anchor="middle" class="lbl">""" + esc("知识图谱分支") + r"""</text>
  <text x="140" y="308" text-anchor="middle" class="sml">""" + esc("省名 / 震级规则 / 应急语境") + r"""</text>
  <text x="140" y="326" text-anchor="middle" class="sml">""" + esc("Neo4j 查询得到图谱上下文") + r"""</text>
  <rect class="node-r" x="260" y="262" width="200" height="76"/>
  <text x="360" y="288" text-anchor="middle" class="lbl">""" + esc("向量检索分支") + r"""</text>
  <text x="360" y="308" text-anchor="middle" class="sml">_build_rag_section</text>
  <text x="360" y="326" text-anchor="middle" class="sml">""" + esc("余弦 Top-K 得到参考资料") + r"""</text>
  <rect class="node" x="480" y="262" width="200" height="76"/>
  <text x="580" y="296" text-anchor="middle" class="lbl">""" + esc("配置开关") + r"""</text>
  <text x="580" y="314" text-anchor="middle" class="sml">KG_CONTEXT</text>
  <text x="580" y="330" text-anchor="middle" class="sml">RAG_ENABLED</text>
  <path class="arrow" d="M 140 338 L 140 380 L 360 380"/>
  <path class="arrow" d="M 360 338 L 360 380"/>
  <path class="arrow" d="M 580 338 L 580 380 L 360 380"/>
  <rect class="node" x="180" y="382" width="360" height="64"/>
  <text x="360" y="396" text-anchor="middle" class="lbl">""" + esc("拼装 user 提示") + r"""</text>
  <text x="360" y="414" text-anchor="middle" class="sml">""" + esc(
        "【知识图谱】【参考资料】+ 规则 + 【问题】（问题在末尾）"
    ) + r"""</text>
  <text x="360" y="430" text-anchor="middle" class="sml">""" + esc("system / user 模板与微调格式一致") + r"""</text>
  <path class="arrow" d="M 360 434 L 360 456"/>
  <rect class="node-m" x="200" y="458" width="320" height="56"/>
  <text x="360" y="482" text-anchor="middle" class="lbl">tokenizer（truncation_side = left）</text>
  <text x="360" y="500" text-anchor="middle" class="sml">max_length = LLM_INPUT_MAX_TOKENS</text>
  <path class="arrow" d="M 360 514 L 360 536"/>
  <rect class="node-m" x="200" y="538" width="320" height="56"/>
  <text x="360" y="562" text-anchor="middle" class="lbl">model.generate</text>
  <text x="360" y="580" text-anchor="middle" class="sml">""" + esc("temperature、top_p、repetition_penalty 等") + r"""</text>
  <path class="arrow" d="M 360 594 L 360 616"/>
  <rect class="node" x="220" y="618" width="280" height="44"/>
  <text x="360" y="646" text-anchor="middle" class="lbl">""" + esc("decode 新生成 token 为文本") + r"""</text>
  <path class="arrow" d="M 360 662 L 360 684"/>
  <rect class="node" x="210" y="686" width="300" height="44"/>
  <text x="360" y="714" text-anchor="middle" class="lbl">""" + esc("JSON { response } 返回前端") + r"""</text>
  <text x="360" y="780" text-anchor="middle" class="sml">""" + esc(
        "另：query_type = kg 仅查 Neo4j，不经大模型（列表与筛选）"
    ) + r"""</text>
  <text x="360" y="802" text-anchor="middle" class="sml">""" + esc(
        "POST /api/update-data：update_from_realtime_data() 刷新目录"
    ) + r"""</text>
</svg>
"""

    rows = [
        ("模块", "主要代码 / 技术", "职责"),
        ("前端", "static/index.html", "对话、地震列表、筛选、触发数据更新"),
        ("应用网关", "app.py（Flask）", "路由、组装 KG/RAG 上下文、调用 LLM"),
        ("知识图谱", "kg/neo4j_kg.py + Neo4j", "地震事件、区域、应急主题与处置步骤"),
        ("向量检索", "rag/emergency_rag.py", "应急知识 JSON 分块、嵌入、余弦 Top-K"),
        ("大模型", "transformers + peft（LoRA）", "本地基座权重 + 微调适配器推理"),
        ("配置", "config/config.py", "Neo4j、路径、RAG/LLM 开关与解码参数"),
    ]
    y0, h_hdr, h_row = 56, 36, 44
    parts3 = [
        r"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="880" height="380" viewBox="0 0 880 380">
  <defs>
    <style type="text/css"><![CDATA[
      .title { font-family: "PingFang SC","Microsoft YaHei","SimHei",sans-serif; font-size: 16px; font-weight: bold; fill: #1a1a1a; }
      .hdr { font-family: "PingFang SC","Microsoft YaHei","SimHei",sans-serif; font-size: 13px; font-weight: bold; fill: #fff; }
      .cell { font-family: "PingFang SC","Microsoft YaHei","SimHei",sans-serif; font-size: 12px; fill: #1a202c; }
      .row0 { fill: #2c5282; }
      .row1 { fill: #f7fafc; stroke: #e2e8f0; stroke-width: 1; }
      .row2 { fill: #ffffff; stroke: #e2e8f0; stroke-width: 1; }
    ]]></style>
  </defs>
  <text x="440" y="32" text-anchor="middle" class="title">"""
        + esc("图 2-1-3 技术栈与模块映射")
        + "</text>\n"
    ]
    y = y0
    parts3.append(f'  <rect class="row0" x="40" y="{y}" width="140" height="{h_hdr}"/>\n')
    parts3.append(f'  <rect class="row0" x="180" y="{y}" width="320" height="{h_hdr}"/>\n')
    parts3.append(f'  <rect class="row0" x="500" y="{y}" width="340" height="{h_hdr}"/>\n')
    parts3.append(f'  <text x="110" y="{y + 24}" text-anchor="middle" class="hdr">{esc(rows[0][0])}</text>\n')
    parts3.append(f'  <text x="340" y="{y + 24}" text-anchor="middle" class="hdr">{esc(rows[0][1])}</text>\n')
    parts3.append(f'  <text x="670" y="{y + 24}" text-anchor="middle" class="hdr">{esc(rows[0][2])}</text>\n')
    y += h_hdr
    for i, (a, b, c) in enumerate(rows[1:], start=1):
        cls = "row1" if i % 2 == 1 else "row2"
        parts3.append(f'  <rect class="{cls}" x="40" y="{y}" width="140" height="{h_row}"/>\n')
        parts3.append(f'  <rect class="{cls}" x="180" y="{y}" width="320" height="{h_row}"/>\n')
        parts3.append(f'  <rect class="{cls}" x="500" y="{y}" width="340" height="{h_row}"/>\n')
        parts3.append(f'  <text x="110" y="{y + 26}" text-anchor="middle" class="cell">{esc(a)}</text>\n')
        parts3.append(f'  <text x="340" y="{y + 26}" text-anchor="middle" class="cell">{esc(b)}</text>\n')
        parts3.append(f'  <text x="520" y="{y + 18}" class="cell">{esc(c)}</text>\n')
        y += h_row
    parts3.append("</svg>\n")
    svg3 = "".join(parts3)

    for name, content in [
        ("fig-2-1-1-system-architecture.svg", svg1),
        ("fig-2-1-2-query-flow.svg", svg2),
        ("fig-2-1-3-tech-stack.svg", svg3),
    ]:
        path = base / name
        path.write_text(content, encoding="utf-8")
        ET.parse(path)
        print("OK", path)


if __name__ == "__main__":
    main()
