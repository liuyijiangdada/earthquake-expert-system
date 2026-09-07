# -*- coding: utf-8 -*-
"""重建论文目录：移动混入目录区的正文、删除旧目录条目、设置标题大纲级别、插入TOC域。"""
import re
import copy
from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

THESIS = "华东师范大学硕士论文.docx"


def main():
    doc = Document(THESIS)
    paras = doc.paragraphs
    body = doc.element.body

    # ========== 1. 定位目录区域 ==========
    toc_heading_idx = None
    body_start_idx = None
    fig_list_idx = None
    for i, p in enumerate(paras):
        t = p.text.strip()
        if t == "目录" and toc_heading_idx is None:
            toc_heading_idx = i
        if t == "图目录" and fig_list_idx is None:
            fig_list_idx = i
        if (toc_heading_idx is not None and i > toc_heading_idx
                and re.match(r"^第[一二三四五六七八九十]+章\s", t) and i > 150):
            body_start_idx = i
            break

    print(f"目录标题: [{toc_heading_idx}], 图目录: [{fig_list_idx}], 正文起始: [{body_start_idx}]")

    # ========== 2. 识别目录区混入的正文段落 ==========
    # 正文段落：无域代码、无页码、长文本
    body_text_elements = []  # (element, text, preceding_heading_text)
    preceding_heading = ""
    entries_to_delete = []

    for i in range(toc_heading_idx + 1, fig_list_idx):
        p = paras[i]
        t = p.text.strip()
        if not t:
            continue
        has_field = "fldChar" in p._p.xml or "instrText" in p._p.xml
        has_page = bool(re.search(r"\s+\d+\s*$", t))
        is_short = len(t) < 60

        if has_field or (has_page and is_short):
            # 这是目录条目，记录标题文本
            preceding_heading = re.sub(r"\s+\d+\s*$", "", t).strip()
            entries_to_delete.append(p._p)
        else:
            # 这是混入的正文
            body_text_elements.append((p._p, t, preceding_heading))
            print(f"  正文段落(属'{preceding_heading}'): {t[:50]}...")

    print(f"待移动正文: {len(body_text_elements)}, 待删除目录条目: {len(entries_to_delete)}")

    # ========== 3. 将正文段落移到正文中对应节末尾 ==========
    # 按标题分组
    from collections import OrderedDict
    grouped = OrderedDict()
    for elem, text, heading in body_text_elements:
        grouped.setdefault(heading, []).append((elem, text))

    moved_count = 0
    for heading_text, items in grouped.items():
        # 在正文中找到该标题段落
        target_heading_elem = None
        next_heading_elem = None
        for p in paras[body_start_idx:]:
            t = p.text.strip()
            # 标题匹配（去掉多余空格）
            t_norm = re.sub(r"\s+", " ", t).strip()
            h_norm = re.sub(r"\s+", " ", heading_text).strip()
            if t_norm == h_norm and target_heading_elem is None:
                target_heading_elem = p._p
            elif target_heading_elem is not None:
                # 找到下一个标题
                if (re.match(r"^[0-9]+\.[0-9]+(\.[0-9]+)?\s", t) or
                        re.match(r"^第[一二三四五六七八九十]+章\s", t) or
                        t in ("参考文献", "致谢", "本章小结")):
                    next_heading_elem = p._p
                    break

        if target_heading_elem is None:
            print(f"  警告：正文未找到标题 '{heading_text}'，跳过移动")
            continue

        # 插入点：下一个标题前（或如果没有下一个标题，插在标题段后几段）
        insert_before = next_heading_elem if next_heading_elem is not None else None
        # 如果找不到下一个标题，插在 target_heading 后面
        if insert_before is None:
            insert_before = target_heading_elem.getnext()
            while insert_before is not None and insert_before.tag == qn("w:p"):
                # 找到该节最后一段的下一个元素
                insert_before = insert_before.getnext()

        for elem, text in items:
            # 从原位置移除
            parent = elem.getparent()
            if parent is not None:
                parent.remove(elem)
            # 插入到目标位置前
            if insert_before is not None:
                insert_before.addprevious(elem)
            else:
                body.append(elem)
            moved_count += 1
            print(f"  移动 '{text[:30]}...' -> '{heading_text}' 节")

    print(f"已移动 {moved_count} 段正文")

    # ========== 4. 删除旧目录条目 ==========
    deleted_count = 0
    for elem in entries_to_delete:
        parent = elem.getparent()
        if parent is not None:
            parent.remove(elem)
            deleted_count += 1
    print(f"已删除 {deleted_count} 个旧目录条目")

    # ========== 5. 给正文标题设置大纲级别 ==========
    # 重新读取段落（结构已变）
    paras = doc.paragraphs
    outline_set = 0
    for p in paras:
        t = p.text.strip()
        level = None
        if re.match(r"^第[一二三四五六七八九十]+章\s", t) and len(t) < 40:
            level = 0
        elif re.match(r"^[0-9]+\.[0-9]+\.[0-9]+\s", t) and len(t) < 60:
            level = 2
        elif re.match(r"^[0-9]+\.[0-9]+\s", t) and len(t) < 50:
            level = 1
        elif t in ("参考文献", "致谢"):
            level = 0

        if level is not None:
            pPr = p._p.find(qn("w:pPr"))
            if pPr is None:
                pPr = OxmlElement("w:pPr")
                p._p.insert(0, pPr)
            # 移除已有的 outlineLvl
            existing = pPr.find(qn("w:outlineLvl"))
            if existing is not None:
                pPr.remove(existing)
            ol = OxmlElement("w:outlineLvl")
            ol.set(qn("w:val"), str(level))
            pPr.append(ol)
            outline_set += 1

    print(f"已设置 {outline_set} 个标题的大纲级别")

    # ========== 6. 在"目录"标题后插入 TOC 域 ==========
    # 重新读取段落
    paras = doc.paragraphs
    toc_heading_elem = None
    for p in paras:
        if p.text.strip() == "目录":
            toc_heading_elem = p._p
            break

    if toc_heading_elem is not None:
        # 创建 TOC 域段落
        toc_p = OxmlElement("w:p")
        pPr = OxmlElement("w:pPr")
        toc_p.append(pPr)

        # fldChar begin
        r1 = OxmlElement("w:r")
        fc1 = OxmlElement("w:fldChar")
        fc1.set(qn("w:fldCharType"), "begin")
        r1.append(fc1)
        toc_p.append(r1)

        # instrText
        r2 = OxmlElement("w:r")
        it = OxmlElement("w:instrText")
        it.set(qn("xml:space"), "preserve")
        it.text = ' TOC \\o "1-3" \\h \\z \\u '
        r2.append(it)
        toc_p.append(r2)

        # fldChar separate
        r3 = OxmlElement("w:r")
        fc2 = OxmlElement("w:fldChar")
        fc2.set(qn("w:fldCharType"), "separate")
        r3.append(fc2)
        toc_p.append(r3)

        # 占位提示文本
        r4 = OxmlElement("w:r")
        t4 = OxmlElement("w:t")
        t4.text = "（请在 WPS 中右键此处→更新域，生成目录）"
        r4.append(t4)
        toc_p.append(r4)

        # fldChar end
        r5 = OxmlElement("w:r")
        fc3 = OxmlElement("w:fldChar")
        fc3.set(qn("w:fldCharType"), "end")
        r5.append(fc3)
        toc_p.append(r5)

        # 插入到"目录"标题段之后
        toc_heading_elem.addnext(toc_p)
        print("已插入 TOC 域")

    # ========== 保存 ==========
    doc.save(THESIS)
    print("完成：目录已重建，请在 WPS 中打开并右键更新域")


if __name__ == "__main__":
    main()
