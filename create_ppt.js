const pptxgen = require("pptxgenjs");

// ============================================================
// SLIDE DIMENSIONS - REQUIRED FIRST, USE EVERYWHERE
// ============================================================
const SLIDE_W = 10;      // inches
const SLIDE_H = 5.625;   // inches

// Define safe content area (with margins)
const MARGIN = 0.5;
const CONTENT_X = MARGIN;
const CONTENT_Y = MARGIN;
const CONTENT_W = SLIDE_W - (2 * MARGIN);  // 9 inches
const CONTENT_H = SLIDE_H - (2 * MARGIN);  // 4.625 inches

// Common layout helpers
const CENTER_X = SLIDE_W / 2;              // 5 inches
const CENTER_Y = SLIDE_H / 2;              // 2.8125 inches

// ============================================================
// COLOR PALETTE - 科技蓝+应急橙
// ============================================================
const COLORS = {
  primary: '1E3A5F',      // 深蓝 - 主色
  secondary: '2E5A8C',    // 中蓝 - 辅助
  accent: 'E85D04',       // 橙色 - 强调/应急
  light: 'F0F4F8',        // 浅蓝灰 - 背景
  white: 'FFFFFF',
  text: '1A1A2E',         // 深色文字
  textLight: '4A5568'     // 浅灰文字
};

// ============================================================
// CONTAINER SYSTEM WITH TEXT OVERFLOW PROTECTION - REQUIRED
// ============================================================
function createVirtualNode(type, data, parentX = 0, parentY = 0) {
  const opts = data.opts || {};
  const node = {
    type, data,
    absX: parentX + (opts.x || 0),
    absY: parentY + (opts.y || 0),
    w: opts.w || 0, h: opts.h || 0,
    children: []
  };
  node.addShape = function(shapeType, opts = {}) {
    const child = createVirtualNode('shape', { shapeType, opts }, node.absX, node.absY);
    node.children.push(child);
    return child;
  };
  node.addText = function(text, opts = {}) {
    const safeOpts = { fit: "shrink", ...opts };
    const bulletRe = /^(?:[\u2022\u2023\u25E6\u2043\u2219\u00B7\u25CF\u25CB\u2013\u2014]\s*|\-\s+)/;
    if (Array.isArray(text)) {
      text = text.map(item => {
        if (item && item.options && item.options.bullet && typeof item.text === 'string') {
          return { ...item, text: item.text.replace(bulletRe, '') };
        }
        return item;
      });
    }
    const child = createVirtualNode('text', { text, opts: safeOpts }, node.absX, node.absY);
    node.children.push(child);
    return child;
  };
  node.addImage = function(opts = {}) {
    const child = createVirtualNode('image', { opts }, node.absX, node.absY);
    node.children.push(child);
    return child;
  };
  node.addTable = function(tableData, opts = {}) {
    const child = createVirtualNode('table', { tableData, opts }, node.absX, node.absY);
    node.children.push(child);
    return child;
  };
  return node;
}

function flattenNode(node, realSlide, pres) {
  const absOpts = { ...node.data.opts, x: node.absX, y: node.absY };
  if (node.type === 'shape') realSlide.addShape(node.data.shapeType, absOpts);
  else if (node.type === 'text') realSlide.addText(node.data.text, absOpts);
  else if (node.type === 'image') realSlide.addImage(absOpts);
  else if (node.type === 'table') realSlide.addTable(node.data.tableData, absOpts);
  node.children.forEach(child => flattenNode(child, realSlide, pres));
}

let pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';
pres.author = '刘一江';
pres.title = '基于知识图谱与向量检索协同的地震应急问答方法研究';

const originalAddSlide = pres.addSlide.bind(pres);
pres.addSlide = function(options) {
  const realSlide = originalAddSlide(options);
  const virtualSlide = {
    children: [],
    _realSlide: realSlide,
    set background(val) { realSlide.background = val; },
    get background() { return realSlide.background; },
    addShape: function(shapeType, opts = {}) {
      const node = createVirtualNode('shape', { shapeType, opts }, 0, 0);
      this.children.push(node);
      return node;
    },
    addText: function(text, opts = {}) {
      const safeOpts = { fit: "shrink", ...opts };
      const node = createVirtualNode('text', { text, opts: safeOpts }, 0, 0);
      this.children.push(node);
      return node;
    },
    addImage: function(opts = {}) {
      const node = createVirtualNode('image', { opts }, 0, 0);
      this.children.push(node);
      return node;
    },
    addTable: function(tableData, opts = {}) {
      const node = createVirtualNode('table', { tableData, opts }, 0, 0);
      this.children.push(node);
      return node;
    },
    addChart: function(chartType, data, opts = {}) {
      realSlide.addChart(chartType, data, opts);
    },
    render: function() {
      this.children.forEach(child => flattenNode(child, realSlide, pres));
    }
  };
  return virtualSlide;
};

// ============================================================
// 第1页 - 封面
// ============================================================
let slide = pres.addSlide();
slide.background = { color: COLORS.primary };

// 装饰线条
slide.addShape(pres.shapes.LINE, {
  x: CENTER_X - 2, y: 1.8, w: 4, h: 0,
  line: { color: COLORS.accent, width: 3 }
});

// 大标题
slide.addText("基于知识图谱与向量检索协同的地震应急问答方法研究", {
  x: 0.5, y: 2.0, w: 9, h: 1.2,
  fontSize: 32, fontFace: "Cambria", color: COLORS.white,
  bold: true, align: "center", valign: "middle",
  charSpacing: 1.5
});

// 副标题
slide.addText("硕士论文答辩", {
  x: 0.5, y: 3.3, w: 9, h: 0.5,
  fontSize: 20, fontFace: "Georgia", color: COLORS.accent,
  align: "center", valign: "middle",
  charSpacing: 1
});

// 作者信息
slide.addText("答辩人：刘一江", {
  x: 0.5, y: 4.0, w: 9, h: 0.4,
  fontSize: 16, fontFace: "Calibri", color: COLORS.white,
  align: "center"
});
slide.addText("导师：何高奇 教授", {
  x: 0.5, y: 4.4, w: 9, h: 0.4,
  fontSize: 16, fontFace: "Calibri", color: COLORS.white,
  align: "center"
});
slide.addText("华东师范大学 计算机学院", {
  x: 0.5, y: 4.8, w: 9, h: 0.4,
  fontSize: 14, fontFace: "Calibri", color: COLORS.light,
  align: "center"
});
slide.addText("2026年3月", {
  x: 0.5, y: 5.1, w: 9, h: 0.3,
  fontSize: 12, fontFace: "Calibri", color: COLORS.light,
  align: "center"
});

slide.render();

// ============================================================
// 第2页 - 目录
// ============================================================
slide = pres.addSlide();
slide.background = { color: COLORS.light };

// 标题
slide.addText("目录", {
  x: CONTENT_X, y: 0.4, w: CONTENT_W, h: 0.8,
  fontSize: 36, fontFace: "Cambria", color: COLORS.primary,
  bold: true, align: "center",
  charSpacing: 2.5
});

// 目录项
const tocItems = [
  { num: "01", title: "研究背景与问题", desc: "地震应急问答现状分析" },
  { num: "02", title: "核心创新点", desc: "四大技术创新方案" },
  { num: "03", title: "系统实现", desc: "技术架构与实现细节" },
  { num: "04", title: "移动端应用", desc: "安卓App功能与设计" },
  { num: "05", title: "实验验证与结论", desc: "实验结果与工作总结" }
];

const tocStartY = 1.2;
const tocItemH = 0.75;
const tocGap = 0.1;

tocItems.forEach((item, i) => {
  const y = tocStartY + i * (tocItemH + tocGap);
  
  // 序号圆圈
  slide.addShape(pres.shapes.OVAL, {
    x: 1.5, y: y + 0.1, w: 0.7, h: 0.7,
    fill: { color: COLORS.accent }
  });
  slide.addText(item.num, {
    x: 1.5, y: y + 0.1, w: 0.7, h: 0.7,
    fontSize: 18, fontFace: "Calibri", color: COLORS.white,
    bold: true, align: "center", valign: "middle"
  });
  
  // 标题
  slide.addText(item.title, {
    x: 2.5, y: y + 0.15, w: 4, h: 0.4,
    fontSize: 22, fontFace: "Cambria", color: COLORS.text,
    bold: true
  });
  
  // 描述
  slide.addText(item.desc, {
    x: 2.5, y: y + 0.55, w: 4, h: 0.3,
    fontSize: 14, fontFace: "Calibri", color: COLORS.textLight
  });
});

slide.render();

// ============================================================
// 第3页 - 研究背景
// ============================================================
slide = pres.addSlide();
slide.background = { color: COLORS.white };

// 标题
slide.addText("研究背景与问题", {
  x: CONTENT_X, y: 0.3, w: CONTENT_W, h: 0.7,
  fontSize: 32, fontFace: "Cambria", color: COLORS.primary,
  bold: true,
  charSpacing: 1.5
});

// 左侧：痛点列表
const painPoints = [
  { icon: "⚡", text: "传统方法响应慢、准确率低" },
  { icon: "❌", text: "大模型存在事实幻觉" },
  { icon: "📚", text: "单一知识源覆盖不足" }
];

painPoints.forEach((point, i) => {
  const y = 1.3 + i * 1.1;
  
  // 图标圆圈
  slide.addShape(pres.shapes.OVAL, {
    x: 0.8, y: y, w: 0.6, h: 0.6,
    fill: { color: COLORS.secondary }
  });
  slide.addText(point.icon, {
    x: 0.8, y: y, w: 0.6, h: 0.6,
    fontSize: 20, align: "center", valign: "middle"
  });
  
  // 痛点文字
  slide.addText(point.text, {
    x: 1.6, y: y + 0.1, w: 3.5, h: 0.5,
    fontSize: 16, fontFace: "Calibri", color: COLORS.text
  });
});

// 右侧：解决方案
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 5.5, y: 1.2, w: 3.8, h: 3.5,
  fill: { color: COLORS.light }, rectRadius: 0.15
});

slide.addText("解决方案", {
  x: 5.7, y: 1.4, w: 3.4, h: 0.5,
  fontSize: 20, fontFace: "Cambria", color: COLORS.accent,
  bold: true
});

slide.addText([
  { text: "知识图谱 + RAG协同", options: { bullet: true, breakLine: true } },
  { text: "结构化事实 + 语义检索", options: { bullet: true, breakLine: true } },
  { text: "提升准确性与完整性", options: { bullet: true } }
], {
  x: 5.7, y: 2.0, w: 3.4, h: 2.0,
  fontSize: 14, fontFace: "Calibri", color: COLORS.text
});

slide.render();

// ============================================================
// 第4页 - 核心创新点总览
// ============================================================
slide = pres.addSlide();
slide.background = { color: COLORS.light };

// 标题
slide.addText("核心创新点", {
  x: CONTENT_X, y: 0.3, w: CONTENT_W, h: 0.7,
  fontSize: 32, fontFace: "Cambria", color: COLORS.primary,
  bold: true, align: "center",
  charSpacing: 1.5
});

// 4个卡片
const innovations = [
  { num: "01", title: "知识图谱与RAG协同", desc: "结构化与语义检索融合" },
  { num: "02", title: "跨平台移动端应用", desc: "安卓应急App开发" },
  { num: "03", title: "左侧截断保尾策略", desc: "优化提示词结构" },
  { num: "04", title: "完整离线部署方案", desc: "端到端技术栈实现" }
];

const cardW = 2.0;
const cardH = 3.0;
const cardGap = 0.2;
const startX = (SLIDE_W - (4 * cardW + 3 * cardGap)) / 2;

innovations.forEach((item, i) => {
  const x = startX + i * (cardW + cardGap);
  const y = 1.3;
  
  // 卡片背景
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: x, y: y, w: cardW, h: cardH,
    fill: { color: COLORS.white }, rectRadius: 0.1,
    shadow: { type: "outer", blur: 5, offset: 2, color: "000000", opacity: 0.1 }
  });
  
  // 顶部装饰条
  slide.addShape(pres.shapes.RECTANGLE, {
    x: x, y: y, w: cardW, h: 0.15,
    fill: { color: COLORS.accent }
  });
  
  // 序号
  slide.addText(item.num, {
    x: x, y: y + 0.4, w: cardW, h: 0.6,
    fontSize: 36, fontFace: "Calibri", color: COLORS.accent,
    bold: true, align: "center"
  });
  
  // 标题
  slide.addText(item.title, {
    x: x + 0.2, y: y + 1.1, w: cardW - 0.4, h: 0.6,
    fontSize: 16, fontFace: "Cambria", color: COLORS.primary,
    bold: true, align: "center"
  });
  
  // 描述
  slide.addText(item.desc, {
    x: x + 0.2, y: y + 1.8, w: cardW - 0.4, h: 0.8,
    fontSize: 12, fontFace: "Calibri", color: COLORS.textLight,
    align: "center"
  });
});

slide.render();

// ============================================================
// 第5页 - 创新点1详解
// ============================================================
slide = pres.addSlide();
slide.background = { color: COLORS.white };

// 标题
slide.addText("创新点1 - 知识图谱与RAG协同机制", {
  x: CONTENT_X, y: 0.3, w: CONTENT_W, h: 0.7,
  fontSize: 28, fontFace: "Cambria", color: COLORS.primary,
  bold: true,
  charSpacing: 1.5
});

// 左侧：核心思想
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 0.5, y: 1.2, w: 4.2, h: 2.0,
  fill: { color: COLORS.light }, rectRadius: 0.1
});

slide.addText("核心思想", {
  x: 0.7, y: 1.4, w: 3.8, h: 0.4,
  fontSize: 18, fontFace: "Cambria", color: COLORS.accent,
  bold: true
});

slide.addText([
  { text: "知识图谱提供结构化事实", options: { bullet: true, breakLine: true } },
  { text: "RAG补充语义表述能力", options: { bullet: true, breakLine: true } },
  { text: "双源协同提升回答质量", options: { bullet: true } }
], {
  x: 0.7, y: 1.9, w: 3.8, h: 1.2,
  fontSize: 14, fontFace: "Calibri", color: COLORS.text
});

// 右侧：流程图（简化版）
const flowY = 1.2;
const nodeW = 1.6;
const nodeH = 0.6;
const nodeGap = 0.3;

// 用户问题
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 5.2, y: flowY, w: nodeW, h: nodeH,
  fill: { color: COLORS.primary }, rectRadius: 0.1
});
slide.addText("用户问题", {
  x: 5.2, y: flowY, w: nodeW, h: nodeH,
  fontSize: 12, fontFace: "Calibri", color: COLORS.white,
  bold: true, align: "center", valign: "middle"
});

// 双分支节点
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 3.6, y: flowY + nodeH + nodeGap, w: nodeW, h: nodeH,
  fill: { color: COLORS.secondary }, rectRadius: 0.1
});
slide.addText("图谱查询", {
  x: 3.6, y: flowY + nodeH + nodeGap, w: nodeW, h: nodeH,
  fontSize: 12, fontFace: "Calibri", color: COLORS.white,
  bold: true, align: "center", valign: "middle"
});

slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 6.8, y: flowY + nodeH + nodeGap, w: nodeW, h: nodeH,
  fill: { color: COLORS.secondary }, rectRadius: 0.1
});
slide.addText("向量检索", {
  x: 6.8, y: flowY + nodeH + nodeGap, w: nodeW, h: nodeH,
  fontSize: 12, fontFace: "Calibri", color: COLORS.white,
  bold: true, align: "center", valign: "middle"
});

// 融合生成
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 5.2, y: flowY + (nodeH + nodeGap) * 2, w: nodeW, h: nodeH,
  fill: { color: COLORS.accent }, rectRadius: 0.1
});
slide.addText("融合生成", {
  x: 5.2, y: flowY + (nodeH + nodeGap) * 2, w: nodeW, h: nodeH,
  fontSize: 12, fontFace: "Calibri", color: COLORS.white,
  bold: true, align: "center", valign: "middle"
});

// 底部关键数据
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 0.5, y: 4.2, w: 9, h: 0.6,
  fill: { color: COLORS.primary }, rectRadius: 0.1
});
slide.addText("关键数据：图谱提供结构化事实，RAG补充语义表述", {
  x: 0.5, y: 4.2, w: 9, h: 0.6,
  fontSize: 14, fontFace: "Calibri", color: COLORS.white,
  bold: true, align: "center", valign: "middle"
});

slide.render();

// ============================================================
// 第6页 - 创新点2详解（安卓App）
// ============================================================
slide = pres.addSlide();
slide.background = { color: COLORS.white };

// 标题
slide.addText("创新点2 - 跨平台移动端应急应用", {
  x: CONTENT_X, y: 0.3, w: CONTENT_W, h: 0.7,
  fontSize: 28, fontFace: "Cambria", color: COLORS.primary,
  bold: true,
  charSpacing: 1.5
});

// 左侧：6个功能模块卡片（2x3网格）
const appFeatures = [
  { icon: "\uD83C\uDFE0", name: "AI智能问答", desc: "对接后端LLM，地震应急知识实时对话" },
  { icon: "\uD83D\uDC65", name: "家庭组管理", desc: "成员安全状态追踪（安全/警告/危险/未知）" },
  { icon: "\uD83D\uDE98", name: "紧急求助", desc: "一键发起求助（救援/医疗/物资/避难/其他）" },
  { icon: "\uD83D\uDCDD", name: "灾情上报", desc: "多类型灾害上报（地震/火灾/洪水/滑坡）" },
  { icon: "\uD83D\uDEE1\uFE0F", name: "安全状态", desc: "实时更新个人安全等级与位置" },
  { icon: "\uD83E\uDD1D", name: "志愿者招募", desc: "技能标签匹配，就近调度" }
];

const appCardW = 2.15;
const appCardH = 1.15;
const appGapX = 0.15;
const appGapY = 0.15;
const appStartX = 0.4;
const appStartY = 1.15;

appFeatures.forEach((feat, i) => {
  const row = Math.floor(i / 2);
  const col = i % 2;
  const x = appStartX + col * (appCardW + appGapX);
  const y = appStartY + row * (appCardH + appGapY);

  // 卡片背景
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: x, y: y, w: appCardW, h: appCardH,
    fill: { color: COLORS.light }, rectRadius: 0.08,
    shadow: { type: "outer", blur: 3, offset: 1, color: "000000", opacity: 0.08 }
  });

  // 左侧装饰条
  slide.addShape(pres.shapes.RECTANGLE, {
    x: x, y: y, w: 0.06, h: appCardH,
    fill: { color: COLORS.accent }
  });

  // 图标
  slide.addText(feat.icon, {
    x: x + 0.12, y: y + 0.1, w: 0.4, h: 0.4,
    fontSize: 16, align: "center", valign: "middle"
  });

  // 功能名称
  slide.addText(feat.name, {
    x: x + 0.5, y: y + 0.08, w: appCardW - 0.65, h: 0.35,
    fontSize: 12, fontFace: "Cambria", color: COLORS.primary,
    bold: true
  });

  // 功能描述
  slide.addText(feat.desc, {
    x: x + 0.12, y: y + 0.5, w: appCardW - 0.24, h: 0.55,
    fontSize: 9, fontFace: "Calibri", color: COLORS.textLight
  });
});

// 右侧：技术栈
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 5.0, y: 1.15, w: 4.6, h: 4.0,
  fill: { color: COLORS.light }, rectRadius: 0.1
});

slide.addText("技术栈", {
  x: 5.2, y: 1.3, w: 4.2, h: 0.5,
  fontSize: 20, fontFace: "Cambria", color: COLORS.accent,
  bold: true
});

const techItems = [
  "Kotlin + Jetpack Compose",
  "Material 3 设计",
  "MVVM架构",
  "Retrofit2 + OkHttp",
  "Navigation Compose",
  "亮色/暗色主题"
];

techItems.forEach((tech, i) => {
  const y = 1.95 + i * 0.5;

  // 技术项背景
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 5.3, y: y, w: 4.0, h: 0.4,
    fill: { color: COLORS.white }, rectRadius: 0.06
  });

  // 小圆点
  slide.addShape(pres.shapes.OVAL, {
    x: 5.45, y: y + 0.1, w: 0.2, h: 0.2,
    fill: { color: COLORS.secondary }
  });

  // 技术名称
  slide.addText(tech, {
    x: 5.75, y: y, w: 3.4, h: 0.4,
    fontSize: 13, fontFace: "Calibri", color: COLORS.text,
    valign: "middle"
  });
});

slide.render();

// ============================================================
// 第7页 - 创新点3详解
// ============================================================
slide = pres.addSlide();
slide.background = { color: COLORS.white };

// 标题
slide.addText("创新点3 - 左侧截断保尾策略", {
  x: CONTENT_X, y: 0.3, w: CONTENT_W, h: 0.7,
  fontSize: 28, fontFace: "Cambria", color: COLORS.primary,
  bold: true,
  charSpacing: 1.5
});

// 左侧：传统右侧截断
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 0.5, y: 1.2, w: 4.2, h: 3.5,
  fill: { color: "FEE2E2" }, rectRadius: 0.1
});

slide.addText("传统右侧截断", {
  x: 0.7, y: 1.4, w: 3.8, h: 0.5,
  fontSize: 18, fontFace: "Cambria", color: "DC2626",
  bold: true
});

slide.addText([
  { text: "用户问题：地震发生时如何自救？", options: { breakLine: true } },
  { text: "上下文：[大量背景知识]...", options: { breakLine: true } },
  { text: "截断后：地震发生时如[截断]", options: { breakLine: true } },
  { text: "问题：核心问题被截掉！", options: { breakLine: true } }
], {
  x: 0.7, y: 2.0, w: 3.8, h: 2.0,
  fontSize: 14, fontFace: "Calibri", color: COLORS.text
});

// 红色X标记
slide.addShape(pres.shapes.OVAL, {
  x: 2.2, y: 3.8, w: 0.8, h: 0.8,
  fill: { color: "DC2626" }
});
slide.addText("❌", {
  x: 2.2, y: 3.8, w: 0.8, h: 0.8,
  fontSize: 32, color: COLORS.white, align: "center", valign: "middle"
});

// 右侧：左侧截断保尾
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 5.3, y: 1.2, w: 4.2, h: 3.5,
  fill: { color: "D1FAE5" }, rectRadius: 0.1
});

slide.addText("左侧截断保尾", {
  x: 5.5, y: 1.4, w: 3.8, h: 0.5,
  fontSize: 18, fontFace: "Cambria", color: "059669",
  bold: true
});

slide.addText([
  { text: "用户问题：地震发生时如何自救？", options: { breakLine: true } },
  { text: "上下文：[大量背景知识]...", options: { breakLine: true } },
  { text: "截断后：[截断]...如何自救？", options: { breakLine: true } },
  { text: "优势：核心问题始终保留！", options: { breakLine: true } }
], {
  x: 5.5, y: 2.0, w: 3.8, h: 2.0,
  fontSize: 14, fontFace: "Calibri", color: COLORS.text
});

// 绿色对勾
slide.addShape(pres.shapes.OVAL, {
  x: 7.0, y: 3.8, w: 0.8, h: 0.8,
  fill: { color: "059669" }
});
slide.addText("✓", {
  x: 7.0, y: 3.8, w: 0.8, h: 0.8,
  fontSize: 40, color: COLORS.white, align: "center", valign: "middle"
});

// 底部说明
slide.addText("核心策略：将用户问题固定置于提示末尾，确保上下文截断时问题信息不丢失", {
  x: 0.5, y: 4.9, w: 9, h: 0.5,
  fontSize: 14, fontFace: "Calibri", color: COLORS.textLight,
  align: "center"
});

slide.render();

// ============================================================
// 第8页 - 创新点4详解
// ============================================================
slide = pres.addSlide();
slide.background = { color: COLORS.light };

// 标题
slide.addText("创新点4 - 完整离线部署方案", {
  x: CONTENT_X, y: 0.3, w: CONTENT_W, h: 0.7,
  fontSize: 28, fontFace: "Cambria", color: COLORS.primary,
  bold: true,
  charSpacing: 1.5
});

// 技术栈网格
const techStack = [
  { name: "Neo4j", desc: "知识图谱", color: COLORS.primary },
  { name: "Milvus", desc: "向量库", color: COLORS.secondary },
  { name: "Qwen1.5-1.8B", desc: "LLM + LoRA", color: COLORS.accent },
  { name: "BGE", desc: "嵌入模型", color: COLORS.primary },
  { name: "Flask", desc: "后端服务", color: COLORS.secondary },
  { name: "Vue 3", desc: "前端框架", color: COLORS.accent }
];

const techCardW = 2.5;
const techCardH = 1.5;
const techGapX = 0.5;
const techGapY = 0.4;
const techStartX = (SLIDE_W - (3 * techCardW + 2 * techGapX)) / 2;
const techStartY = 1.3;

techStack.forEach((tech, i) => {
  const row = Math.floor(i / 3);
  const col = i % 3;
  const x = techStartX + col * (techCardW + techGapX);
  const y = techStartY + row * (techCardH + techGapY);
  
  // 卡片背景
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: x, y: y, w: techCardW, h: techCardH,
    fill: { color: COLORS.white }, rectRadius: 0.1,
    shadow: { type: "outer", blur: 4, offset: 1, color: "000000", opacity: 0.1 }
  });
  
  // 左侧装饰条
  slide.addShape(pres.shapes.RECTANGLE, {
    x: x, y: y, w: 0.08, h: techCardH,
    fill: { color: tech.color }
  });
  
  // 技术名称
  slide.addText(tech.name, {
    x: x + 0.2, y: y + 0.3, w: techCardW - 0.4, h: 0.5,
    fontSize: 18, fontFace: "Cambria", color: tech.color,
    bold: true, align: "center"
  });
  
  // 描述
  slide.addText(tech.desc, {
    x: x + 0.2, y: y + 0.85, w: techCardW - 0.4, h: 0.4,
    fontSize: 12, fontFace: "Calibri", color: COLORS.textLight,
    align: "center"
  });
});

slide.render();

// ============================================================
// 第9页 - 系统架构
// ============================================================
slide = pres.addSlide();
slide.background = { color: COLORS.white };

// 标题
slide.addText("系统总体架构", {
  x: CONTENT_X, y: 0.3, w: CONTENT_W, h: 0.7,
  fontSize: 32, fontFace: "Cambria", color: COLORS.primary,
  bold: true,
  charSpacing: 1.5
});

// 三层架构
const layers = [
  { label: "展示层", sub: "Vue 3 SPA + Android App (Kotlin)", color: COLORS.primary },
  { label: "应用服务层", sub: "Flask API", color: COLORS.secondary },
  { label: "数据模型层", sub: "Neo4j + Milvus + LLM", color: COLORS.accent }
];

const layerW = 6;
const layerH = 0.8;
const layerGap = 0.5;
const layerStartX = (SLIDE_W - layerW) / 2;
const layerStartY = 1.2;

layers.forEach((layer, i) => {
  const y = layerStartY + i * (layerH + layerGap);
  
  // 层背景
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: layerStartX, y: y, w: layerW, h: layerH,
    fill: { color: layer.color }, rectRadius: 0.1,
    shadow: { type: "outer", blur: 5, offset: 2, color: "000000", opacity: 0.2 }
  });
  
  // 层名称
  slide.addText(layer.label, {
    x: layerStartX, y: y, w: layerW / 2, h: layerH,
    fontSize: 18, fontFace: "Cambria", color: COLORS.white,
    bold: true, align: "center", valign: "middle"
  });
  
  // 子说明
  slide.addText(layer.sub, {
    x: layerStartX + layerW / 2, y: y, w: layerW / 2, h: layerH,
    fontSize: 14, fontFace: "Calibri", color: COLORS.light,
    align: "center", valign: "middle"
  });
  
  // 连接箭头
  if (i < layers.length - 1) {
    slide.addShape(pres.shapes.LINE, {
      x: SLIDE_W / 2, y: y + layerH, w: 0, h: layerGap,
      line: { color: COLORS.textLight, width: 2, endArrowType: "triangle" }
    });
  }
});

// 底部说明
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 0.5, y: 4.8, w: 9, h: 0.6,
  fill: { color: COLORS.light }, rectRadius: 0.1
});

slide.addText("架构特点：前后端分离、微服务化设计、支持离线部署", {
  x: 0.5, y: 4.8, w: 9, h: 0.6,
  fontSize: 12, fontFace: "Calibri", color: COLORS.text,
  align: "center", valign: "middle"
});

slide.render();

// ============================================================
// 第10页 - 知识图谱设计
// ============================================================
slide = pres.addSlide();
slide.background = { color: COLORS.light };

// 标题
slide.addText("知识图谱模式设计", {
  x: CONTENT_X, y: 0.3, w: CONTENT_W, h: 0.7,
  fontSize: 32, fontFace: "Cambria", color: COLORS.primary,
  bold: true,
  charSpacing: 1.5
});

// 实体节点
const entities = [
  { name: "Earthquake", label: "地震事件", x: 1.5, y: 1.5, color: COLORS.primary },
  { name: "Region", label: "区域", x: 6.5, y: 1.5, color: COLORS.secondary },
  { name: "EmergencyTopic", label: "应急主题", x: 1.5, y: 3.5, color: COLORS.accent },
  { name: "GuidanceStep", label: "处置步骤", x: 6.5, y: 3.5, color: COLORS.primary }
];

// 关系
const relations = [
  { from: 0, to: 1, label: "OCCURRED_IN" },
  { from: 2, to: 3, label: "HAS_STEP" },
  { from: 1, to: 2, label: "SUGGESTS_TOPIC" }
];

// 绘制关系线
relations.forEach(rel => {
  const from = entities[rel.from];
  const to = entities[rel.to];
  const nodeW = 2.0;
  const nodeH = 0.8;
  
  // 计算连接点
  let x1 = from.x + nodeW / 2;
  let y1 = from.y + nodeH / 2;
  let x2 = to.x + nodeW / 2;
  let y2 = to.y + nodeH / 2;
  
  // 调整起点终点
  if (Math.abs(x1 - x2) > Math.abs(y1 - y2)) {
    // 水平连接
    if (x1 < x2) { x1 += nodeW / 2; x2 -= nodeW / 2; }
    else { x1 -= nodeW / 2; x2 += nodeW / 2; }
  } else {
    // 垂直连接
    if (y1 < y2) { y1 += nodeH / 2; y2 -= nodeH / 2; }
    else { y1 -= nodeH / 2; y2 += nodeH / 2; }
  }
  
  slide.addShape(pres.shapes.LINE, {
    x: x1, y: y1, w: x2 - x1, h: y2 - y1,
    line: { color: COLORS.textLight, width: 2, endArrowType: "triangle" }
  });
  
  // 关系标签
  slide.addText(rel.label, {
    x: (x1 + x2) / 2 - 0.8, y: (y1 + y2) / 2 - 0.2, w: 1.6, h: 0.3,
    fontSize: 10, fontFace: "Calibri", color: COLORS.textLight,
    align: "center", valign: "middle"
  });
});

// 绘制实体节点
entities.forEach(ent => {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: ent.x, y: ent.y, w: 2.0, h: 0.8,
    fill: { color: ent.color }, rectRadius: 0.1,
    shadow: { type: "outer", blur: 4, offset: 2, color: "000000", opacity: 0.2 }
  });
  
  slide.addText(ent.name, {
    x: ent.x, y: ent.y, w: 2.0, h: 0.45,
    fontSize: 12, fontFace: "Calibri", color: COLORS.white,
    bold: true, align: "center", valign: "middle"
  });
  
  slide.addText(ent.label, {
    x: ent.x, y: ent.y + 0.45, w: 2.0, h: 0.35,
    fontSize: 10, fontFace: "Calibri", color: COLORS.light,
    align: "center", valign: "middle"
  });
});

slide.render();

// ============================================================
// 第11页 - RAG检索流程
// ============================================================
slide = pres.addSlide();
slide.background = { color: COLORS.white };

// 标题
slide.addText("RAG向量检索流程", {
  x: CONTENT_X, y: 0.3, w: CONTENT_W, h: 0.7,
  fontSize: 32, fontFace: "Cambria", color: COLORS.primary,
  bold: true,
  charSpacing: 1.5
});

// 水平流程图
const flowSteps = [
  { label: "JSON文件", color: COLORS.primary },
  { label: "Topic分块", color: COLORS.secondary },
  { label: "句向量编码", color: COLORS.accent },
  { label: "向量矩阵", color: COLORS.primary },
  { label: "相似度计算", color: COLORS.secondary },
  { label: "Top-K检索", color: COLORS.accent }
];

const stepW = 1.4;
const stepH = 0.8;
const stepGap = 0.2;
const flowStartX = 0.4;
const flowStartY = 2.0;

flowSteps.forEach((step, i) => {
  const x = flowStartX + i * (stepW + stepGap);
  
  // 步骤框
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: x, y: flowStartY, w: stepW, h: stepH,
    fill: { color: step.color }, rectRadius: 0.1,
    shadow: { type: "outer", blur: 3, offset: 1, color: "000000", opacity: 0.2 }
  });
  
  // 步骤文字
  slide.addText(step.label, {
    x: x, y: flowStartY, w: stepW, h: stepH,
    fontSize: 11, fontFace: "Calibri", color: COLORS.white,
    bold: true, align: "center", valign: "middle"
  });
  
  // 连接箭头
  if (i < flowSteps.length - 1) {
    slide.addShape(pres.shapes.LINE, {
      x: x + stepW, y: flowStartY + stepH / 2, w: stepGap, h: 0,
      line: { color: COLORS.textLight, width: 1.5, endArrowType: "triangle" }
    });
  }
});

// 流程说明
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 0.5, y: 3.5, w: 9, h: 1.5,
  fill: { color: COLORS.light }, rectRadius: 0.1
});

slide.addText([
  { text: "流程说明：", options: { bold: true, breakLine: true } },
  { text: "1. 从JSON文件读取地震应急知识", options: { breakLine: true } },
  { text: "2. 按主题分块处理文本内容", options: { breakLine: true } },
  { text: "3. 使用BGE模型编码为向量", options: { breakLine: true } },
  { text: "4. 存储至Milvus向量数据库，支持相似度检索" }
], {
  x: 0.7, y: 3.7, w: 8.6, h: 1.2,
  fontSize: 12, fontFace: "Calibri", color: COLORS.text
});

slide.render();

// ============================================================
// 第12页 - 实验结果
// ============================================================
slide = pres.addSlide();
slide.background = { color: COLORS.light };

// 标题
slide.addText("实验结果对比", {
  x: CONTENT_X, y: 0.3, w: CONTENT_W, h: 0.7,
  fontSize: 32, fontFace: "Cambria", color: COLORS.primary,
  bold: true,
  charSpacing: 1.5
});

// 柱状图
const chartData = [{
  name: "事实一致性准确率",
  labels: ["B0仅LLM", "B1仅图谱", "B2仅RAG", "B3全协同"],
  values: [52.3, 85.7, 68.4, 91.2]
}];

slide.addChart(pres.charts.BAR, chartData, {
  x: 1, y: 1.2, w: 8, h: 3.5, barDir: "col",
  chartColors: [COLORS.accent],
  chartArea: { fill: { color: COLORS.white }, roundedCorners: true },
  catAxisLabelColor: COLORS.text,
  valAxisLabelColor: COLORS.text,
  valGridLine: { color: "E2E8F0", size: 0.5 },
  catGridLine: { style: "none" },
  showValue: true,
  dataLabelPosition: "outEnd",
  dataLabelColor: COLORS.text,
  showLegend: false,
  valAxisMaxVal: 100
});

// 底部标注
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 0.5, y: 4.8, w: 9, h: 0.6,
  fill: { color: COLORS.primary }, rectRadius: 0.1
});

slide.addText("全协同方案在三项指标上均最优，准确率达91.2%", {
  x: 0.5, y: 4.8, w: 9, h: 0.6,
  fontSize: 14, fontFace: "Calibri", color: COLORS.white,
  bold: true, align: "center", valign: "middle"
});

slide.render();

// ============================================================
// 第13页 - 实验结论
// ============================================================
slide = pres.addSlide();
slide.background = { color: COLORS.white };

// 标题
slide.addText("关键结论", {
  x: CONTENT_X, y: 0.3, w: CONTENT_W, h: 0.7,
  fontSize: 32, fontFace: "Cambria", color: COLORS.primary,
  bold: true,
  charSpacing: 1.5
});

// 3个结论
const conclusions = [
  { num: "1", text: "知识图谱提供可验证事实，解决大模型幻觉问题" },
  { num: "2", text: "RAG补充语义表述，提升回答完整性" },
  { num: "3", text: "协同机制优于单一方案，全协同达91.2%准确率" }
];

conclusions.forEach((con, i) => {
  const y = 1.3 + i * 1.2;
  
  // 数字圆圈
  slide.addShape(pres.shapes.OVAL, {
    x: 1.0, y: y, w: 0.8, h: 0.8,
    fill: { color: COLORS.accent }
  });
  slide.addText(con.num, {
    x: 1.0, y: y, w: 0.8, h: 0.8,
    fontSize: 24, fontFace: "Calibri", color: COLORS.white,
    bold: true, align: "center", valign: "middle"
  });
  
  // 结论文字
  slide.addText(con.text, {
    x: 2.0, y: y + 0.15, w: 7, h: 0.6,
    fontSize: 18, fontFace: "Calibri", color: COLORS.text
  });
});

slide.render();

// ============================================================
// 第14页 - 总结与展望
// ============================================================
slide = pres.addSlide();
slide.background = { color: COLORS.light };

// 标题
slide.addText("总结与展望", {
  x: CONTENT_X, y: 0.3, w: CONTENT_W, h: 0.7,
  fontSize: 32, fontFace: "Cambria", color: COLORS.primary,
  bold: true,
  charSpacing: 1.5
});

// 左侧：工作总结
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 0.5, y: 1.2, w: 4.3, h: 3.8,
  fill: { color: COLORS.white }, rectRadius: 0.1,
  shadow: { type: "outer", blur: 4, offset: 2, color: "000000", opacity: 0.1 }
});

slide.addText("工作总结", {
  x: 0.7, y: 1.4, w: 3.9, h: 0.5,
  fontSize: 20, fontFace: "Cambria", color: COLORS.primary,
  bold: true
});

slide.addText([
  { text: "提出知识图谱与RAG协同机制", options: { bullet: true, breakLine: true } },
  { text: "设计左侧截断保尾策略", options: { bullet: true, breakLine: true } },
  { text: "实现完整离线部署方案", options: { bullet: true, breakLine: true } },
  { text: "实验验证达到91.2%准确率", options: { bullet: true } }
], {
  x: 0.7, y: 2.0, w: 3.9, h: 2.8,
  fontSize: 14, fontFace: "Calibri", color: COLORS.text
});

// 右侧：未来展望
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 5.2, y: 1.2, w: 4.3, h: 3.8,
  fill: { color: COLORS.white }, rectRadius: 0.1,
  shadow: { type: "outer", blur: 4, offset: 2, color: "000000", opacity: 0.1 }
});

slide.addText("未来展望", {
  x: 5.4, y: 1.4, w: 3.9, h: 0.5,
  fontSize: 20, fontFace: "Cambria", color: COLORS.accent,
  bold: true
});

slide.addText([
  { text: "扩展多模态知识融合", options: { bullet: true, breakLine: true } },
  { text: "优化实时响应性能", options: { bullet: true, breakLine: true } },
  { text: "推广至其他应急领域", options: { bullet: true } }
], {
  x: 5.4, y: 2.0, w: 3.9, h: 2.8,
  fontSize: 14, fontFace: "Calibri", color: COLORS.text
});

slide.render();

// ============================================================
// 第15页 - 结束页
// ============================================================
slide = pres.addSlide();
slide.background = { color: COLORS.primary };

// 装饰线条
slide.addShape(pres.shapes.LINE, {
  x: CENTER_X - 1.5, y: 2.0, w: 3, h: 0,
  line: { color: COLORS.accent, width: 3 }
});

// 标题
slide.addText("谢谢聆听！", {
  x: 0.5, y: 2.3, w: 9, h: 1.0,
  fontSize: 44, fontFace: "Cambria", color: COLORS.white,
  bold: true, align: "center", valign: "middle",
  charSpacing: 2.5
});

// 副标题
slide.addText("敬请批评指正", {
  x: 0.5, y: 3.5, w: 9, h: 0.5,
  fontSize: 20, fontFace: "Georgia", color: COLORS.light,
  align: "center"
});

// 作者信息
slide.addText("刘一江 | 华东师范大学 计算机学院", {
  x: 0.5, y: 4.5, w: 9, h: 0.4,
  fontSize: 14, fontFace: "Calibri", color: COLORS.light,
  align: "center"
});

slide.render();

// ============================================================
// 保存文件
// ============================================================
pres.writeFile({ fileName: "/sessions/69dc8c47c1ce118fe433f325/workspace/论文答辩PPT.pptx" });
console.log("PPT已生成：/sessions/69dc8c47c1ce118fe433f325/workspace/论文答辩PPT.pptx");
