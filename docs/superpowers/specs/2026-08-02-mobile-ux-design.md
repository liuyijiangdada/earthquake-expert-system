# 移动端布局与交互优化设计

日期：2026-08-02  
状态：待实现  
范围选择：方案 2（移动壳层）；登录/鉴权不动

## 目标

在现有 Vue SPA 上优化 ≤991px 体验，使应急问答在手机上「对话优先、触控可用」：

1. 震情统计/图表默认折叠，对话占首屏
2. 快捷提问单击即发送
3. Composer 固定底部，并处理底部安全区

桌面（≥992px）布局与交互保持现状。

## 非目标

- 不改登录门控、PostgreSQL 鉴权、业务 API
- 不新建独立 Mobile 页面或 vue-router 分支
- 不改图表数据源与推理链路
- 不做注册/找回密码等账号增强

## 决策摘要（已确认）

| 项 | 选择 |
|---|---|
| 震情面板 | 窄屏默认折叠，点开再看 |
| 快捷提问 | 窄屏单击直接发送 |
| Composer | 窄屏固定底部 |
| 实现路径 | 现有组件 + 窄屏壳层，非纯 CSS、非双 UI |

## 信息架构（≤991px）

自上而下：

1. **精简顶栏**：标题 + 状态徽章 + 用户/退出；副标题隐藏；高度压缩
2. **对话主区**（占满剩余视口高度）：
   - 快捷提问
   - 消息列表（可滚动）
   - Composer（固定在对话壳底部）
3. **震情面板**：默认折叠为一行摘要；展开后显示 StatsBar + InsightChart

≥992px：保持 `col-lg-7` / `col-lg-5` 左右两栏，震情始终展开。

## 交互细节

### 快捷提问（QuickPanel）

- 断点：`matchMedia('(max-width: 991px)')`（与 Bootstrap `lg` 对齐）
- 窄屏：单击 chip → `submitQuick`（直接发送）
- 桌面：单击填入、双击 / ⌘·Ctrl+单击发送（现逻辑）
- 窄屏提示：「点选即可发送」；隐藏 ⌘/Ctrl 说明

### 固定底部 Composer

- 对话卡窄屏为 flex 列：消息区 `flex: 1; overflow-y: auto`，Composer sticky/固定底部
- Composer 背景不透明，避免气泡透出
- `padding-bottom: env(safe-area-inset-bottom)`（或等价），避免挡住 Home 指示条
- 新消息后仍自动滚到底（现有 watch 逻辑保留）

### 震情折叠

- 状态：`App.vue` 本地 `ref`（如 `statsCollapsed`），默认 `true` 于窄屏；桌面不渲染折叠条、始终展开
- 折叠条文案示例：`震情概览 · {total} 条 · 最大 M{max}` + 展开/收起图标
- 不持久化到 localStorage

### 顶栏

- 窄屏隐藏 `.subtitle`
- 标题字号略减；徽章与退出可换行但不抢首屏高度

## 组件与文件

| 文件 | 责任 |
|---|---|
| `frontend/src/App.vue` | 窄屏壳 class；震情折叠开关；布局结构 |
| `frontend/src/components/QuickPanel.vue` | 断点感知单击行为与提示文案 |
| `frontend/src/components/StatsBar.vue` | 可选：接收 `collapsed` / 摘要 props；或由 App 外包折叠条 |
| `frontend/src/components/Composer.vue` | 窄屏贴底相关 class（若需） |
| `frontend/src/components/AppHeader.vue` | 窄屏样式钩子（必要时） |
| `frontend/src/assets/main.css` | `@media (max-width: 991px)` 壳层、安全区、折叠条、顶栏 |

断点统一为 **991px**，与现有 Bootstrap 栅格一致。

## 错误与边界

- 震情数据未加载时，折叠摘要显示 `—` / `0`，不报错
- `matchMedia` 在 SSR/无窗口环境不存在：本项目为纯客户端 SPA，挂载后读取即可；可用 `onMounted` + `change` 监听以便旋转屏幕更新
- 桌面缩到窄屏：折叠默认开启；从窄屏拉宽到桌面：恢复两栏展开，不保留折叠 UI

## 测试 / 验收

1. 视口 ≤991px：首屏以对话为主，震情默认折叠；展开可见统计与图表
2. 点快捷 chip 直接发出用户消息并触发查询
3. Composer 贴底；底部安全区不挡发送按钮（有 Home 条的设备上目视确认）
4. ≥992px：左右两栏与现交互一致（单击填入、双击发送）
5. `npm run build` 成功；登录门控行为不变

## 实现备注

实现前另写 `docs/superpowers/plans/` 下的分步计划；构建产物仍输出到 `static/spa/`（沿用现有前端构建流程）。
