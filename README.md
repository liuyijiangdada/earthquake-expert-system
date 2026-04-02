# 地震智能问答（知识图谱 + RAG + 本地 LLM）

Flask 后端：`app.py`。默认访问地址见 `config/config.py` 中的 `DEPLOY_HOST`、`DEPLOY_PORT`（常见为 `http://127.0.0.1:8000`）。

更完整的数据流与模块说明见仓库内 **`process.md`**。

---

## 前端（Vue 3 + Vite）

源码目录：`frontend/`。生产环境由 Vite 构建到 **`static/spa/`**；`app.py` 在存在 `static/spa/index.html` 时优先提供该 SPA，否则回退 **`static/index.legacy.html`**（旧版单文件页）。

### 首次安装依赖

```bash
cd frontend
npm install
```

### 构建生产静态资源（`npm run build`）

在修改 Vue 源码后，若要通过 **仅启动 Flask** 使用新版界面，需要重新构建：

```bash
cd frontend
npm run build
```

- 产物输出目录：**`static/spa/`**（含 `index.html` 与 `assets/` 下的 JS/CSS）。
- 构建完成后在项目根目录执行 **`python app.py`**，浏览器访问根路径即可加载 Vue 页面。

### 本地开发（热更新 + API 代理）

终端 1：启动后端（需已配置 Neo4j、模型等）。

```bash
python app.py
```

终端 2：启动 Vite（默认 `http://127.0.0.1:5173`，`/api` 已代理到 `http://127.0.0.1:8000`）。

```bash
cd frontend
npm run dev
```

开发时用 **5173** 端口访问；与生产构建路径无关。

### 预览构建结果（可选）

```bash
cd frontend
npm run preview
```

用于本地检查 `npm run build` 后的包，不替代 Flask 托管。

---

## 后端启动（简要）

```bash
python app.py
```

依赖与离线模型缓存等要求见 **`process.md`** 第 2 节。
