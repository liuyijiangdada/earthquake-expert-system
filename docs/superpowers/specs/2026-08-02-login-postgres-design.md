# 登录页 + PostgreSQL 用户库（前端门控）设计

日期：2026-08-02  
状态：已实现  
范围选择：方案 A（仅前端门控，API 不强制鉴权）

## 目标

为地震应急问答 SPA 增加登录页。账号密码存 PostgreSQL，默认种子用户 `admin` / `admin`。未登录只显示登录页；登录成功后进入现有聊天界面。

## 非目标

- 不为 `/api/query` 等业务接口强制 401 鉴权（可后续升级）
- 不提供注册/找回密码/角色权限
- 不引入 vue-router、JWT、OAuth

## 架构

```
浏览器 Login.vue
  → POST /api/auth/login {username, password}
  → Flask 查 PostgreSQL users 表（password_hash 校验）
  → 成功：前端 localStorage 记登录态 → 渲染现有 App 聊天 UI
  → 退出：清 localStorage，回到登录页
```

## 基础设施

### docker-compose

新增服务 `postgres`：

- 镜像：`postgres:16-alpine`
- 端口：`5432:5432`
- 环境：`POSTGRES_USER=earthquake` / `POSTGRES_PASSWORD=earthquake` / `POSTGRES_DB=earthquake_qa`
- volume：`postgres_data`

### 配置

`config/config.py` + `.env.example`：

- `DATABASE_URL=postgresql://earthquake:earthquake@localhost:5432/earthquake_qa`
- 可选：`AUTH_DEFAULT_ADMIN_USER=admin`、`AUTH_DEFAULT_ADMIN_PASSWORD=admin`

### 依赖

`requirements.txt` 增加：`psycopg2-binary`

密码哈希使用 Flask 已间接可用的 `werkzeug.security`（generate_password_hash / check_password_hash）。

## 数据模型

表 `users`：

| 列 | 类型 | 说明 |
|---|---|---|
| id | SERIAL PRIMARY KEY | |
| username | VARCHAR(64) UNIQUE NOT NULL | |
| password_hash | VARCHAR(255) NOT NULL | werkzeug 哈希 |
| created_at | TIMESTAMPTZ DEFAULT now() | |

启动时：`CREATE TABLE IF NOT EXISTS`；若表中无任何用户，插入默认 `admin`（哈希后的密码）。

## 后端 API

新建 `services/auth_db.py`：连接池/连接、建表、种子用户、`verify_user(username, password) -> bool`。

`app.py` 增加：

- `POST /api/auth/login`：body `{username, password}` → `{ok:true, username}` 或 401
- `POST /api/auth/logout`：返回 `{ok:true}`（前端清状态；服务端无 session）
- `GET /api/auth/me`：可选；前端以 localStorage 为准，可不依赖

业务 API 不加 `before_request` 鉴权。

Postgres 不可用时：登录接口返回明确 503，不阻塞模型加载失败以外的启动（尽量延迟连库到首次 login / 模块 init 时重试）。

## 前端

- 新建 `frontend/src/components/Login.vue`：用户名、密码、提交、错误提示；风格贴合现有深色聊天 UI
- 改 `frontend/src/App.vue`：`isAuthenticated`（读 localStorage key，如 `eq_auth_user`）；未登录只渲染 `Login`；已登录渲染现有布局；顶栏增加退出
- 改 `frontend/src/api.js`：`login()` / `logout()`；`credentials` 非必须（无 cookie session）
- 构建：`npm run build` → `static/spa/`

本地登录态示例：

```json
{ "username": "admin", "loggedInAt": "ISO-8601" }
```

## 错误处理

- 空用户名/密码：前端拦截 + 后端 400
- 账号或密码错误：统一「用户名或密码错误」（不暴露哪个错）
- 数据库连不上：503 +「认证服务暂不可用」

## 测试

- 单元：密码哈希校验、种子用户逻辑（可用 mock 或测试库）
- 手工：compose 起 postgres → 启动 app → `admin/admin` 登录进聊天 → 退出回到登录页；错误密码提示

## 验收

1. `docker compose up -d postgres` 后，系统可登录
2. 默认 `admin` / `admin` 可进聊天页
3. 刷新仍保持登录（localStorage）
4. 退出后需重新登录才能见聊天 UI
5. 密码不以明文写入数据库
