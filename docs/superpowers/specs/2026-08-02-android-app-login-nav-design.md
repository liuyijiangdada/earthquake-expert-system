# 安卓 App：登录门控 + 导航重组设计

日期：2026-08-02  
状态：待实现  
范围：仅 `disaster_app/android`（方案 2）  
关联作废：`2026-08-02-mobile-ux-design.md`（Web 窄屏优化，对象选错）

## 目标

优化原生安卓应急 App（Jetpack Compose）的功能合理性，并增加与 Web 一致的登录能力：

1. 启动需登录（对接现有 PostgreSQL / `POST /api/auth/login`）
2. 登录页可配置服务器地址（模拟器默认 + 真机局域网）
3. 底栏由 6 Tab 精简为「问答 / 指引 / 更多」，其余演示功能进「更多」
4. 登录态用 EncryptedSharedPreferences；支持退出

## 非目标

- 不改 Web 前端响应式布局
- 不为 Flask 业务 API 强制鉴权 / 不上 JWT
- 不重写求助、家人、上报、志愿页的本地演示逻辑
- 不做注册、找回密码、角色权限
- 不引入 Hilt 等大重构

## 决策摘要（已确认）

| 项 | 选择 |
|---|---|
| 范围 | 只改安卓 App |
| 底栏 | 3 Tab：问答、指引、更多 |
| 登录 | 客户端门控 + EncryptedSharedPreferences + 退出 |
| 服务器地址 | 登录页可编辑，默认 `http://10.0.2.2:8000/` |
| 实现路径 | 登录 + 导航重组（方案 2） |

## 信息架构

### 启动流

```
冷启动
  → 读 AuthStore
  → 已登录？ → AppNavigation（3 Tab）
  → 否 → LoginPage
```

登录成功：校验 `POST {baseUrl}/api/auth/login` → 写入 `username`、`baseUrl`、`loggedInAt` → 进入主界面。  
退出：清除登录用户信息；可保留 `baseUrl`；回到 LoginPage。

### 底栏

| Tab | 内容 |
|---|---|
| 问答 | 现有 `HomePage`（聊天 + 快捷提问 + 多模态） |
| 指引 | 现有 `SafetyPage` |
| 更多 | 新建 `MorePage`：列表入口 → 求助 / 家人 / 上报 / 志愿；可含用户信息与退出 |

未登录不展示底栏与业务页。

## 登录、API 与存储

### API

- 复用 Flask：`POST /api/auth/login` body `{username, password}` → `{ok, username}` 或 401/503
- 可选调用 `POST /api/auth/logout`
- `DisasterApiService` 增加 login；问答/多模态接口不变
- `ApiClient`：按已保存 `baseUrl` 动态创建/重建 Retrofit（改地址后生效）
- 业务请求不做 token 头（与 Web「仅前端门控」一致）

### AuthStore（EncryptedSharedPreferences）

键示例：

- `username`
- `baseUrl`
- `loggedInAt`（ISO-8601，可选）

退出：清除 `username` / `loggedInAt`；`baseUrl` 默认保留。

依赖：`androidx.security:security-crypto`。

### 登录页 UX

- 字段：服务器地址、用户名、密码、登录按钮
- 默认地址：`http://10.0.2.2:8000/`（模拟器访问宿主机）
- 校验：空字段前端拦截
- 错误文案：
  - 401 →「用户名或密码错误」
  - 网络失败 →「无法连接服务器，请检查地址与后端」
  - 503 →「认证服务暂不可用」
- 视觉：贴合现有深色应急主题（`BgDeep` / `AccentBlue` 等）

### 网络安全

- 演示环境允许 HTTP cleartext（模拟器 + 局域网 IP）
- 更新 `network_security_config.xml`：除 `localhost` 外，允许 cleartext 或对调试 build 放宽（以实现时可编译、真机可连为准）
- 生产 HTTPS：文档注明，本次不强制

## 组件与文件（预期）

| 文件 | 责任 |
|---|---|
| `app/build.gradle.kts` | security-crypto 依赖 |
| `data/api/ApiClient.kt` | 动态 baseUrl |
| `data/api/DisasterApiService.kt` | login 接口 |
| `data/auth/AuthStore.kt`（新建） | 加密偏好读写 |
| `ui/login/LoginPage.kt`（新建） | 登录 UI |
| `ui/more/MorePage.kt`（新建） | 更多入口 + 退出 |
| `viewmodel/AuthViewModel.kt`（新建） | 登录/退出状态 |
| `ui/navigation/AppNavigation.kt` | 门控 + 3 Tab + 子路由 |
| `MainActivity.kt` | 根组合：登录 vs 主界面 |
| `res/xml/network_security_config.xml` | cleartext 策略 |
| `HomePage` / 顶栏组件 | 可选：显示用户名、退出入口 |

命名以最终实现为准，职责不变。

## 错误与边界

- `baseUrl` 规范化：去尾部多余 `/`，保证拼接 `api/...` 正确
- Retrofit 单例在 baseUrl 变更时必须重建，避免仍打旧地址
- Postgres 未启动：登录 503，提示明确，不崩溃
- 旋转屏幕：登录表单状态可用 `rememberSaveable` 保留输入（可选增强）

## 验收

1. 未登录只能见登录页；默认账号 `admin` / `admin` + 正确地址可进主界面
2. 底栏仅「问答 / 指引 / 更多」；更多内可进入求助、家人、上报、志愿
3. 修改服务器地址并登录后，问答请求打到新地址
4. 退出后需重新登录；`baseUrl` 可保留预填
5. Debug 包可在模拟器编译安装；真机改局域网 IP 后可连本机后端（需后端监听 `0.0.0.0`）

## 实现备注

- 实现前另写 `docs/superpowers/plans/2026-08-02-android-app-login-nav.md`
- Web 端登录已存在，安卓只做客户端对接，不重复建用户表
