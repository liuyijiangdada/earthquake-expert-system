# Android Login + Nav Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add login gate (Postgres-backed `/api/auth/login`) with configurable server URL, and reorganize bottom nav to 问答 / 指引 / 更多 inside `disaster_app/android`.

**Architecture:** EncryptedSharedPreferences holds `username` + `baseUrl`; `ApiClient` rebuilds Retrofit when base URL changes; `MainActivity` shows `LoginPage` or `AppNavigation` (3 tabs). Help/Family/Report/Volunteer stay as-is under More.

**Tech Stack:** Kotlin, Jetpack Compose, Navigation Compose, Retrofit, OkHttp, androidx.security:security-crypto, existing Flask auth API

## Global Constraints

- Scope: only `disaster_app/android/` (no Web responsive work, no Flask JWT middleware)
- Default base URL: `http://10.0.2.2:8000/`
- Auth is client-side gate only; business APIs remain unauthenticated
- Bottom bar exactly 3 items: 问答, 指引, 更多
- Logout clears username; keep baseUrl for next login
- Cleartext HTTP allowed for demo (emulator + LAN IP)
- Package root: `com.example.disaster_app`

---

## File map

| File | Responsibility |
|---|---|
| `app/build.gradle.kts` | Add `security-crypto` |
| `res/xml/network_security_config.xml` | Allow cleartext for demo (base-config) |
| `data/auth/BaseUrl.kt` | Pure `normalizeBaseUrl` |
| `data/auth/AuthStore.kt` | Encrypted prefs read/write |
| `data/model/AuthModels.kt` | Login request/response DTOs |
| `data/api/DisasterApiService.kt` | `login` endpoint |
| `data/api/ApiClient.kt` | Dynamic Retrofit + `configure(baseUrl)` |
| `data/repository/ChatRepository.kt` | Use live `ApiClient.apiService` each call |
| `viewmodel/AuthViewModel.kt` | Login/logout UI state |
| `ui/login/LoginPage.kt` | Login form UI |
| `ui/more/MorePage.kt` | Links + logout |
| `ui/navigation/AppNavigation.kt` | 3-tab + nested routes |
| `MainActivity.kt` | Auth gate root |
| `app/src/test/.../BaseUrlTest.kt` | Unit tests for normalize |

---

### Task 1: Base URL util + AuthStore + cleartext + dependency

**Files:**
- Create: `disaster_app/android/app/src/main/kotlin/com/example/disaster_app/data/auth/BaseUrl.kt`
- Create: `disaster_app/android/app/src/main/kotlin/com/example/disaster_app/data/auth/AuthStore.kt`
- Create: `disaster_app/android/app/src/test/java/com/example/disaster_app/data/auth/BaseUrlTest.kt`
- Modify: `disaster_app/android/app/build.gradle.kts`
- Modify: `disaster_app/android/app/src/main/res/xml/network_security_config.xml`

**Interfaces:**
- Produces: `fun normalizeBaseUrl(raw: String): String`
- Produces: `class AuthStore(context: Context)` with `isLoggedIn()`, `getUsername()`, `getBaseUrl()`, `saveSession(username, baseUrl)`, `clearSession(keepBaseUrl: Boolean = true)`, `DEFAULT_BASE_URL`

- [ ] **Step 1: Write failing unit test for normalizeBaseUrl**

```kotlin
package com.example.disaster_app.data.auth

import org.junit.Assert.assertEquals
import org.junit.Test

class BaseUrlTest {
    @Test
    fun trims_and_ensures_trailing_slash() {
        assertEquals("http://10.0.2.2:8000/", normalizeBaseUrl("http://10.0.2.2:8000"))
        assertEquals("http://10.0.2.2:8000/", normalizeBaseUrl("  http://10.0.2.2:8000/  "))
        assertEquals("http://192.168.1.8:8000/", normalizeBaseUrl("http://192.168.1.8:8000"))
    }

    @Test
    fun empty_falls_back_to_default() {
        assertEquals(AuthStore.DEFAULT_BASE_URL, normalizeBaseUrl(""))
        assertEquals(AuthStore.DEFAULT_BASE_URL, normalizeBaseUrl("   "))
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run from `disaster_app/android`:

```bash
./gradlew :app:testDebugUnitTest --tests com.example.disaster_app.data.auth.BaseUrlTest
```

Expected: FAIL (class/function missing) or compile error.

- [ ] **Step 3: Add dependency**

In `app/build.gradle.kts` dependencies block add:

```kotlin
implementation("androidx.security:security-crypto:1.1.0-alpha06")
```

- [ ] **Step 4: Implement BaseUrl + AuthStore**

`BaseUrl.kt`:

```kotlin
package com.example.disaster_app.data.auth

fun normalizeBaseUrl(raw: String): String {
    val trimmed = raw.trim()
    if (trimmed.isEmpty()) return AuthStore.DEFAULT_BASE_URL
    return if (trimmed.endsWith("/")) trimmed else "$trimmed/"
}
```

`AuthStore.kt`:

```kotlin
package com.example.disaster_app.data.auth

import android.content.Context
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import java.time.Instant

class AuthStore(context: Context) {
    companion object {
        const val DEFAULT_BASE_URL = "http://10.0.2.2:8000/"
        private const val PREFS = "eq_auth_secure"
        private const val KEY_USERNAME = "username"
        private const val KEY_BASE_URL = "baseUrl"
        private const val KEY_LOGGED_IN_AT = "loggedInAt"
    }

    private val prefs = EncryptedSharedPreferences.create(
        context.applicationContext,
        PREFS,
        MasterKey.Builder(context.applicationContext)
            .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
            .build(),
        EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
        EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
    )

    fun isLoggedIn(): Boolean = !getUsername().isNullOrBlank()

    fun getUsername(): String? = prefs.getString(KEY_USERNAME, null)?.takeIf { it.isNotBlank() }

    fun getBaseUrl(): String = normalizeBaseUrl(prefs.getString(KEY_BASE_URL, null) ?: DEFAULT_BASE_URL)

    fun saveSession(username: String, baseUrl: String) {
        prefs.edit()
            .putString(KEY_USERNAME, username.trim())
            .putString(KEY_BASE_URL, normalizeBaseUrl(baseUrl))
            .putString(KEY_LOGGED_IN_AT, Instant.now().toString())
            .apply()
    }

    fun clearSession(keepBaseUrl: Boolean = true) {
        val kept = if (keepBaseUrl) getBaseUrl() else DEFAULT_BASE_URL
        prefs.edit()
            .remove(KEY_USERNAME)
            .remove(KEY_LOGGED_IN_AT)
            .putString(KEY_BASE_URL, kept)
            .apply()
    }
}
```

- [ ] **Step 5: Relax cleartext for LAN demo**

Replace `network_security_config.xml` with:

```xml
<?xml version="1.0" encoding="utf-8"?>
<network-security-config>
    <!-- Demo only: allow HTTP to emulator/LAN backend -->
    <base-config cleartextTrafficPermitted="true" />
</network-security-config>
```

- [ ] **Step 6: Run unit tests**

```bash
./gradlew :app:testDebugUnitTest --tests com.example.disaster_app.data.auth.BaseUrlTest
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add disaster_app/android/app/build.gradle.kts \
  disaster_app/android/app/src/main/kotlin/com/example/disaster_app/data/auth/ \
  disaster_app/android/app/src/test/java/com/example/disaster_app/data/auth/BaseUrlTest.kt \
  disaster_app/android/app/src/main/res/xml/network_security_config.xml
git commit -m "$(cat <<'EOF'
feat(android): add AuthStore and base URL normalization

EOF
)"
```

---

### Task 2: Dynamic ApiClient + login API + ChatRepository live service

**Files:**
- Create: `disaster_app/android/app/src/main/kotlin/com/example/disaster_app/data/model/AuthModels.kt`
- Modify: `disaster_app/android/app/src/main/kotlin/com/example/disaster_app/data/api/DisasterApiService.kt`
- Modify: `disaster_app/android/app/src/main/kotlin/com/example/disaster_app/data/api/ApiClient.kt`
- Modify: `disaster_app/android/app/src/main/kotlin/com/example/disaster_app/data/repository/ChatRepository.kt`
- Create: `disaster_app/android/app/src/test/java/com/example/disaster_app/data/auth/ApiClientBaseUrlTest.kt` (optional pure check via normalize only — skip if flaky; at minimum compile `:app:compileDebugKotlin`)

**Interfaces:**
- Consumes: `normalizeBaseUrl`
- Produces: `ApiClient.configure(baseUrl: String)`, `ApiClient.apiService` (always current)
- Produces: `DisasterApiService.login(LoginRequest): LoginResponse`
- Produces: DTOs `LoginRequest`, `LoginResponse`

- [ ] **Step 1: Add AuthModels**

```kotlin
package com.example.disaster_app.data.model

data class LoginRequest(
    val username: String,
    val password: String
)

data class LoginResponse(
    val ok: Boolean? = null,
    val username: String? = null,
    val error: String? = null
)
```

- [ ] **Step 2: Extend DisasterApiService**

Add imports and method:

```kotlin
import com.example.disaster_app.data.model.LoginRequest
import com.example.disaster_app.data.model.LoginResponse
import retrofit2.Response
import retrofit2.http.POST

@POST("api/auth/login")
suspend fun login(@Body request: LoginRequest): Response<LoginResponse>
```

Keep existing `queryLlm` / `multimodalQuery` unchanged.

- [ ] **Step 3: Rewrite ApiClient for dynamic base URL**

```kotlin
package com.example.disaster_app.data.api

import com.example.disaster_app.data.auth.AuthStore
import com.example.disaster_app.data.auth.normalizeBaseUrl
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

object ApiClient {
    private val loggingInterceptor = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.BODY
    }

    private val okHttpClient = OkHttpClient.Builder()
        .addInterceptor(loggingInterceptor)
        .connectTimeout(60, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()

    @Volatile
    private var currentBaseUrl: String = AuthStore.DEFAULT_BASE_URL

    @Volatile
    private var retrofit: Retrofit = buildRetrofit(currentBaseUrl)

    @Volatile
    private var service: DisasterApiService = retrofit.create(DisasterApiService::class.java)

    val apiService: DisasterApiService
        get() = service

    @Synchronized
    fun configure(baseUrl: String) {
        val normalized = normalizeBaseUrl(baseUrl)
        if (normalized == currentBaseUrl && ::service.isInitialized) {
            // still rebuild if first configure after process start is fine; always rebuild for simplicity:
        }
        currentBaseUrl = normalized
        retrofit = buildRetrofit(normalized)
        service = retrofit.create(DisasterApiService::class.java)
    }

    private fun buildRetrofit(baseUrl: String): Retrofit =
        Retrofit.Builder()
            .baseUrl(baseUrl)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
}
```

Simplify the empty `if` in configure — final code should be:

```kotlin
@Synchronized
fun configure(baseUrl: String) {
    val normalized = normalizeBaseUrl(baseUrl)
    currentBaseUrl = normalized
    retrofit = buildRetrofit(normalized)
    service = retrofit.create(DisasterApiService::class.java)
}
```

- [ ] **Step 4: Fix ChatRepository to not cache stale service**

Replace:

```kotlin
private val apiService = ApiClient.apiService
```

with per-call:

```kotlin
private fun api() = ApiClient.apiService
```

and replace `apiService.` usages with `api().` in `sendMessage` / multimodal methods.

- [ ] **Step 5: Compile**

```bash
cd disaster_app/android && ./gradlew :app:compileDebugKotlin
```

Expected: BUILD SUCCESSFUL

- [ ] **Step 6: Commit**

```bash
git add disaster_app/android/app/src/main/kotlin/com/example/disaster_app/data/
git commit -m "$(cat <<'EOF'
feat(android): dynamic ApiClient and auth login endpoint

EOF
)"
```

---

### Task 3: AuthViewModel + LoginPage + MainActivity gate

**Files:**
- Create: `disaster_app/android/app/src/main/kotlin/com/example/disaster_app/viewmodel/AuthViewModel.kt`
- Create: `disaster_app/android/app/src/main/kotlin/com/example/disaster_app/ui/login/LoginPage.kt`
- Modify: `disaster_app/android/app/src/main/kotlin/com/example/disaster_app/MainActivity.kt`
- Modify: `disaster_app/android/app/src/main/kotlin/com/example/disaster_app/DisasterApp.kt` (only if needed for nothing — prefer AuthStore in Activity/ViewModel)

**Interfaces:**
- Consumes: `AuthStore`, `ApiClient.configure`, `DisasterApiService.login`
- Produces: `AuthViewModel` with `uiState: StateFlow<AuthUiState>`, `login(baseUrl, username, password)`, `logout()`, `refreshFromStore()`
- Produces: `data class AuthUiState(isLoggedIn, username, baseUrl, loading, error)`

- [ ] **Step 1: Implement AuthViewModel**

```kotlin
package com.example.disaster_app.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.disaster_app.data.api.ApiClient
import com.example.disaster_app.data.auth.AuthStore
import com.example.disaster_app.data.model.LoginRequest
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import retrofit2.HttpException
import java.io.IOException

data class AuthUiState(
    val isLoggedIn: Boolean = false,
    val username: String? = null,
    val baseUrl: String = AuthStore.DEFAULT_BASE_URL,
    val loading: Boolean = false,
    val error: String? = null
)

class AuthViewModel(app: Application) : AndroidViewModel(app) {
    private val store = AuthStore(app)

    private val _uiState = MutableStateFlow(
        AuthUiState(
            isLoggedIn = store.isLoggedIn(),
            username = store.getUsername(),
            baseUrl = store.getBaseUrl()
        )
    )
    val uiState: StateFlow<AuthUiState> = _uiState.asStateFlow()

    init {
        if (store.isLoggedIn()) {
            ApiClient.configure(store.getBaseUrl())
        }
    }

    fun login(baseUrl: String, username: String, password: String) {
        val u = username.trim()
        val p = password
        if (u.isEmpty() || p.isEmpty()) {
            _uiState.update { it.copy(error = "请输入用户名和密码") }
            return
        }
        if (baseUrl.trim().isEmpty()) {
            _uiState.update { it.copy(error = "请输入服务器地址") }
            return
        }
        viewModelScope.launch {
            _uiState.update { it.copy(loading = true, error = null) }
            try {
                ApiClient.configure(baseUrl)
                val resp = ApiClient.apiService.login(LoginRequest(u, p))
                when {
                    resp.isSuccessful && resp.body()?.ok == true -> {
                        val name = resp.body()?.username ?: u
                        store.saveSession(name, baseUrl)
                        ApiClient.configure(store.getBaseUrl())
                        _uiState.update {
                            it.copy(
                                isLoggedIn = true,
                                username = name,
                                baseUrl = store.getBaseUrl(),
                                loading = false,
                                error = null
                            )
                        }
                    }
                    resp.code() == 401 -> _uiState.update {
                        it.copy(loading = false, error = "用户名或密码错误")
                    }
                    resp.code() == 503 -> _uiState.update {
                        it.copy(loading = false, error = "认证服务暂不可用")
                    }
                    else -> _uiState.update {
                        it.copy(
                            loading = false,
                            error = resp.body()?.error ?: "登录失败，请稍后重试"
                        )
                    }
                }
            } catch (e: IOException) {
                _uiState.update {
                    it.copy(loading = false, error = "无法连接服务器，请检查地址与后端")
                }
            } catch (e: HttpException) {
                _uiState.update {
                    it.copy(loading = false, error = "登录失败（HTTP ${e.code()}）")
                }
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(loading = false, error = "登录失败，请稍后重试")
                }
            }
        }
    }

    fun logout() {
        store.clearSession(keepBaseUrl = true)
        _uiState.update {
            AuthUiState(
                isLoggedIn = false,
                username = null,
                baseUrl = store.getBaseUrl(),
                loading = false,
                error = null
            )
        }
    }
}
```

- [ ] **Step 2: Implement LoginPage**

Create `ui/login/LoginPage.kt` with Compose form:
- fields: baseUrl, username, password (PasswordVisualTransformation)
- defaults from `state.baseUrl`, username hint `admin`
- primary button calls `onLogin(baseUrl, username, password)`
- show `state.error` / loading spinner
- colors: `BgDeep`, `BgCard`, `AccentBlue`, `TextPrimary`, `TextMuted`, `DangerRed` from theme

Skeleton:

```kotlin
@Composable
fun LoginPage(
    state: AuthUiState,
    onLogin: (baseUrl: String, username: String, password: String) -> Unit
) {
    var baseUrl by rememberSaveable { mutableStateOf(state.baseUrl) }
    var username by rememberSaveable { mutableStateOf("admin") }
    var password by rememberSaveable { mutableStateOf("") }

    // Column fillMaxSize background BgDeep, card with OutlinedTextFields + Button
}
```

Fill full Compose matching existing theme (see `HomePage` / `Theme.kt`).

- [ ] **Step 3: Wire MainActivity auth gate**

```kotlin
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            DisasterAppTheme {
                val authVm: AuthViewModel = viewModel()
                val authState by authVm.uiState.collectAsState()
                Surface(Modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    if (authState.isLoggedIn) {
                        AppNavigation(
                            username = authState.username.orEmpty(),
                            onLogout = { authVm.logout() }
                        )
                    } else {
                        LoginPage(
                            state = authState,
                            onLogin = { base, user, pass -> authVm.login(base, user, pass) }
                        )
                    }
                }
            }
        }
    }
}
```

Add imports for `viewModel`, `collectAsState`, `AuthViewModel`, `LoginPage`.

Temporarily keep `AppNavigation()` no-arg compile by adding default params in Task 4 — **in this task**, if `AppNavigation` signature not yet updated, pass through a stub overload:

Option A (preferred in same commit as Task 4): do Task 3+4 together if compile breaks.  
Option B: update `AppNavigation` signature now with defaults:

```kotlin
fun AppNavigation(
    username: String = "",
    onLogout: () -> Unit = {}
)
```

Do Option B in Step 3 so Task 3 compiles alone.

- [ ] **Step 4: Compile**

```bash
./gradlew :app:compileDebugKotlin
```

Expected: BUILD SUCCESSFUL

- [ ] **Step 5: Commit**

```bash
git add disaster_app/android/app/src/main/kotlin/com/example/disaster_app/viewmodel/AuthViewModel.kt \
  disaster_app/android/app/src/main/kotlin/com/example/disaster_app/ui/login/LoginPage.kt \
  disaster_app/android/app/src/main/kotlin/com/example/disaster_app/MainActivity.kt \
  disaster_app/android/app/src/main/kotlin/com/example/disaster_app/ui/navigation/AppNavigation.kt
git commit -m "$(cat <<'EOF'
feat(android): add login screen and auth gate

EOF
)"
```

---

### Task 4: Three-tab navigation + MorePage + logout

**Files:**
- Create: `disaster_app/android/app/src/main/kotlin/com/example/disaster_app/ui/more/MorePage.kt`
- Modify: `disaster_app/android/app/src/main/kotlin/com/example/disaster_app/ui/navigation/AppNavigation.kt`

**Interfaces:**
- Consumes: `username`, `onLogout`
- Produces: bottomNavItems = Home, Safety, More; nested routes `help|family|report|volunteer`

- [ ] **Step 1: Rewrite Screen sealed class and bottom items**

In `AppNavigation.kt`:

```kotlin
sealed class Screen(val route: String, val title: String, val icon: ImageVector) {
    data object Home : Screen("home", "问答", Icons.Default.Forum)
    data object Safety : Screen("safety", "指引", Icons.Default.MenuBook)
    data object More : Screen("more", "更多", Icons.Default.MoreHoriz)
    data object Help : Screen("help", "求助", Icons.Default.Warning)
    data object Family : Screen("family", "家人", Icons.Default.People)
    data object Report : Screen("report", "上报", Icons.Default.Description)
    data object Volunteer : Screen("volunteer", "志愿", Icons.Default.VolunteerActivism)
}

val bottomNavItems = listOf(Screen.Home, Screen.Safety, Screen.More)
```

- [ ] **Step 2: Hide bottom bar on nested more destinations**

```kotlin
val hideBottomBar = currentDestination?.route in setOf(
    Screen.Help.route, Screen.Family.route, Screen.Report.route, Screen.Volunteer.route
)
```

Only show `NavigationBar` when `!hideBottomBar`.

- [ ] **Step 3: NavHost routes**

```kotlin
@Composable
fun AppNavigation(
    username: String = "",
    onLogout: () -> Unit = {}
) {
    val navController = rememberNavController()
    // ... Scaffold as above
    NavHost(navController, startDestination = Screen.Home.route, modifier = Modifier.padding(innerPadding)) {
        composable(Screen.Home.route) { HomePage() }
        composable(Screen.Safety.route) { SafetyPage() }
        composable(Screen.More.route) {
            MorePage(
                username = username,
                onOpen = { route -> navController.navigate(route) },
                onLogout = onLogout
            )
        }
        composable(Screen.Help.route) { HelpPage() }
        composable(Screen.Family.route) { FamilyPage() }
        composable(Screen.Report.route) { ReportPage() }
        composable(Screen.Volunteer.route) { VolunteerPage() }
    }
}
```

For nested pages, users use system back; optionally add TopAppBar back in each page later — YAGNI: system back is enough. Ensure nested pages are not in bottom bar selection logic (already excluded).

- [ ] **Step 4: Implement MorePage**

```kotlin
@Composable
fun MorePage(
    username: String,
    onOpen: (route: String) -> Unit,
    onLogout: () -> Unit
) {
    val items = listOf(
        Triple("求助", "help", Icons.Default.Warning),
        Triple("家人", "family", Icons.Default.People),
        Triple("上报", "report", Icons.Default.Description),
        Triple("志愿", "volunteer", Icons.Default.VolunteerActivism)
    )
    Column(Modifier.fillMaxSize().background(BgDeep).padding(16.dp)) {
        EmergencyTopBar(title = "更多", subtitle = "账号与应急辅助功能")
        Spacer(8.dp)
        Text("当前用户：$username", color = TextMuted)
        Spacer(16.dp)
        items.forEach { (title, route, icon) ->
            ListItem(
                headlineContent = { Text(title) },
                leadingContent = { Icon(icon, null) },
                modifier = Modifier.clickable { onOpen(route) }
            )
            HorizontalDivider()
        }
        Spacer(24.dp)
        Button(onClick = onLogout, colors = ButtonDefaults.buttonColors(containerColor = DangerRed)) {
            Text("退出登录")
        }
    }
}
```

Use theme colors that exist (`DangerRed` or `Red500` — match `ui/theme`).

- [ ] **Step 5: Compile debug APK**

```bash
./gradlew :app:assembleDebug
```

Expected: BUILD SUCCESSFUL

- [ ] **Step 6: Commit**

```bash
git add disaster_app/android/app/src/main/kotlin/com/example/disaster_app/ui/navigation/AppNavigation.kt \
  disaster_app/android/app/src/main/kotlin/com/example/disaster_app/ui/more/MorePage.kt
git commit -m "$(cat <<'EOF'
feat(android): collapse bottom nav into chat, guide, more

EOF
)"
```

---

### Task 5: Manual E2E verification

**Files:** none (verification only)

- [ ] **Step 1: Start backend + Postgres**

```bash
docker compose up -d postgres
# start Flask as you normally do (e.g. ./start.sh or python app.py) listening on 0.0.0.0:8000
```

- [ ] **Step 2: Install on emulator**

```bash
cd disaster_app/android && ./gradlew :app:installDebug
```

- [ ] **Step 3: Check login gate**

1. Cold start → LoginPage only  
2. Wrong password →「用户名或密码错误」  
3. `admin` / `admin` + default URL → enters app  
4. Bottom bar shows only 问答 / 指引 / 更多  
5. 更多 → 求助/家人/上报/志愿 each opens  
6. 退出 → back to login; baseUrl still filled  
7. Send a quick prompt on 问答 → gets answer (backend up)

- [ ] **Step 4: Optional true-device check**

On phone: set baseUrl to `http://<电脑局域网IP>:8000/`, same login flow.

- [ ] **Step 5: Final commit if any polish from QA**

Only if fixes were needed; message like `fix(android): polish login errors after device QA`.

---

## Spec coverage checklist

| Spec requirement | Task |
|---|---|
| Login gate + Postgres API | 2, 3 |
| Configurable server URL | 3 (LoginPage + AuthStore) |
| EncryptedSharedPreferences | 1 |
| Logout keep baseUrl | 1, 3, 4 |
| 3-tab nav + More children | 4 |
| Cleartext LAN | 1 |
| Chat uses new baseUrl | 2 (ApiClient + ChatRepository) |
| No Web / no JWT middleware | Global constraints |

## Plan self-review notes

- No TBD placeholders left in steps
- `AppNavigation` signature updated in Task 3 with defaults so gate compiles before MorePage
- `ChatRepository` must not cache `apiService` — covered in Task 2
- Theme color names: implementer must use existing `DangerRed` / `Red500` from `ui/theme` (check `Color.kt` at implement time)
