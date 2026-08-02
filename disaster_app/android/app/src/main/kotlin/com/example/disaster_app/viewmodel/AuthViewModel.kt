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

    fun refreshFromStore() {
        val loggedIn = store.isLoggedIn()
        if (loggedIn) {
            ApiClient.configure(store.getBaseUrl())
        }
        _uiState.update {
            it.copy(
                isLoggedIn = loggedIn,
                username = store.getUsername(),
                baseUrl = store.getBaseUrl(),
                loading = false,
                error = null
            )
        }
    }
}
