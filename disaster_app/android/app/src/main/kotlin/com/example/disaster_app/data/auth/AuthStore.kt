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
