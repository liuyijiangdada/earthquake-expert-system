package com.example.disaster_app.data.auth

fun normalizeBaseUrl(raw: String): String {
    val trimmed = raw.trim()
    if (trimmed.isEmpty()) return AuthStore.DEFAULT_BASE_URL
    return if (trimmed.endsWith("/")) trimmed else "$trimmed/"
}
