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
