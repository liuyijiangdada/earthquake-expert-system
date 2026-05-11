package com.example.disaster_app.data.model

data class DisasterReport(
    val id: String = "",
    val type: String = "",
    val location: String = "",
    val description: String = "",
    val timestamp: Long = System.currentTimeMillis(),
    val status: String = "pending"
)

data class SafetyStatus(
    val userId: String = "",
    val status: SafetyLevel = SafetyLevel.UNKNOWN,
    val location: String = "",
    val lastUpdate: Long = System.currentTimeMillis()
)

enum class SafetyLevel {
    SAFE,
    WARNING,
    DANGER,
    UNKNOWN
}
