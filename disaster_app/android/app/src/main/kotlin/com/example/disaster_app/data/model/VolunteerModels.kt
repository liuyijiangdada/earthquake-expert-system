package com.example.disaster_app.data.model

data class Volunteer(
    val id: String = "",
    val name: String = "",
    val phone: String = "",
    val skills: List<String> = emptyList(),
    val available: Boolean = true,
    val location: String = ""
)

data class HelpRequest(
    val id: String = "",
    val type: HelpType = HelpType.OTHER,
    val location: String = "",
    val description: String = "",
    val urgent: Boolean = false,
    val timestamp: Long = System.currentTimeMillis()
)

enum class HelpType(val displayName: String) {
    RESCUE("搜救"),
    MEDICAL("医疗"),
    SUPPLY("物资"),
    SHELTER("避难所"),
    OTHER("其他")
}
