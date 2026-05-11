package com.example.disaster_app.data.model

data class FamilyGroup(
    val id: String = "",
    val name: String = "",
    val members: List<FamilyMember> = emptyList()
)

data class FamilyMember(
    val id: String = "",
    val name: String = "",
    val phone: String = "",
    val safetyStatus: SafetyLevel = SafetyLevel.UNKNOWN
)
