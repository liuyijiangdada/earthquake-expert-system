package com.example.disaster_app.data.model

data class ChatMessage(
    val role: String,
    val content: String
)

data class ChatRequest(
    val instruction: String,
    val input: String
)

data class ChatResponse(
    val response: String,
    val debug: DebugInfo? = null
)

data class DebugInfo(
    val kg_enabled: Boolean,
    val rag_enabled: Boolean,
    val rag_topic_ids: List<String>
)
