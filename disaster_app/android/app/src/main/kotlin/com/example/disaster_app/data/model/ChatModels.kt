package com.example.disaster_app.data.model

import android.net.Uri
import com.google.gson.annotations.SerializedName

data class ChatMessage(
    val role: String,
    val content: String,
    val imageUri: Uri? = null,
    val phase: String? = null,
    val urgency: Float = 0f,
    val mediaCaptions: List<String> = emptyList()
)

data class LlmQueryRequest(
    @SerializedName("query_type") val queryType: String = "llm",
    val params: LlmQueryParams
)

data class LlmQueryParams(
    val input: String,
    val history: List<HistoryTurn> = emptyList()
)

data class HistoryTurn(
    val role: String,
    val content: String
)

data class ChatResponse(
    val response: String? = null,
    val error: String? = null,
    @SerializedName("meta") val meta: ResponseMeta? = null,
    /** 兼容旧接口字段名 debug */
    @SerializedName("debug") val debug: ResponseMeta? = null
) {
    fun responseMeta(): ResponseMeta? = meta ?: debug
}

data class ResponseMeta(
    val phase: String? = null,
    val urgency: Float? = null,
    @SerializedName("static_confidence") val staticConfidence: Float? = null,
    @SerializedName("dynamic_availability") val dynamicAvailability: Float? = null,
    @SerializedName("reliability_hint") val reliabilityHint: String? = null,
    @SerializedName("media_resources") val mediaResources: List<MediaResource>? = null
)

data class MediaResource(
    val id: String? = null,
    val type: String? = null,
    val url: String? = null,
    val caption: String? = null,
    val source: String? = null
)
