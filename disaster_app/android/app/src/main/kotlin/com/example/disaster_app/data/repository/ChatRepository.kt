package com.example.disaster_app.data.repository

import android.content.Context
import android.net.Uri
import com.example.disaster_app.data.api.ApiClient
import com.example.disaster_app.data.model.ChatMessage
import com.example.disaster_app.data.model.ChatResponse
import com.example.disaster_app.data.model.HistoryTurn
import com.example.disaster_app.data.model.LlmQueryParams
import com.example.disaster_app.data.model.LlmQueryRequest
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody

class ChatRepository {
    private val apiService = ApiClient.apiService

    private fun buildHistory(messages: List<ChatMessage>): List<HistoryTurn> {
        return messages
            .filter { it.role == "user" || it.role == "assistant" }
            .takeLast(6)
            .map { HistoryTurn(role = it.role, content = it.content) }
    }

    suspend fun sendMessage(
        messages: List<ChatMessage>,
        userInput: String
    ): Result<ChatResponse> = withContext(Dispatchers.IO) {
        try {
            val history = buildHistory(messages.filter { it.content.isNotBlank() })
            val request = LlmQueryRequest(
                params = LlmQueryParams(input = userInput, history = history)
            )
            val response = apiService.queryLlm(request)
            if (!response.error.isNullOrBlank()) {
                Result.failure(Exception(response.error))
            } else if (response.response.isNullOrBlank()) {
                Result.failure(Exception("未收到有效回答"))
            } else {
                Result.success(response)
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun sendMultimodalMessage(
        context: Context,
        imageUri: Uri,
        userInput: String,
        historyMessages: List<ChatMessage> = emptyList()
    ): Result<ChatResponse> = withContext(Dispatchers.IO) {
        try {
            val inputStream = context.contentResolver.openInputStream(imageUri)
                ?: throw Exception("无法读取图片")

            val bytes = inputStream.readBytes()
            inputStream.close()

            val requestBody = bytes.toRequestBody("image/*".toMediaTypeOrNull())
            val imagePart = MultipartBody.Part.createFormData("image", "photo.jpg", requestBody)
            val inputPart = (userInput.ifBlank { "请分析这张与地震应急相关的图片" })
                .toRequestBody("text/plain".toMediaTypeOrNull())

            val historyJson = buildHistory(historyMessages)
            val historyPart = if (historyJson.isNotEmpty()) {
                com.google.gson.Gson().toJson(historyJson)
                    .toRequestBody("application/json".toMediaTypeOrNull())
            } else null

            val response = if (historyPart != null) {
                apiService.multimodalQuery(imagePart, inputPart, historyPart)
            } else {
                apiService.multimodalQuery(imagePart, inputPart, null)
            }
            if (!response.error.isNullOrBlank()) {
                Result.failure(Exception(response.error))
            } else if (response.response.isNullOrBlank()) {
                Result.failure(Exception("未收到有效回答"))
            } else {
                Result.success(response)
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}
