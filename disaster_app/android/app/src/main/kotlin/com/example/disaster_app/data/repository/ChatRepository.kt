package com.example.disaster_app.data.repository

import com.example.disaster_app.data.api.ApiClient
import com.example.disaster_app.data.model.ChatMessage
import com.example.disaster_app.data.model.ChatRequest
import com.example.disaster_app.data.model.ChatResponse
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class ChatRepository {
    private val apiService = ApiClient.apiService

    suspend fun sendMessage(
        messages: List<ChatMessage>,
        userInput: String
    ): Result<ChatResponse> = withContext(Dispatchers.IO) {
        try {
            val instruction = messages.joinToString("\n") {
                "${it.role}: ${it.content}"
            }
            val request = ChatRequest(
                instruction = instruction,
                input = userInput
            )
            val response = apiService.query(request)
            Result.success(response)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}
