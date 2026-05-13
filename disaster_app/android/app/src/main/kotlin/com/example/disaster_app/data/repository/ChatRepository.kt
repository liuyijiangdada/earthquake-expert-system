package com.example.disaster_app.data.repository

import android.content.Context
import android.net.Uri
import com.example.disaster_app.data.api.ApiClient
import com.example.disaster_app.data.model.ChatMessage
import com.example.disaster_app.data.model.ChatRequest
import com.example.disaster_app.data.model.ChatResponse
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody

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

    suspend fun sendMultimodalMessage(
        context: Context,
        imageUri: Uri,
        userInput: String
    ): Result<ChatResponse> = withContext(Dispatchers.IO) {
        try {
            val inputStream = context.contentResolver.openInputStream(imageUri)
                ?: throw Exception("无法读取图片")

            val bytes = inputStream.readBytes()
            inputStream.close()

            val requestBody = bytes.toRequestBody("image/*".toMediaTypeOrNull())
            val imagePart = MultipartBody.Part.createFormData("image", "photo.jpg", requestBody)
            val inputPart = userInput.toRequestBody("text/plain".toMediaTypeOrNull())

            val response = apiService.multimodalQuery(imagePart, inputPart)
            Result.success(response)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}
