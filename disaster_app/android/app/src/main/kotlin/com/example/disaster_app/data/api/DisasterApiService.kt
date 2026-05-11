package com.example.disaster_app.data.api

import com.example.disaster_app.data.model.ChatRequest
import com.example.disaster_app.data.model.ChatResponse
import retrofit2.http.Body
import retrofit2.http.POST

interface DisasterApiService {
    @POST("api/query")
    suspend fun query(@Body request: ChatRequest): ChatResponse
}
