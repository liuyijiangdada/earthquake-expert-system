package com.example.disaster_app.data.api

import com.example.disaster_app.data.model.ChatRequest
import com.example.disaster_app.data.model.ChatResponse
import okhttp3.MultipartBody
import okhttp3.RequestBody
import retrofit2.http.Body
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

interface DisasterApiService {
    @POST("api/query")
    suspend fun query(@Body request: ChatRequest): ChatResponse

    @Multipart
    @POST("api/multimodal-query")
    suspend fun multimodalQuery(
        @Part image: MultipartBody.Part,
        @Part("input") input: RequestBody
    ): ChatResponse
}
