package com.example.disaster_app.data.api

import com.example.disaster_app.data.model.ChatResponse
import com.example.disaster_app.data.model.LlmQueryRequest
import com.example.disaster_app.data.model.LoginRequest
import com.example.disaster_app.data.model.LoginResponse
import okhttp3.MultipartBody
import okhttp3.RequestBody
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

interface DisasterApiService {
    @POST("api/auth/login")
    suspend fun login(@Body request: LoginRequest): Response<LoginResponse>

    @POST("api/query")
    suspend fun queryLlm(@Body request: LlmQueryRequest): ChatResponse

    @Multipart
    @POST("api/multimodal-query")
    suspend fun multimodalQuery(
        @Part image: MultipartBody.Part,
        @Part("input") input: RequestBody,
        @Part("history") history: RequestBody? = null
    ): ChatResponse
}
