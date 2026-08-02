package com.example.disaster_app.data.api

import com.example.disaster_app.data.auth.AuthStore
import com.example.disaster_app.data.auth.normalizeBaseUrl
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

object ApiClient {
    private val loggingInterceptor = HttpLoggingInterceptor().apply {
        // HEADERS only — BODY would log login credentials in Logcat
        level = HttpLoggingInterceptor.Level.HEADERS
    }

    private val okHttpClient = OkHttpClient.Builder()
        .addInterceptor(loggingInterceptor)
        .connectTimeout(60, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()

    @Volatile
    private var currentBaseUrl: String = AuthStore.DEFAULT_BASE_URL

    @Volatile
    private var retrofit: Retrofit = buildRetrofit(currentBaseUrl)

    @Volatile
    private var service: DisasterApiService = retrofit.create(DisasterApiService::class.java)

    val apiService: DisasterApiService
        get() = service

    @Synchronized
    fun configure(baseUrl: String) {
        val normalized = normalizeBaseUrl(baseUrl)
        currentBaseUrl = normalized
        retrofit = buildRetrofit(normalized)
        service = retrofit.create(DisasterApiService::class.java)
    }

    private fun buildRetrofit(baseUrl: String): Retrofit =
        Retrofit.Builder()
            .baseUrl(baseUrl)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
}
