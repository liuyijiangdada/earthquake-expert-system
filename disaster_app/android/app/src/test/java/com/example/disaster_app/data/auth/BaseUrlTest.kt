package com.example.disaster_app.data.auth

import org.junit.Assert.assertEquals
import org.junit.Test

class BaseUrlTest {
    @Test
    fun trims_and_ensures_trailing_slash() {
        assertEquals("http://10.0.2.2:8000/", normalizeBaseUrl("http://10.0.2.2:8000"))
        assertEquals("http://10.0.2.2:8000/", normalizeBaseUrl("  http://10.0.2.2:8000/  "))
        assertEquals("http://192.168.1.8:8000/", normalizeBaseUrl("http://192.168.1.8:8000"))
    }

    @Test
    fun empty_falls_back_to_default() {
        assertEquals(AuthStore.DEFAULT_BASE_URL, normalizeBaseUrl(""))
        assertEquals(AuthStore.DEFAULT_BASE_URL, normalizeBaseUrl("   "))
    }
}
