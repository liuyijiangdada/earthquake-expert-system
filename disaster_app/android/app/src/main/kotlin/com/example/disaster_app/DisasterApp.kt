package com.example.disaster_app

import android.app.Application
import android.util.Log

class DisasterApp : Application() {
    override fun onCreate() {
        super.onCreate()
        Log.i("DisasterApp", "地震应急应用启动")
    }
}
