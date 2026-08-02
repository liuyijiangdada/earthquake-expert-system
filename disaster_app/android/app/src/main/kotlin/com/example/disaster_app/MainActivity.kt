package com.example.disaster_app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.disaster_app.ui.login.LoginPage
import com.example.disaster_app.ui.navigation.AppNavigation
import com.example.disaster_app.ui.theme.DisasterAppTheme
import com.example.disaster_app.viewmodel.AuthViewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            DisasterAppTheme {
                val authVm: AuthViewModel = viewModel()
                val authState by authVm.uiState.collectAsState()
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    if (authState.isLoggedIn) {
                        AppNavigation(
                            username = authState.username.orEmpty(),
                            onLogout = { authVm.logout() }
                        )
                    } else {
                        LoginPage(
                            state = authState,
                            onLogin = { base, user, pass -> authVm.login(base, user, pass) }
                        )
                    }
                }
            }
        }
    }
}
