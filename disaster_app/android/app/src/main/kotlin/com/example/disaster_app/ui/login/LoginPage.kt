package com.example.disaster_app.ui.login

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.Public
import androidx.compose.material.icons.filled.Shield
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.FocusDirection
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.example.disaster_app.ui.theme.AccentBlue
import com.example.disaster_app.ui.theme.AccentBlueDim
import com.example.disaster_app.ui.theme.BgCard
import com.example.disaster_app.ui.theme.BgDeep
import com.example.disaster_app.ui.theme.BotBubbleBorder
import com.example.disaster_app.ui.theme.DangerRed
import com.example.disaster_app.ui.theme.TextMuted
import com.example.disaster_app.ui.theme.TextPrimary
import com.example.disaster_app.viewmodel.AuthUiState

@Composable
fun LoginPage(
    state: AuthUiState,
    onLogin: (baseUrl: String, username: String, password: String) -> Unit
) {
    var baseUrl by rememberSaveable { mutableStateOf(state.baseUrl) }
    var username by rememberSaveable { mutableStateOf("admin") }
    var password by rememberSaveable { mutableStateOf("") }
    val focusManager = LocalFocusManager.current

    LaunchedEffect(state.baseUrl) {
        baseUrl = state.baseUrl
    }

    val fieldColors = OutlinedTextFieldDefaults.colors(
        focusedBorderColor = AccentBlue,
        unfocusedBorderColor = BotBubbleBorder,
        disabledBorderColor = BotBubbleBorder,
        focusedTextColor = TextPrimary,
        unfocusedTextColor = TextPrimary,
        disabledTextColor = TextMuted,
        focusedLabelColor = AccentBlue,
        unfocusedLabelColor = TextMuted,
        cursorColor = AccentBlue,
        focusedLeadingIconColor = AccentBlue,
        unfocusedLeadingIconColor = TextMuted,
        focusedPlaceholderColor = TextMuted,
        unfocusedPlaceholderColor = TextMuted
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(BgDeep)
            .padding(horizontal = 24.dp, vertical = 32.dp),
        contentAlignment = Alignment.Center
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Icon(
                imageVector = Icons.Default.Shield,
                contentDescription = null,
                tint = AccentBlue,
                modifier = Modifier.size(48.dp)
            )
            Spacer(modifier = Modifier.height(12.dp))
            Text(
                text = "地震应急智能问答",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
                color = TextPrimary,
                textAlign = TextAlign.Center
            )
            Spacer(modifier = Modifier.height(6.dp))
            Text(
                text = "请登录后继续使用系统",
                style = MaterialTheme.typography.bodyMedium,
                color = TextMuted,
                textAlign = TextAlign.Center
            )

            Spacer(modifier = Modifier.height(28.dp))

            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = BgCard),
                border = BorderStroke(1.dp, BotBubbleBorder)
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(20.dp),
                    verticalArrangement = Arrangement.spacedBy(14.dp)
                ) {
                    OutlinedTextField(
                        value = baseUrl,
                        onValueChange = { baseUrl = it },
                        modifier = Modifier.fillMaxWidth(),
                        enabled = !state.loading,
                        singleLine = true,
                        label = { Text("服务器地址") },
                        placeholder = { Text("http://10.0.2.2:8000/") },
                        leadingIcon = {
                            Icon(Icons.Default.Public, contentDescription = null)
                        },
                        keyboardOptions = KeyboardOptions(
                            keyboardType = KeyboardType.Uri,
                            imeAction = ImeAction.Next
                        ),
                        keyboardActions = KeyboardActions(
                            onNext = { focusManager.moveFocus(FocusDirection.Down) }
                        ),
                        shape = RoundedCornerShape(12.dp),
                        colors = fieldColors
                    )

                    OutlinedTextField(
                        value = username,
                        onValueChange = { username = it },
                        modifier = Modifier.fillMaxWidth(),
                        enabled = !state.loading,
                        singleLine = true,
                        label = { Text("用户名") },
                        placeholder = { Text("admin") },
                        leadingIcon = {
                            Icon(Icons.Default.Person, contentDescription = null)
                        },
                        keyboardOptions = KeyboardOptions(
                            keyboardType = KeyboardType.Text,
                            imeAction = ImeAction.Next
                        ),
                        keyboardActions = KeyboardActions(
                            onNext = { focusManager.moveFocus(FocusDirection.Down) }
                        ),
                        shape = RoundedCornerShape(12.dp),
                        colors = fieldColors
                    )

                    OutlinedTextField(
                        value = password,
                        onValueChange = { password = it },
                        modifier = Modifier.fillMaxWidth(),
                        enabled = !state.loading,
                        singleLine = true,
                        label = { Text("密码") },
                        placeholder = { Text("请输入密码") },
                        leadingIcon = {
                            Icon(Icons.Default.Lock, contentDescription = null)
                        },
                        visualTransformation = PasswordVisualTransformation(),
                        keyboardOptions = KeyboardOptions(
                            keyboardType = KeyboardType.Password,
                            imeAction = ImeAction.Done
                        ),
                        keyboardActions = KeyboardActions(
                            onDone = {
                                focusManager.clearFocus()
                                if (!state.loading) {
                                    onLogin(baseUrl, username, password)
                                }
                            }
                        ),
                        shape = RoundedCornerShape(12.dp),
                        colors = fieldColors
                    )

                    state.error?.let { error ->
                        Text(
                            text = error,
                            style = MaterialTheme.typography.bodySmall,
                            color = DangerRed,
                            modifier = Modifier.fillMaxWidth()
                        )
                    }

                    Button(
                        onClick = { onLogin(baseUrl, username, password) },
                        enabled = !state.loading,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(48.dp),
                        shape = RoundedCornerShape(12.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = AccentBlueDim,
                            contentColor = TextPrimary,
                            disabledContainerColor = AccentBlueDim.copy(alpha = 0.5f),
                            disabledContentColor = TextMuted
                        )
                    ) {
                        if (state.loading) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(20.dp),
                                strokeWidth = 2.dp,
                                color = TextPrimary
                            )
                            Spacer(modifier = Modifier.size(10.dp))
                            Text("登录中…")
                        } else {
                            Text(
                                text = "登录",
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.SemiBold
                            )
                        }
                    }
                }
            }

//            Spacer(modifier = Modifier.height(16.dp))
//            Text(
//                text = "模拟器默认地址 10.0.2.2 指向本机后端",
//                style = MaterialTheme.typography.labelSmall,
//                color = TextMuted,
//                textAlign = TextAlign.Center
//            )
        }
    }
}
