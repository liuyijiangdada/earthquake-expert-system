package com.example.disaster_app.ui.home

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.disaster_app.data.model.ChatMessage
import com.example.disaster_app.ui.components.*
import com.example.disaster_app.ui.theme.*
import com.example.disaster_app.viewmodel.ChatViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomePage(
    viewModel: ChatViewModel = viewModel()
) {
    val uiState by viewModel.uiState.collectAsState()
    var inputText by remember { mutableStateOf("") }
    var selectedImageUri by remember { mutableStateOf<Uri?>(null) }
    val listState = rememberLazyListState()
    val context = LocalContext.current

    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { selectedImageUri = it }
    }

    LaunchedEffect(uiState.messages.size) {
        if (uiState.messages.isNotEmpty()) {
            listState.animateScrollToItem(uiState.messages.size - 1)
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(BgDeep)
            .padding(horizontal = 16.dp, vertical = 12.dp)
    ) {
        EmergencyTopBar(
            title = "地震应急智能问答",
            subtitle = "KG + RAG + 实时震情 · 震前 / 震中 / 震后"
        )

        Spacer(modifier = Modifier.height(10.dp))
        CapabilityTags()
        Spacer(modifier = Modifier.height(10.dp))
        QuickPromptRow(onPromptClick = { q ->
            inputText = q
            viewModel.sendMessage(q)
        })

        Spacer(modifier = Modifier.height(12.dp))

        Card(
            modifier = Modifier.weight(1f),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = BgChat),
            border = androidx.compose.foundation.BorderStroke(1.dp, BotBubbleBorder)
        ) {
            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(12.dp),
                state = listState,
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                items(uiState.messages) { message ->
                    ChatBubble(message = message)
                }

                if (uiState.isLoading) {
                    item {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(22.dp),
                                strokeWidth = 2.dp,
                                color = AccentBlue
                            )
                            Text(
                                "正在协同检索知识源…",
                                style = MaterialTheme.typography.bodySmall,
                                color = TextMuted
                            )
                        }
                    }
                }
            }
        }

        uiState.error?.let { error ->
            Spacer(modifier = Modifier.height(8.dp))
            Card(
                colors = CardDefaults.cardColors(containerColor = DangerRed.copy(alpha = 0.15f)),
                modifier = Modifier.fillMaxWidth()
            ) {
                Row(
                    modifier = Modifier.padding(12.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(error, color = DangerRed, style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
                    TextButton(onClick = { viewModel.clearError() }) {
                        Text("关闭", color = DangerRed)
                    }
                }
            }
        }

        selectedImageUri?.let { uri ->
            Spacer(modifier = Modifier.height(8.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                AsyncImage(
                    model = uri,
                    contentDescription = "待发送图片",
                    modifier = Modifier
                        .size(64.dp)
                        .clip(RoundedCornerShape(10.dp)),
                    contentScale = ContentScale.Crop
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text("已选图片，可补充文字后发送", style = MaterialTheme.typography.bodySmall, color = TextMuted)
                Spacer(modifier = Modifier.weight(1f))
                IconButton(onClick = { selectedImageUri = null }) {
                    Icon(Icons.Default.Close, contentDescription = "移除", tint = DangerRed)
                }
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.Bottom
        ) {
            IconButton(onClick = { imagePickerLauncher.launch("image/*") }) {
                Icon(Icons.Default.AddPhotoAlternate, contentDescription = "图片", tint = AccentBlue)
            }

            OutlinedTextField(
                value = inputText,
                onValueChange = { inputText = it },
                modifier = Modifier.weight(1f),
                placeholder = { Text("震前准备 / 震中避险 / 震后恢复…", color = TextMuted) },
                shape = RoundedCornerShape(16.dp),
                maxLines = 3,
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = AccentBlue,
                    unfocusedBorderColor = BotBubbleBorder,
                    focusedTextColor = TextPrimary,
                    unfocusedTextColor = TextPrimary
                )
            )

            Spacer(modifier = Modifier.width(8.dp))

            FilledIconButton(
                onClick = {
                    val hasInput = inputText.isNotBlank()
                    val hasImage = selectedImageUri != null
                    if (hasInput || hasImage) {
                        if (hasImage) {
                            viewModel.sendMultimodalMessage(context, selectedImageUri!!, inputText)
                            selectedImageUri = null
                        } else {
                            viewModel.sendMessage(inputText)
                        }
                        inputText = ""
                    }
                },
                enabled = (inputText.isNotBlank() || selectedImageUri != null) && !uiState.isLoading,
                colors = IconButtonDefaults.filledIconButtonColors(containerColor = AccentBlueDim)
            ) {
                Icon(Icons.Default.Send, contentDescription = "发送", tint = Color.White)
            }
        }
    }
}

@Composable
fun ChatBubble(message: ChatMessage) {
    val isUser = message.role == "user"
    val isSystem = message.role == "system"

    if (isSystem) {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center) {
            Surface(
                shape = RoundedCornerShape(12.dp),
                color = BgCard,
                border = androidx.compose.foundation.BorderStroke(1.dp, BotBubbleBorder)
            ) {
                Text(
                    text = message.content,
                    modifier = Modifier.padding(horizontal = 14.dp, vertical = 8.dp),
                    style = MaterialTheme.typography.bodySmall,
                    color = TextMuted
                )
            }
        }
        return
    }

    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start
    ) {
        Column(
            horizontalAlignment = if (isUser) Alignment.End else Alignment.Start,
            modifier = Modifier.widthIn(max = 300.dp)
        ) {
            if (!isUser) {
                PhaseBadge(phase = message.phase, urgency = message.urgency)
                Spacer(modifier = Modifier.height(4.dp))
            }

            message.imageUri?.let { uri ->
                AsyncImage(
                    model = uri,
                    contentDescription = "用户图片",
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(max = 180.dp)
                        .clip(RoundedCornerShape(12.dp)),
                    contentScale = ContentScale.Crop
                )
                Spacer(modifier = Modifier.height(4.dp))
            }

            if (message.content.isNotBlank()) {
                Surface(
                    shape = RoundedCornerShape(
                        topStart = 14.dp,
                        topEnd = 14.dp,
                        bottomStart = if (isUser) 14.dp else 4.dp,
                        bottomEnd = if (isUser) 4.dp else 14.dp
                    ),
                    color = if (isUser) UserBubble else BotBubble,
                    border = if (isUser) null else androidx.compose.foundation.BorderStroke(1.dp, BotBubbleBorder)
                ) {
                    Text(
                        text = message.content,
                        modifier = Modifier.padding(12.dp),
                        color = if (isUser) Color.White else TextPrimary,
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }

            if (!isUser && message.mediaCaptions.isNotEmpty()) {
                Spacer(modifier = Modifier.height(6.dp))
                message.mediaCaptions.take(3).forEach { cap ->
                    Text(
                        text = "📎 $cap",
                        style = MaterialTheme.typography.labelSmall,
                        color = AccentBlue
                    )
                }
            }
        }
    }
}
