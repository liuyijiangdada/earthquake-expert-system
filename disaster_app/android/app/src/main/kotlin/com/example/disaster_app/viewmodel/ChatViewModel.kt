package com.example.disaster_app.viewmodel

import android.content.Context
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.disaster_app.data.model.ChatMessage
import com.example.disaster_app.data.repository.ChatRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class ChatUiState(
    val messages: List<ChatMessage> = emptyList(),
    val isLoading: Boolean = false,
    val error: String? = null
)

class ChatViewModel : ViewModel() {
    private val repository = ChatRepository()

    private val _uiState = MutableStateFlow(ChatUiState())
    val uiState: StateFlow<ChatUiState> = _uiState.asStateFlow()

    init {
        addSystemMessage(
            "你好，我是地震应急智能助手。可咨询震前准备、震中避险、震后恢复；支持文字与图片问答。请确保手机已连接后端服务（模拟器默认 10.0.2.2:8000）。"
        )
    }

    private fun addSystemMessage(content: String) {
        val currentMessages = _uiState.value.messages.toMutableList()
        currentMessages.add(ChatMessage(role = "system", content = content))
        _uiState.value = _uiState.value.copy(messages = currentMessages)
    }

    fun sendMessage(userInput: String) {
        viewModelScope.launch {
            val currentMessages = _uiState.value.messages.toMutableList()
            currentMessages.add(ChatMessage(role = "user", content = userInput))
            _uiState.value = _uiState.value.copy(messages = currentMessages, isLoading = true, error = null)

            val history = currentMessages.filter { it.role != "system" }
            repository.sendMessage(history, userInput)
                .onSuccess { response ->
                    val assistant = buildAssistantMessage(response.response!!, response.debug)
                    currentMessages.add(assistant)
                    _uiState.value = _uiState.value.copy(messages = currentMessages, isLoading = false)
                }
                .onFailure { exception ->
                    _uiState.value = _uiState.value.copy(
                        isLoading = false,
                        error = exception.message ?: "发送失败"
                    )
                }
        }
    }

    fun sendMultimodalMessage(context: Context, imageUri: Uri, userInput: String) {
        viewModelScope.launch {
            val currentMessages = _uiState.value.messages.toMutableList()
            val text = userInput.ifBlank { "请分析这张与地震应急相关的图片" }
            currentMessages.add(
                ChatMessage(role = "user", content = text, imageUri = imageUri)
            )
            _uiState.value = _uiState.value.copy(messages = currentMessages, isLoading = true, error = null)

            val history = currentMessages.filter { it.role != "system" }
            repository.sendMultimodalMessage(context, imageUri, text, history)
                .onSuccess { response ->
                    val assistant = buildAssistantMessage(response.response!!, response.debug)
                    currentMessages.add(assistant)
                    _uiState.value = _uiState.value.copy(messages = currentMessages, isLoading = false)
                }
                .onFailure { exception ->
                    _uiState.value = _uiState.value.copy(
                        isLoading = false,
                        error = exception.message ?: "发送失败"
                    )
                }
        }
    }

    fun clearError() {
        _uiState.value = _uiState.value.copy(error = null)
    }

    private fun buildAssistantMessage(text: String, debug: com.example.disaster_app.data.model.DebugInfo?): ChatMessage {
        val captions = debug?.mediaResources?.mapNotNull { it.caption } ?: emptyList()
        return ChatMessage(
            role = "assistant",
            content = text,
            phase = debug?.phase,
            urgency = debug?.urgency ?: 0f,
            mediaCaptions = captions
        )
    }
}
