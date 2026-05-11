package com.example.disaster_app.viewmodel

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
        addSystemMessage("您好！我是地震应急助手，请问有什么可以帮助您的？")
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

            repository.sendMessage(currentMessages, userInput)
                .onSuccess { response ->
                    currentMessages.add(ChatMessage(role = "assistant", content = response.response))
                    _uiState.value = _uiState.value.copy(
                        messages = currentMessages,
                        isLoading = false
                    )
                }
                .onFailure { exception ->
                    _uiState.value = _uiState.value.copy(
                        isLoading = false,
                        error = exception.message ?: "发送失败"
                    )
                }
        }
    }

    fun clearChat() {
        _uiState.value = ChatUiState()
        addSystemMessage("您好！我是地震应急助手，请问有什么可以帮助您的？")
    }
}
