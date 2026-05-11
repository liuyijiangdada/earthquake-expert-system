package com.example.disaster_app.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.disaster_app.data.model.HelpRequest
import com.example.disaster_app.data.model.HelpType
import com.example.disaster_app.data.repository.DisasterRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class HelpUiState(
    val requests: List<HelpRequest> = emptyList(),
    val isLoading: Boolean = false,
    val error: String? = null,
    val submitSuccess: Boolean = false
)

class HelpViewModel : ViewModel() {
    private val repository = DisasterRepository()

    private val _uiState = MutableStateFlow(HelpUiState())
    val uiState: StateFlow<HelpUiState> = _uiState.asStateFlow()

    fun loadRequests() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true)
            repository.getHelpRequests()
                .onSuccess { requests ->
                    _uiState.value = _uiState.value.copy(requests = requests, isLoading = false)
                }
                .onFailure { exception ->
                    _uiState.value = _uiState.value.copy(error = exception.message, isLoading = false)
                }
        }
    }

    fun submitRequest(type: HelpType, location: String, description: String, urgent: Boolean) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true, submitSuccess = false)
            val request = HelpRequest(
                type = type,
                location = location,
                description = description,
                urgent = urgent
            )
            repository.submitHelpRequest(request)
                .onSuccess {
                    _uiState.value = _uiState.value.copy(isLoading = false, submitSuccess = true)
                    loadRequests()
                }
                .onFailure { exception ->
                    _uiState.value = _uiState.value.copy(error = exception.message, isLoading = false)
                }
        }
    }

    fun clearSubmitSuccess() {
        _uiState.value = _uiState.value.copy(submitSuccess = false)
    }
}
