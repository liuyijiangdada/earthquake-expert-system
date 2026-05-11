package com.example.disaster_app.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.disaster_app.data.model.SafetyLevel
import com.example.disaster_app.data.model.SafetyStatus
import com.example.disaster_app.data.repository.DisasterRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class SafetyUiState(
    val currentStatus: SafetyStatus = SafetyStatus(),
    val isLoading: Boolean = false,
    val error: String? = null
)

class SafetyViewModel : ViewModel() {
    private val repository = DisasterRepository()

    private val _uiState = MutableStateFlow(SafetyUiState())
    val uiState: StateFlow<SafetyUiState> = _uiState.asStateFlow()

    fun updateStatus(level: SafetyLevel, location: String) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true)
            val status = SafetyStatus(
                status = level,
                location = location,
                lastUpdate = System.currentTimeMillis()
            )
            repository.updateSafetyStatus(status)
                .onSuccess {
                    _uiState.value = _uiState.value.copy(currentStatus = status, isLoading = false)
                }
                .onFailure { exception ->
                    _uiState.value = _uiState.value.copy(error = exception.message, isLoading = false)
                }
        }
    }
}
