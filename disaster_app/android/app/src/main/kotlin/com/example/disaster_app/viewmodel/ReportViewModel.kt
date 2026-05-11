package com.example.disaster_app.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.disaster_app.data.model.DisasterReport
import com.example.disaster_app.data.repository.DisasterRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class ReportUiState(
    val reports: List<DisasterReport> = emptyList(),
    val isLoading: Boolean = false,
    val error: String? = null,
    val submitSuccess: Boolean = false
)

class ReportViewModel : ViewModel() {
    private val repository = DisasterRepository()

    private val _uiState = MutableStateFlow(ReportUiState())
    val uiState: StateFlow<ReportUiState> = _uiState.asStateFlow()

    fun loadReports() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true)
            repository.getReports()
                .onSuccess { reports ->
                    _uiState.value = _uiState.value.copy(reports = reports, isLoading = false)
                }
                .onFailure { exception ->
                    _uiState.value = _uiState.value.copy(error = exception.message, isLoading = false)
                }
        }
    }

    fun submitReport(type: String, location: String, description: String) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true, submitSuccess = false)
            val report = DisasterReport(
                type = type,
                location = location,
                description = description
            )
            repository.submitReport(report)
                .onSuccess {
                    _uiState.value = _uiState.value.copy(isLoading = false, submitSuccess = true)
                    loadReports()
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
