package com.example.disaster_app.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.disaster_app.data.model.Volunteer
import com.example.disaster_app.data.repository.DisasterRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class VolunteerUiState(
    val volunteers: List<Volunteer> = emptyList(),
    val isLoading: Boolean = false,
    val error: String? = null,
    val registerSuccess: Boolean = false
)

class VolunteerViewModel : ViewModel() {
    private val repository = DisasterRepository()

    private val _uiState = MutableStateFlow(VolunteerUiState())
    val uiState: StateFlow<VolunteerUiState> = _uiState.asStateFlow()

    fun loadVolunteers() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true)
            repository.getVolunteers()
                .onSuccess { volunteers ->
                    _uiState.value = _uiState.value.copy(volunteers = volunteers, isLoading = false)
                }
                .onFailure { exception ->
                    _uiState.value = _uiState.value.copy(error = exception.message, isLoading = false)
                }
        }
    }

    fun registerVolunteer(name: String, phone: String, skills: List<String>, location: String) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true, registerSuccess = false)
            val volunteer = Volunteer(
                name = name,
                phone = phone,
                skills = skills,
                location = location
            )
            repository.registerVolunteer(volunteer)
                .onSuccess {
                    _uiState.value = _uiState.value.copy(isLoading = false, registerSuccess = true)
                    loadVolunteers()
                }
                .onFailure { exception ->
                    _uiState.value = _uiState.value.copy(error = exception.message, isLoading = false)
                }
        }
    }

    fun clearRegisterSuccess() {
        _uiState.value = _uiState.value.copy(registerSuccess = false)
    }
}
