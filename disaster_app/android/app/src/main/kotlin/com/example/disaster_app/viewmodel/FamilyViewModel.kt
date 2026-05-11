package com.example.disaster_app.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.disaster_app.data.model.FamilyGroup
import com.example.disaster_app.data.model.FamilyMember
import com.example.disaster_app.data.repository.DisasterRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class FamilyUiState(
    val groups: List<FamilyGroup> = emptyList(),
    val isLoading: Boolean = false,
    val error: String? = null
)

class FamilyViewModel : ViewModel() {
    private val repository = DisasterRepository()

    private val _uiState = MutableStateFlow(FamilyUiState())
    val uiState: StateFlow<FamilyUiState> = _uiState.asStateFlow()

    init {
        loadGroups()
    }

    fun loadGroups() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true)
            repository.getFamilyGroups()
                .onSuccess { groups ->
                    _uiState.value = _uiState.value.copy(groups = groups, isLoading = false)
                }
                .onFailure { exception ->
                    _uiState.value = _uiState.value.copy(
                        error = exception.message,
                        isLoading = false
                    )
                }
        }
    }

    fun addMember(groupId: String, member: FamilyMember) {
        viewModelScope.launch {
            repository.addFamilyMember(groupId, member)
                .onSuccess { loadGroups() }
                .onFailure { exception ->
                    _uiState.value = _uiState.value.copy(error = exception.message)
                }
        }
    }
}
