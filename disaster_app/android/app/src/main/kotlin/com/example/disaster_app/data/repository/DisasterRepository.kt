package com.example.disaster_app.data.repository

import com.example.disaster_app.data.model.DisasterReport
import com.example.disaster_app.data.model.FamilyGroup
import com.example.disaster_app.data.model.FamilyMember
import com.example.disaster_app.data.model.HelpRequest
import com.example.disaster_app.data.model.SafetyLevel
import com.example.disaster_app.data.model.SafetyStatus
import com.example.disaster_app.data.model.Volunteer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class DisasterRepository {
    private val reports = mutableListOf<DisasterReport>()
    private val familyGroups = mutableListOf<FamilyGroup>()
    private val volunteers = mutableListOf<Volunteer>()
    private val helpRequests = mutableListOf<HelpRequest>()
    private var currentSafetyStatus: SafetyStatus = SafetyStatus()

    suspend fun submitReport(report: DisasterReport): Result<DisasterReport> = withContext(Dispatchers.IO) {
        try {
            val newReport = report.copy(id = System.currentTimeMillis().toString())
            reports.add(newReport)
            Result.success(newReport)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun getReports(): Result<List<DisasterReport>> = withContext(Dispatchers.IO) {
        Result.success(reports.toList())
    }

    suspend fun updateSafetyStatus(status: SafetyStatus): Result<SafetyStatus> = withContext(Dispatchers.IO) {
        try {
            currentSafetyStatus = status
            Result.success(status)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun getSafetyStatus(): Result<SafetyStatus> = withContext(Dispatchers.IO) {
        Result.success(currentSafetyStatus)
    }

    suspend fun getFamilyGroups(): Result<List<FamilyGroup>> = withContext(Dispatchers.IO) {
        if (familyGroups.isEmpty()) {
            familyGroups.add(
                FamilyGroup(
                    id = "1",
                    name = "我的家庭",
                    members = listOf(
                        FamilyMember(id = "1", name = "父亲", phone = "138****1234", safetyStatus = SafetyLevel.SAFE),
                        FamilyMember(id = "2", name = "母亲", phone = "139****5678", safetyStatus = SafetyLevel.SAFE)
                    )
                )
            )
        }
        Result.success(familyGroups.toList())
    }

    suspend fun addFamilyMember(groupId: String, member: FamilyMember): Result<FamilyMember> = withContext(Dispatchers.IO) {
        try {
            val group = familyGroups.find { it.id == groupId }
            if (group != null) {
                val newMember = member.copy(id = System.currentTimeMillis().toString())
                familyGroups[familyGroups.indexOf(group)] = group.copy(
                    members = group.members + newMember
                )
                Result.success(newMember)
            } else {
                Result.failure(Exception("Group not found"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun getVolunteers(): Result<List<Volunteer>> = withContext(Dispatchers.IO) {
        Result.success(volunteers.toList())
    }

    suspend fun registerVolunteer(volunteer: Volunteer): Result<Volunteer> = withContext(Dispatchers.IO) {
        try {
            val newVolunteer = volunteer.copy(id = System.currentTimeMillis().toString())
            volunteers.add(newVolunteer)
            Result.success(newVolunteer)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun submitHelpRequest(request: HelpRequest): Result<HelpRequest> = withContext(Dispatchers.IO) {
        try {
            val newRequest = request.copy(id = System.currentTimeMillis().toString())
            helpRequests.add(newRequest)
            Result.success(newRequest)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun getHelpRequests(): Result<List<HelpRequest>> = withContext(Dispatchers.IO) {
        Result.success(helpRequests.toList())
    }
}
