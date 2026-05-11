package com.example.disaster_app.ui.safety

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.LocationOn
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.disaster_app.data.model.SafetyLevel
import com.example.disaster_app.ui.theme.Green500
import com.example.disaster_app.ui.theme.Red500
import com.example.disaster_app.ui.theme.Yellow500
import com.example.disaster_app.viewmodel.SafetyViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SafetyPage(
    viewModel: SafetyViewModel = viewModel()
) {
    val uiState by viewModel.uiState.collectAsState()
    var selectedLevel by remember { mutableStateOf<SafetyLevel?>(null) }
    var location by remember { mutableStateOf("") }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "安全状态",
            style = MaterialTheme.typography.headlineMedium,
            color = MaterialTheme.colorScheme.primary
        )

        Spacer(modifier = Modifier.height(32.dp))

        uiState.currentStatus.status.let { currentStatus ->
            SafetyStatusCard(currentStatus = currentStatus)
        }

        Spacer(modifier = Modifier.height(32.dp))

        Text(
            text = "更新安全状态",
            style = MaterialTheme.typography.titleMedium
        )

        Spacer(modifier = Modifier.height(16.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            SafetyLevel.entries.filter { it != SafetyLevel.UNKNOWN }.forEach { level ->
                SafetyLevelButton(
                    level = level,
                    selected = selectedLevel == level,
                    onClick = { selectedLevel = level }
                )
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        OutlinedTextField(
            value = location,
            onValueChange = { location = it },
            label = { Text("当前位置") },
            leadingIcon = { Icon(Icons.Default.LocationOn, contentDescription = null) },
            singleLine = true,
            modifier = Modifier.fillMaxWidth()
        )

        Spacer(modifier = Modifier.height(24.dp))

        Button(
            onClick = {
                selectedLevel?.let { level ->
                    viewModel.updateStatus(level, location)
                }
            },
            enabled = selectedLevel != null && location.isNotBlank(),
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("更新状态")
        }
    }
}

@Composable
fun SafetyStatusCard(currentStatus: SafetyLevel) {
    val (color, icon, text) = when (currentStatus) {
        SafetyLevel.SAFE -> Triple(Green500, Icons.Default.Check, "安全")
        SafetyLevel.WARNING -> Triple(Yellow500, Icons.Default.Warning, "警告")
        SafetyLevel.DANGER -> Triple(Red500, Icons.Default.Warning, "危险")
        SafetyLevel.UNKNOWN -> Triple(Color.Gray, Icons.Default.Warning, "未知")
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = color.copy(alpha = 0.1f)
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Surface(
                shape = CircleShape,
                color = color,
                modifier = Modifier.size(80.dp)
            ) {
                Box(contentAlignment = Alignment.Center) {
                    Icon(
                        imageVector = icon,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(48.dp)
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            Text(
                text = text,
                fontSize = 28.sp,
                fontWeight = FontWeight.Bold,
                color = color
            )

            Text(
                text = when (currentStatus) {
                    SafetyLevel.SAFE -> "目前未发现危险，请继续保持警惕"
                    SafetyLevel.WARNING -> "请注意安全，关注后续通知"
                    SafetyLevel.DANGER -> "请立即转移到安全区域"
                    SafetyLevel.UNKNOWN -> "请更新您的安全状态"
                },
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
fun SafetyLevelButton(
    level: SafetyLevel,
    selected: Boolean,
    onClick: () -> Unit
) {
    val color = when (level) {
        SafetyLevel.SAFE -> Green500
        SafetyLevel.WARNING -> Yellow500
        SafetyLevel.DANGER -> Red500
        SafetyLevel.UNKNOWN -> Color.Gray
    }

    val text = when (level) {
        SafetyLevel.SAFE -> "安全"
        SafetyLevel.WARNING -> "警告"
        SafetyLevel.DANGER -> "危险"
        SafetyLevel.UNKNOWN -> "未知"
    }

    FilledTonalButton(
        onClick = onClick,
        colors = ButtonDefaults.filledTonalButtonColors(
            containerColor = if (selected) color else MaterialTheme.colorScheme.surfaceVariant
        ),
        modifier = Modifier.size(80.dp)
    ) {
        Text(
            text = text,
            color = if (selected) Color.White else MaterialTheme.colorScheme.onSurfaceVariant,
            fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal
        )
    }
}
