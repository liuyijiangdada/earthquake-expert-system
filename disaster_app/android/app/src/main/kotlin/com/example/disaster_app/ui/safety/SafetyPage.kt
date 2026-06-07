package com.example.disaster_app.ui.safety

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.disaster_app.data.model.SafetyLevel
import com.example.disaster_app.ui.components.EmergencyTopBar
import com.example.disaster_app.ui.components.SectionTitle
import com.example.disaster_app.ui.theme.*
import com.example.disaster_app.viewmodel.SafetyViewModel

data class PhaseGuide(val phase: String, val color: Color, val icon: ImageVector, val tips: List<String>)

private val phaseGuides = listOf(
    PhaseGuide(
        "震前",
        PhasePre,
        Icons.Default.Shield,
        listOf("准备应急包与家庭联络方式", "固定家具、熟悉疏散路线", "关注官方科普与预警开通方式")
    ),
    PhaseGuide(
        "震中",
        PhaseDuring,
        Icons.Default.Bolt,
        listOf("室内：伏地遮挡抓牢", "室外：远离建筑到空旷处", "勿乘电梯，注意余震")
    ),
    PhaseGuide(
        "震后",
        PhasePost,
        Icons.Default.Favorite,
        listOf("检查房屋结构是否安全", "防范余震与次生灾害", "关注政府救助与复课通知")
    )
)

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
            .background(BgDeep)
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        EmergencyTopBar(
            title = "应急安全指引",
            subtitle = "三阶段要点 + 个人安全状态上报"
        )

        Spacer(modifier = Modifier.height(16.dp))

        SectionTitle(title = "震前 · 震中 · 震后 要点", icon = Icons.Default.MenuBook)

        Spacer(modifier = Modifier.height(8.dp))

        phaseGuides.forEach { guide ->
            PhaseGuideCard(guide)
            Spacer(modifier = Modifier.height(8.dp))
        }

        Spacer(modifier = Modifier.height(16.dp))

        uiState.currentStatus.status.let { currentStatus ->
            SafetyStatusCard(currentStatus = currentStatus)
        }

        Spacer(modifier = Modifier.height(24.dp))

        Text(
            text = "上报我的安全状态",
            style = MaterialTheme.typography.titleMedium,
            color = TextPrimary,
            fontWeight = FontWeight.SemiBold
        )

        Spacer(modifier = Modifier.height(12.dp))

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

        Spacer(modifier = Modifier.height(12.dp))

        OutlinedTextField(
            value = location,
            onValueChange = { location = it },
            label = { Text("当前位置") },
            leadingIcon = { Icon(Icons.Default.LocationOn, contentDescription = null) },
            singleLine = true,
            modifier = Modifier.fillMaxWidth(),
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = AccentBlue,
                focusedTextColor = TextPrimary,
                unfocusedTextColor = TextPrimary
            )
        )

        Spacer(modifier = Modifier.height(16.dp))

        Button(
            onClick = {
                selectedLevel?.let { level ->
                    viewModel.updateStatus(level, location)
                }
            },
            enabled = selectedLevel != null && location.isNotBlank(),
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.buttonColors(containerColor = AccentBlueDim)
        ) {
            Text("更新状态")
        }
    }
}

@Composable
fun PhaseGuideCard(guide: PhaseGuide) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = BgCard),
        border = androidx.compose.foundation.BorderStroke(1.dp, guide.color.copy(alpha = 0.35f))
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(guide.icon, contentDescription = null, tint = guide.color, modifier = Modifier.size(20.dp))
                Spacer(modifier = Modifier.width(8.dp))
                Text(guide.phase, fontWeight = FontWeight.Bold, color = guide.color)
            }
            Spacer(modifier = Modifier.height(8.dp))
            guide.tips.forEach { tip ->
                Text("· $tip", style = MaterialTheme.typography.bodySmall, color = TextMuted)
            }
        }
    }
}

@Composable
fun SafetyStatusCard(currentStatus: SafetyLevel) {
    val (color, icon, text) = when (currentStatus) {
        SafetyLevel.SAFE -> Triple(SafeGreen, Icons.Default.Check, "安全")
        SafetyLevel.WARNING -> Triple(WarnAmber, Icons.Default.Warning, "注意")
        SafetyLevel.DANGER -> Triple(DangerRed, Icons.Default.Warning, "危险")
        SafetyLevel.UNKNOWN -> Triple(PhaseGeneral, Icons.Default.Warning, "未知")
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = color.copy(alpha = 0.12f)),
        border = androidx.compose.foundation.BorderStroke(1.dp, color.copy(alpha = 0.4f))
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Surface(shape = CircleShape, color = color, modifier = Modifier.size(72.dp)) {
                Box(contentAlignment = Alignment.Center) {
                    Icon(icon, contentDescription = null, tint = Color.White, modifier = Modifier.size(40.dp))
                }
            }
            Spacer(modifier = Modifier.height(12.dp))
            Text(text = text, fontSize = 26.sp, fontWeight = FontWeight.Bold, color = color)
            Text(
                text = when (currentStatus) {
                    SafetyLevel.SAFE -> "目前未发现危险，请保持警惕"
                    SafetyLevel.WARNING -> "请关注后续通知，做好避险准备"
                    SafetyLevel.DANGER -> "请立即转移到安全区域"
                    SafetyLevel.UNKNOWN -> "请更新您的安全状态"
                },
                style = MaterialTheme.typography.bodyMedium,
                color = TextMuted
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
        SafetyLevel.SAFE -> SafeGreen
        SafetyLevel.WARNING -> WarnAmber
        SafetyLevel.DANGER -> DangerRed
        SafetyLevel.UNKNOWN -> PhaseGeneral
    }
    val text = when (level) {
        SafetyLevel.SAFE -> "安全"
        SafetyLevel.WARNING -> "注意"
        SafetyLevel.DANGER -> "危险"
        SafetyLevel.UNKNOWN -> "未知"
    }

    FilledTonalButton(
        onClick = onClick,
        colors = ButtonDefaults.filledTonalButtonColors(
            containerColor = if (selected) color else BgChat
        ),
        modifier = Modifier.size(76.dp)
    ) {
        Text(
            text = text,
            color = if (selected) Color.White else TextMuted,
            fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal
        )
    }
}
