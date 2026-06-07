package com.example.disaster_app.ui.components

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.disaster_app.ui.theme.*

data class QuickPrompt(val label: String, val query: String, val phase: String)

val DefaultQuickPrompts = listOf(
    QuickPrompt("应急包", "家庭应急包该准备什么？", "震前"),
    QuickPrompt("室内避险", "地震发生时室内如何避险？", "震中"),
    QuickPrompt("避难所", "我在成都，最近的应急避难所在哪？", "震中"),
    QuickPrompt("最新震情", "刚才地震多大？震中在哪？", "震中"),
    QuickPrompt("房屋安全", "震后房屋安全怎么判断？", "震后"),
    QuickPrompt("救助政策", "灾后重建补贴政策有哪些？", "震后"),
)

@Composable
fun EmergencyTopBar(
    title: String,
    subtitle: String,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier.fillMaxWidth()) {
        Text(
            text = title,
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Bold,
            color = TextPrimary
        )
        Text(
            text = subtitle,
            style = MaterialTheme.typography.bodySmall,
            color = TextMuted
        )
    }
}

@Composable
fun PhaseBadge(phase: String?, urgency: Float = 0f, modifier: Modifier = Modifier) {
    if (phase.isNullOrBlank()) return
    Row(modifier = modifier, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        val (bg, fg, icon) = when (phase) {
            "震前" -> Triple(PhasePre.copy(alpha = 0.2f), PhasePre, Icons.Default.Shield)
            "震中" -> Triple(PhaseDuring.copy(alpha = 0.2f), PhaseDuring, Icons.Default.Bolt)
            "震后" -> Triple(PhasePost.copy(alpha = 0.2f), PhasePost, Icons.Default.Favorite)
            else -> Triple(PhaseGeneral.copy(alpha = 0.2f), PhaseGeneral, Icons.Default.Info)
        }
        Surface(shape = RoundedCornerShape(999.dp), color = bg) {
            Row(
                modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Icon(icon, contentDescription = null, tint = fg, modifier = Modifier.size(14.dp))
                Text(text = "$phase", style = MaterialTheme.typography.labelSmall, color = fg)
            }
        }
        if (urgency > 0.5f) {
            Surface(shape = RoundedCornerShape(999.dp), color = WarnAmber.copy(alpha = 0.2f)) {
                Text(
                    text = "紧急",
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                    style = MaterialTheme.typography.labelSmall,
                    color = WarnAmber
                )
            }
        }
    }
}

@Composable
fun QuickPromptRow(
    prompts: List<QuickPrompt> = DefaultQuickPrompts,
    onPromptClick: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        Text(
            text = "场景快捷提问",
            style = MaterialTheme.typography.labelLarge,
            color = TextMuted,
            modifier = Modifier.padding(bottom = 8.dp)
        )
        Row(
            modifier = Modifier.horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            prompts.forEach { p ->
                val borderColor = when (p.phase) {
                    "震前" -> PhasePre.copy(alpha = 0.5f)
                    "震中" -> PhaseDuring.copy(alpha = 0.5f)
                    "震后" -> PhasePost.copy(alpha = 0.5f)
                    else -> PhaseGeneral.copy(alpha = 0.5f)
                }
                SuggestionChip(
                    onClick = { onPromptClick(p.query) },
                    label = { Text(p.label) },
                    border = SuggestionChipDefaults.suggestionChipBorder(
                        enabled = true,
                        borderColor = borderColor
                    ),
                    colors = SuggestionChipDefaults.suggestionChipColors(
                        containerColor = BgChat,
                        labelColor = TextPrimary
                    )
                )
            }
        }
    }
}

@Composable
fun CapabilityTags(modifier: Modifier = Modifier) {
    Row(
        modifier = modifier.horizontalScroll(rememberScrollState()),
        horizontalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        listOf("KG", "RAG", "USGS", "高德", "多模态").forEach { tag ->
            Surface(
                shape = RoundedCornerShape(6.dp),
                color = Color.White.copy(alpha = 0.05f),
                border = androidx.compose.foundation.BorderStroke(1.dp, BotBubbleBorder)
            ) {
                Text(
                    text = tag,
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                    style = MaterialTheme.typography.labelSmall,
                    color = TextMuted
                )
            }
        }
    }
}

@Composable
fun SectionTitle(
    title: String,
    icon: ImageVector? = null,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier,
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        if (icon != null) {
            Icon(icon, contentDescription = null, tint = AccentBlue, modifier = Modifier.size(20.dp))
        }
        Text(
            text = title,
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold,
            color = TextPrimary
        )
    }
}
