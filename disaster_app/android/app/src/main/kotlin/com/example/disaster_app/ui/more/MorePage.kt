package com.example.disaster_app.ui.more

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.People
import androidx.compose.material.icons.filled.VolunteerActivism
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.ListItem
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.disaster_app.ui.components.EmergencyTopBar
import com.example.disaster_app.ui.theme.BgDeep
import com.example.disaster_app.ui.theme.DangerRed
import com.example.disaster_app.ui.theme.TextMuted

@Composable
fun MorePage(
    username: String,
    onOpen: (route: String) -> Unit,
    onLogout: () -> Unit
) {
    val items = listOf(
        Triple("求助", "help", Icons.Default.Warning),
        Triple("家人", "family", Icons.Default.People),
        Triple("上报", "report", Icons.Default.Description),
        Triple("志愿", "volunteer", Icons.Default.VolunteerActivism)
    )
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(BgDeep)
            .padding(16.dp)
    ) {
        EmergencyTopBar(title = "更多", subtitle = "账号与应急辅助功能")
        Spacer(Modifier.height(8.dp))
        Text("当前用户：$username", color = TextMuted)
        Spacer(Modifier.height(16.dp))
        items.forEach { (title, route, icon) ->
            ListItem(
                headlineContent = { Text(title) },
                leadingContent = { Icon(icon, contentDescription = null) },
                modifier = Modifier.clickable { onOpen(route) }
            )
            HorizontalDivider()
        }
        Spacer(Modifier.height(24.dp))
        Button(
            onClick = onLogout,
            colors = ButtonDefaults.buttonColors(containerColor = DangerRed)
        ) {
            Text("退出登录")
        }
    }
}
