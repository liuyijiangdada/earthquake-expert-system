package com.example.disaster_app.ui.theme

import android.app.Activity
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

private val EmergencyDarkScheme = darkColorScheme(
    primary = AccentBlue,
    onPrimary = TextPrimary,
    secondary = PhasePre,
    tertiary = PhasePost,
    background = BgDeep,
    onBackground = TextPrimary,
    surface = BgCard,
    onSurface = TextPrimary,
    surfaceVariant = BgChat,
    onSurfaceVariant = TextMuted,
    error = DangerRed,
    errorContainer = Color(0x33F87171),
    onErrorContainer = DangerRed,
    outline = BotBubbleBorder,
)

@Composable
fun DisasterAppTheme(
    content: @Composable () -> Unit
) {
    val colorScheme = EmergencyDarkScheme
    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = BgDeep.toArgb()
            window.navigationBarColor = BgDeep.toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = false
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        content = content
    )
}
