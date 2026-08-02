package com.example.disaster_app.ui.navigation

import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.example.disaster_app.ui.family.FamilyPage
import com.example.disaster_app.ui.help.HelpPage
import com.example.disaster_app.ui.home.HomePage
import com.example.disaster_app.ui.more.MorePage
import com.example.disaster_app.ui.report.ReportPage
import com.example.disaster_app.ui.safety.SafetyPage
import com.example.disaster_app.ui.volunteer.VolunteerPage
import com.example.disaster_app.ui.theme.BgCard
import com.example.disaster_app.ui.theme.BgDeep

sealed class Screen(
    val route: String,
    val title: String,
    val icon: ImageVector
) {
    data object Home : Screen("home", "问答", Icons.Default.Forum)
    data object Safety : Screen("safety", "指引", Icons.Default.MenuBook)
    data object More : Screen("more", "更多", Icons.Default.MoreHoriz)
    data object Help : Screen("help", "求助", Icons.Default.Warning)
    data object Family : Screen("family", "家人", Icons.Default.People)
    data object Report : Screen("report", "上报", Icons.Default.Description)
    data object Volunteer : Screen("volunteer", "志愿", Icons.Default.VolunteerActivism)
}

val bottomNavItems = listOf(Screen.Home, Screen.Safety, Screen.More)

@Composable
fun AppNavigation(
    username: String = "",
    onLogout: () -> Unit = {}
) {
    val navController = rememberNavController()
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentDestination = navBackStackEntry?.destination
    val hideBottomBar = currentDestination?.route in setOf(
        Screen.Help.route, Screen.Family.route, Screen.Report.route, Screen.Volunteer.route
    )

    Scaffold(
        containerColor = BgDeep,
        bottomBar = {
            if (!hideBottomBar) {
                NavigationBar(
                    containerColor = BgCard,
                    tonalElevation = 0.dp
                ) {
                    bottomNavItems.forEach { screen ->
                        NavigationBarItem(
                            icon = { Icon(screen.icon, contentDescription = screen.title) },
                            label = { Text(screen.title) },
                            selected = currentDestination?.hierarchy?.any { it.route == screen.route } == true,
                            onClick = {
                                navController.navigate(screen.route) {
                                    popUpTo(navController.graph.findStartDestination().id) {
                                        saveState = true
                                    }
                                    launchSingleTop = true
                                    restoreState = true
                                }
                            }
                        )
                    }
                }
            }
        }
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = Screen.Home.route,
            modifier = Modifier.padding(innerPadding)
        ) {
            composable(Screen.Home.route) { HomePage() }
            composable(Screen.Safety.route) { SafetyPage() }
            composable(Screen.More.route) {
                MorePage(
                    username = username,
                    onOpen = { route -> navController.navigate(route) },
                    onLogout = onLogout
                )
            }
            composable(Screen.Help.route) { HelpPage() }
            composable(Screen.Family.route) { FamilyPage() }
            composable(Screen.Report.route) { ReportPage() }
            composable(Screen.Volunteer.route) { VolunteerPage() }
        }
    }
}
