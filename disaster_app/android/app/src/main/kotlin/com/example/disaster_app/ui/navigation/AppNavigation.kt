package com.example.disaster_app.ui.navigation

import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.example.disaster_app.ui.family.FamilyPage
import com.example.disaster_app.ui.help.HelpPage
import com.example.disaster_app.ui.home.HomePage
import com.example.disaster_app.ui.report.ReportPage
import com.example.disaster_app.ui.safety.SafetyPage
import com.example.disaster_app.ui.volunteer.VolunteerPage

sealed class Screen(
    val route: String,
    val title: String,
    val icon: ImageVector
) {
    data object Home : Screen("home", "首页", Icons.Default.Home)
    data object Family : Screen("family", "家庭", Icons.Default.People)
    data object Help : Screen("help", "求助", Icons.Default.Warning)
    data object Report : Screen("report", "上报", Icons.Default.Description)
    data object Safety : Screen("safety", "安全", Icons.Default.Security)
    data object Volunteer : Screen("volunteer", "志愿", Icons.Default.VolunteerActivism)
}

val bottomNavItems = listOf(
    Screen.Home,
    Screen.Family,
    Screen.Help,
    Screen.Report,
    Screen.Safety,
    Screen.Volunteer
)

@Composable
fun AppNavigation() {
    val navController = rememberNavController()
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentDestination = navBackStackEntry?.destination

    Scaffold(
        bottomBar = {
            NavigationBar {
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
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = Screen.Home.route,
            modifier = Modifier.padding(innerPadding)
        ) {
            composable(Screen.Home.route) { HomePage() }
            composable(Screen.Family.route) { FamilyPage() }
            composable(Screen.Help.route) { HelpPage() }
            composable(Screen.Report.route) { ReportPage() }
            composable(Screen.Safety.route) { SafetyPage() }
            composable(Screen.Volunteer.route) { VolunteerPage() }
        }
    }
}
