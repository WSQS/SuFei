package dev.wceng.sufei.ui.navigation

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.MenuBook
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.Settings
import androidx.compose.ui.graphics.vector.ImageVector
import dev.wceng.sufei.R
import dev.wceng.sufei.fork.sopho.ui.navigation.Study  // fork-specific

enum class MainTab(val titleRes: Int, val icon: ImageVector) {
    Home(R.string.tab_home, Icons.Default.Home),
    Explore(R.string.tab_explore, Icons.Default.Search),
    Collection(R.string.tab_collection, Icons.Default.Favorite),
    Study(R.string.tab_study, Icons.Default.MenuBook),  // fork-specific
    Settings(R.string.tab_settings, Icons.Default.Settings)
}

fun MainTab.toRoute(): Any = when (this) {
    MainTab.Home -> Home
    MainTab.Explore -> Explore()
    MainTab.Collection -> Collection
    MainTab.Study -> Study  // fork-specific
    MainTab.Settings -> Settings
}
