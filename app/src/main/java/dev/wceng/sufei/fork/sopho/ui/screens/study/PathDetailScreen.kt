package dev.wceng.sufei.fork.sopho.ui.screens.study

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.RadioButtonUnchecked
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.nestedscroll.nestedScroll
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import dev.wceng.sufei.R
import dev.wceng.sufei.fork.sopho.data.model.PathItem
import dev.wceng.sufei.fork.sopho.data.model.ProgressState
import dev.wceng.sufei.fork.sopho.data.model.ReadingPath

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PathDetailScreen(
    onBack: () -> Unit,
    onPoemClick: (String) -> Unit,
    viewModel: PathDetailViewModel = hiltViewModel(),
) {
    val uiState by viewModel.uiState.collectAsState()
    val scrollBehavior = TopAppBarDefaults.enterAlwaysScrollBehavior()

    Scaffold(
        modifier = Modifier.nestedScroll(scrollBehavior.nestedScrollConnection),
        topBar = {
            TopAppBar(
                title = {
                    val title = (uiState as? PathDetailUiState.Success)?.path?.title
                        ?: stringResource(R.string.tab_study)
                    Text(
                        text = title,
                        style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold),
                    )
                },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = stringResource(R.string.action_back),
                        )
                    }
                },
                scrollBehavior = scrollBehavior,
            )
        },
        containerColor = MaterialTheme.colorScheme.background,
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding),
        ) {
            when (val state = uiState) {
                is PathDetailUiState.Loading -> {
                    CircularProgressIndicator(modifier = Modifier.align(Alignment.Center))
                }

                is PathDetailUiState.Error -> {
                    Text(
                        text = stringResource(R.string.study_path_not_found),
                        color = MaterialTheme.colorScheme.error,
                        modifier = Modifier.align(Alignment.Center),
                    )
                }

                is PathDetailUiState.Success -> {
                    PathDetailContent(
                        path = state.path,
                        items = state.items,
                        onPoemClick = { poemId ->
                            viewModel.markRead(poemId)
                            onPoemClick(poemId)
                        },
                        onToggleRead = { poemId, isRead ->
                            if (isRead) viewModel.markUnread(poemId)
                            else viewModel.markRead(poemId)
                        },
                    )
                }
            }
        }
    }
}

@Composable
private fun PathDetailContent(
    path: ReadingPath,
    items: List<PathItem>,
    onPoemClick: (String) -> Unit,
    onToggleRead: (String, Boolean) -> Unit,
) {
    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(
            horizontal = 16.dp,
            vertical = 8.dp,
        ),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        item {
            PathSummary(path = path)
        }

        items(items, key = { it.poem.id }) { item ->
            PathItemRow(
                item = item,
                onClick = { onPoemClick(item.poem.id) },
                onToggleRead = { onToggleRead(item.poem.id, item.isRead) },
            )
        }
    }
}

@Composable
private fun PathSummary(path: ReadingPath) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
    ) {
        Text(
            text = path.description,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        Spacer(modifier = Modifier.height(14.dp))

        val progressText = when (path.progressState) {
            ProgressState.Completed -> stringResource(R.string.study_progress_completed, path.total)
            ProgressState.NotStarted -> stringResource(R.string.study_progress_not_started, path.total)
            ProgressState.InProgress -> stringResource(R.string.study_progress_in_progress, path.readCount, path.total)
        }
        Text(
            text = progressText,
            style = MaterialTheme.typography.labelLarge,
            color = MaterialTheme.colorScheme.primary,
            fontWeight = FontWeight.Bold,
        )
        Spacer(modifier = Modifier.height(6.dp))
        LinearProgressIndicator(
            progress = { path.progress },
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

@Composable
private fun PathItemRow(
    item: PathItem,
    onClick: () -> Unit,
    onToggleRead: () -> Unit,
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        colors = CardDefaults.cardColors(
            containerColor = if (item.isRead) {
                MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f)
            } else {
                MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
            },
        ),
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = item.poem.title,
                    style = MaterialTheme.typography.titleMedium.copy(
                        fontWeight = FontWeight.Bold,
                        color = if (item.isRead) {
                            MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)
                        } else {
                            MaterialTheme.colorScheme.onSurface
                        },
                        ),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "${item.poem.dynasty} · ${item.poem.author}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.primary,
                )
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = item.poem.content.replace("\n", " "),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis,
                )
            }

            Spacer(modifier = Modifier.width(12.dp))

            IconButton(onClick = onToggleRead) {
                Icon(
                    imageVector = if (item.isRead) {
                        Icons.Default.CheckCircle
                    } else {
                        Icons.Default.RadioButtonUnchecked
                    },
                    contentDescription = if (item.isRead) stringResource(R.string.study_mark_unread) else stringResource(R.string.study_mark_read),
                    tint = if (item.isRead) {
                        MaterialTheme.colorScheme.primary
                    } else {
                        MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.4f)
                    },
                )
            }
        }
    }
}
