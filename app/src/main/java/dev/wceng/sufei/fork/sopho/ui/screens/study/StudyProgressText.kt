package dev.wceng.sufei.fork.sopho.ui.screens.study

import androidx.compose.runtime.Composable
import androidx.compose.ui.res.stringResource
import dev.wceng.sufei.R
import dev.wceng.sufei.fork.sopho.data.model.ProgressState
import dev.wceng.sufei.fork.sopho.data.model.ReadingPath

/** 将 [ReadingPath.progressState] 映射为本地化进度文案。 */
@Composable
internal fun ReadingPath.progressText(): String = when (progressState) {
    ProgressState.Completed -> stringResource(R.string.study_progress_completed, total)
    ProgressState.NotStarted -> stringResource(R.string.study_progress_not_started, total)
    ProgressState.InProgress -> stringResource(R.string.study_progress_in_progress, readCount, total)
}
