package dev.wceng.sufei.fork.sopho.ui.screens.study

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import dev.wceng.sufei.fork.sopho.data.model.ReadingPath
import dev.wceng.sufei.fork.sopho.data.repository.ReadingPathRepository
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import javax.inject.Inject

/**
 * 研习 Tab：观察全部选集及其阅读进度。
 *
 * 首帧为 [StudyUiState.Loading]（stateIn 初始值）；仓库首次发射后
 * 一律进入 [StudyUiState.Success]（含空列表，由 UI 展示空态）。
 */
@HiltViewModel
class StudyViewModel @Inject constructor(
    readingPathRepository: ReadingPathRepository,
) : ViewModel() {

    val uiState: StateFlow<StudyUiState> = readingPathRepository.observeAllPaths()
        .map { paths -> StudyUiState.Success(paths) }
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = StudyUiState.Loading,
        )
}

sealed interface StudyUiState {
    data object Loading : StudyUiState
    data class Success(val paths: List<ReadingPath>) : StudyUiState
}
