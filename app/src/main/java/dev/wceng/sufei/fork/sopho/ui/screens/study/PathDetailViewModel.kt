package dev.wceng.sufei.fork.sopho.ui.screens.study

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dagger.assisted.Assisted
import dagger.assisted.AssistedFactory
import dagger.assisted.AssistedInject
import dagger.hilt.android.lifecycle.HiltViewModel
import dev.wceng.sufei.fork.sopho.data.model.PathItem
import dev.wceng.sufei.fork.sopho.data.model.ReadingPath
import dev.wceng.sufei.fork.sopho.data.repository.ReadingPathRepository
import dev.wceng.sufei.fork.sopho.ui.navigation.PathDetail
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

@HiltViewModel(assistedFactory = PathDetailViewModel.Factory::class)
class PathDetailViewModel @AssistedInject constructor(
    private val readingPathRepository: ReadingPathRepository,
    @Assisted private val pathDetail: PathDetail,
) : ViewModel() {

    val uiState: StateFlow<PathDetailUiState> = combine(
        readingPathRepository.observePath(pathDetail.pathId),
        readingPathRepository.observePathItems(pathDetail.pathId),
    ) { path, items ->
        if (path == null) {
            PathDetailUiState.Error
        } else {
            PathDetailUiState.Success(path = path, items = items)
        }
    }.stateIn(
        scope = viewModelScope,
        started = SharingStarted.WhileSubscribed(5000),
        initialValue = PathDetailUiState.Loading,
    )

    fun markRead(poemId: String) {
        viewModelScope.launch {
            readingPathRepository.markRead(pathDetail.pathId, poemId)
        }
    }

    fun markUnread(poemId: String) {
        viewModelScope.launch {
            readingPathRepository.markUnread(pathDetail.pathId, poemId)
        }
    }

    @AssistedFactory
    interface Factory {
        fun create(pathDetail: PathDetail): PathDetailViewModel
    }
}

sealed interface PathDetailUiState {
    data object Loading : PathDetailUiState
    data object Error : PathDetailUiState
    data class Success(
        val path: ReadingPath,
        val items: List<PathItem>,
    ) : PathDetailUiState
}
