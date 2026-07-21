package dev.wceng.sufei.ui.screens.splash

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import dev.wceng.sufei.data.repository.ImportRepository
import dev.wceng.sufei.data.repository.ImportState
import dev.wceng.sufei.fork.sopho.data.repository.ForkImportRepository  // fork-specific
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class SplashViewModel @Inject constructor(
    private val importRepository: ImportRepository,
    private val forkImportRepository: ForkImportRepository, // fork-specific
) : ViewModel() {

    val importState: StateFlow<ImportState> = importRepository.importState

    init {
        startImport()
    }

    private fun startImport() {
        viewModelScope.launch {
            importRepository.startImportIfNeeded()
            forkImportRepository.startImportIfNeeded() // fork-specific
        }
    }
}
