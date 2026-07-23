package dev.wceng.sufei.fork.sopho.data.tts

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import javax.inject.Inject
import javax.inject.Singleton

interface TtsEngine {
    val isPlaying: StateFlow<Boolean>
    val currentSentenceIndex: StateFlow<Int?>
    fun speak(sentences: List<String>)
    fun stop()
    fun release()
}

@Singleton
class ForkTtsOrchestrator @Inject constructor(
    private val systemTts: dev.wceng.sufei.data.tts.TtsManager,
    private val sherpaTts: SherpaOnnxTtsManager,
) {
    enum class Engine { SYSTEM, SHERPA_ONNX }

    private val _activeEngine = MutableStateFlow(Engine.SHERPA_ONNX)
    val activeEngine: StateFlow<Engine> = _activeEngine.asStateFlow()

    val isPlaying: StateFlow<Boolean> get() = currentEngine.isPlaying
    val currentSentenceIndex: StateFlow<Int?> get() = currentEngine.currentSentenceIndex

    val isSherpaReady: StateFlow<Boolean> = sherpaTts.isReady

    private val currentEngine: TtsEngine
        get() = when (_activeEngine.value) {
            Engine.SYSTEM -> object : TtsEngine {
                override val isPlaying = systemTts.isPlaying
                override val currentSentenceIndex = systemTts.currentSentenceIndex
                override fun speak(sentences: List<String>) = systemTts.speak(sentences)
                override fun stop() = systemTts.stop()
                override fun release() = systemTts.release()
            }
            Engine.SHERPA_ONNX -> object : TtsEngine {
                override val isPlaying = sherpaTts.isPlaying
                override val currentSentenceIndex = sherpaTts.currentSentenceIndex
                override fun speak(sentences: List<String>) = sherpaTts.speak(sentences)
                override fun stop() = sherpaTts.stop()
                override fun release() = sherpaTts.release()
            }
        }

    init {
        sherpaTts.init()
    }

    fun setEngine(engine: Engine) {
        if (_activeEngine.value != engine) {
            currentEngine.stop()
            _activeEngine.value = engine
        }
    }

    fun speak(sentences: List<String>) {
        currentEngine.speak(sentences)
    }

    fun stop() {
        currentEngine.stop()
    }

    fun release() {
        systemTts.release()
        sherpaTts.release()
    }
}
