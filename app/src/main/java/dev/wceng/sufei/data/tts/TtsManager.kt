package dev.wceng.sufei.data.tts

import android.content.Context
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.Locale
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class TtsManager @Inject constructor(
    @ApplicationContext private val context: Context
) {
    private var tts: TextToSpeech? = null
    private val _isPlaying = MutableStateFlow(false)
    val isPlaying: StateFlow<Boolean> = _isPlaying.asStateFlow()

    private val _currentSentenceIndex = MutableStateFlow<Int?>(null)
    val currentSentenceIndex: StateFlow<Int?> = _currentSentenceIndex.asStateFlow()

    private var isInitialized = false

    companion object {
        private const val DEFAULT_SPEECH_RATE = 0.9f
        private const val DEFAULT_PITCH = 0.95f

        private const val PAUSE_AFTER_TITLE = 600L
        private const val PAUSE_AFTER_DYNASTY = 200L
        private const val PAUSE_AFTER_AUTHOR = 600L
        private const val PAUSE_COMMA = 250L
        private const val PAUSE_SENTENCE_END = 500L
        private const val PAUSE_DEFAULT = 350L
    }

    private fun initTts(onInitComplete: () -> Unit = {}) {
        if (isInitialized) {
            onInitComplete()
            return
        }
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.let {
                    it.language = Locale.CHINESE
                    it.setSpeechRate(DEFAULT_SPEECH_RATE)
                    it.setPitch(DEFAULT_PITCH)
                    it.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                        override fun onStart(utteranceId: String?) {
                            _isPlaying.value = true
                            if (utteranceId?.startsWith("sentence_") == true) {
                                _currentSentenceIndex.value = utteranceId
                                    .substringAfter("sentence_")
                                    .removeSuffix("_end")
                                    .toIntOrNull()
                            }
                        }

                        override fun onDone(utteranceId: String?) {
                            if (utteranceId?.startsWith("sentence_") == true) {
                                // 如果是最后一段，标记播放结束
                                if (utteranceId.endsWith("_end")) {
                                    _isPlaying.value = false
                                    _currentSentenceIndex.value = null
                                }
                            }
                        }

                        @Deprecated("Deprecated in Java")
                        override fun onError(utteranceId: String?) {
                            _isPlaying.value = false
                            _currentSentenceIndex.value = null
                        }
                    })
                    isInitialized = true
                    onInitComplete()
                }
            }
        }
    }

    fun speak(sentences: List<String>) {
        _isPlaying.value = true
        _currentSentenceIndex.value = 0
        if (!isInitialized) {
            initTts {
                if (_isPlaying.value) {
                    speakSentences(sentences)
                }
            }
        } else {
            speakSentences(sentences)
        }
    }

    private fun speakSentences(sentences: List<String>) {
        sentences.forEachIndexed { index, sentence ->
            val mode = if (index == 0) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD
            val isLast = index == sentences.size - 1
            val id = "sentence_$index${if (isLast) "_end" else ""}"
            tts?.speak(sentence, mode, null, id)

            if (!isLast) {
                val pauseMs = computePause(index, sentence)
                if (pauseMs > 0) {
                    tts?.playSilentUtterance(pauseMs, TextToSpeech.QUEUE_ADD, "silence_$index")
                }
            }
        }
    }

    private fun computePause(index: Int, sentence: String): Long {
        return when (index) {
            0 -> PAUSE_AFTER_TITLE
            1 -> PAUSE_AFTER_DYNASTY
            2 -> PAUSE_AFTER_AUTHOR
            else -> {
                val trimmed = sentence.trimEnd()
                when {
                    trimmed.endsWith('，') || trimmed.endsWith('、') -> PAUSE_COMMA
                    trimmed.endsWith('。') || trimmed.endsWith('！') ||
                        trimmed.endsWith('？') || trimmed.endsWith('；') -> PAUSE_SENTENCE_END
                    else -> PAUSE_DEFAULT
                }
            }
        }
    }

    fun stop() {
        _isPlaying.value = false
        _currentSentenceIndex.value = null
        tts?.stop()
    }

    fun release() {
        tts?.stop()
        tts?.shutdown()
        tts = null
        isInitialized = false
        _isPlaying.value = false
    }
}
