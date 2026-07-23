package dev.wceng.sufei.fork.sopho.data.tts

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.util.Log
import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.OfflineTtsConfig
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsVitsModelConfig
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.File
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class SherpaOnnxTtsManager @Inject constructor(
    @ApplicationContext private val context: Context
) {
    companion object {
        private const val TAG = "SherpaOnnxTts"
        private const val MODEL_ASSET_DIR = "sherpa_tts"
        private const val MODEL_FILE = "zh_CN-xiao_ya-medium.onnx"
        private const val TOKENS_FILE = "tokens.txt"
        private const val LEXICON_FILE = "lexicon.txt"
        private const val RULE_FSTS = "date.fst,number.fst,phone.fst"
    }

    private val _isPlaying = MutableStateFlow(false)
    val isPlaying: StateFlow<Boolean> = _isPlaying.asStateFlow()

    private val _currentSentenceIndex = MutableStateFlow<Int?>(null)
    val currentSentenceIndex: StateFlow<Int?> = _currentSentenceIndex.asStateFlow()

    private val _isReady = MutableStateFlow(false)
    val isReady: StateFlow<Boolean> = _isReady.asStateFlow()

    private var tts: OfflineTts? = null
    private var audioTrack: AudioTrack? = null
    private var modelDir: String? = null
    private var stopped = false
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    fun init() {
        if (tts != null) return
        scope.launch {
            try {
                modelDir = copyAssetsToInternal()
                val dir = modelDir!!
                val config = OfflineTtsConfig(
                    model = OfflineTtsModelConfig(
                        vits = OfflineTtsVitsModelConfig(
                            model = "$dir/$MODEL_FILE",
                            lexicon = "$dir/$LEXICON_FILE",
                            tokens = "$dir/$TOKENS_FILE",
                            dataDir = dir,
                        ),
                        numThreads = 2,
                        debug = false,
                        provider = "cpu",
                    ),
                    ruleFsts = RULE_FSTS.split(",").joinToString(",") { "$dir/$it" },
                    maxNumSentences = 1,
                    silenceScale = 0.2f,
                )
                tts = OfflineTts(config = config)
                initAudioTrack(tts!!.sampleRate())
                _isReady.value = true
                Log.i(TAG, "Sherpa-ONNX TTS initialized, sampleRate=${tts!!.sampleRate()}")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to init Sherpa-ONNX TTS", e)
            }
        }
    }

    private fun copyAssetsToInternal(): String {
        val destDir = File(context.filesDir, MODEL_ASSET_DIR)
        if (!destDir.exists()) destDir.mkdirs()
        val files = listOf(MODEL_FILE, TOKENS_FILE, LEXICON_FILE) + RULE_FSTS.split(",")
        for (f in files) {
            val dest = File(destDir, f)
            if (!dest.exists()) {
                context.assets.open("$MODEL_ASSET_DIR/$f").use { input ->
                    dest.outputStream().use { output -> input.copyTo(output) }
                }
            }
        }
        return destDir.absolutePath
    }

    private fun initAudioTrack(sampleRate: Int) {
        val bufLength = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_FLOAT,
        )
        val attr = AudioAttributes.Builder()
            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
            .setUsage(AudioAttributes.USAGE_MEDIA)
            .build()
        val format = AudioFormat.Builder()
            .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
            .setSampleRate(sampleRate)
            .build()
        audioTrack = AudioTrack(
            attr, format, bufLength, AudioTrack.MODE_STREAM,
            AudioManager.AUDIO_SESSION_ID_GENERATE,
        )
    }

    fun speak(sentences: List<String>) {
        if (tts == null) {
            Log.w(TAG, "TTS not ready yet")
            return
        }
        _isPlaying.value = true
        _currentSentenceIndex.value = 0
        stopped = false
        scope.launch {
            val track = audioTrack ?: return@launch
            track.play()
            sentences.forEachIndexed { index, sentence ->
                if (stopped) return@forEachIndexed
                _currentSentenceIndex.value = index
                tts?.generateWithCallback(
                    text = sentence,
                    sid = 0,
                    speed = 0.9f,
                ) { samples ->
                    if (!stopped) {
                        track.write(samples, 0, samples.size, AudioTrack.WRITE_BLOCKING)
                        1
                    } else {
                        0
                    }
                }
            }
            if (!stopped) {
                _isPlaying.value = false
                _currentSentenceIndex.value = null
            }
            track.stop()
        }
    }

    fun stop() {
        stopped = true
        _isPlaying.value = false
        _currentSentenceIndex.value = null
        audioTrack?.pause()
        audioTrack?.flush()
    }

    fun release() {
        stopped = true
        audioTrack?.stop()
        audioTrack?.release()
        audioTrack = null
        tts?.release()
        tts = null
        _isReady.value = false
        _isPlaying.value = false
        _currentSentenceIndex.value = null
    }
}
