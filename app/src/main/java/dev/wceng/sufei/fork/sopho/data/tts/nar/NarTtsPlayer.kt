package dev.wceng.sufei.fork.sopho.data.tts.nar

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Plays NAR TTS audio on Android.
 *
 * Wraps NarOnnxEngine + AudioTrack for coroutine-friendly playback.
 */
@Singleton
class NarTtsPlayer @Inject constructor(
    private val engine: NarOnnxEngine,
) {
    private val tag = "NarTtsPlayer"

    private val _isPlaying = MutableStateFlow(false)
    val isPlaying: StateFlow<Boolean> = _isPlaying.asStateFlow()

    private val _isReady = MutableStateFlow(false)
    val isReady: StateFlow<Boolean> = _isReady.asStateFlow()

    private var audioTrack: AudioTrack? = null

    /**
     * Synthesize and play Chinese text.
     * Runs inference on Dispatchers.Default, playback on AudioTrack.
     */
    suspend fun speak(text: String) = withContext(Dispatchers.Default) {
        if (_isPlaying.value) return@withContext

        _isPlaying.value = true
        try {
            val audio = engine.synthesize(text)
            playAudio(audio)
        } catch (e: Exception) {
            Log.e(tag, "TTS failed", e)
        } finally {
            _isPlaying.value = false
        }
    }

    fun stop() {
        audioTrack?.let {
            it.pause()
            it.stop()
            it.release()
        }
        audioTrack = null
        _isPlaying.value = false
    }

    private fun playAudio(samples: FloatArray) {
        val sampleRate = engine.sampleRate
        val bufferSize = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_FLOAT,
        ).coerceAtLeast(samples.size)

        audioTrack = AudioTrack(
            AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build(),
            AudioFormat.Builder()
                .setSampleRate(sampleRate)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .build(),
            bufferSize,
            AudioTrack.MODE_STATIC,
            AudioManager.AUDIO_SESSION_ID_GENERATE,
        )

        audioTrack?.let { track ->
            track.write(samples, 0, samples.size, AudioTrack.WRITE_BLOCKING)
            track.play()
            // Wait for playback to finish
            Thread.sleep((samples.size.toLong() * 1000 / sampleRate) + 100)
        }

        audioTrack?.release()
        audioTrack = null
    }
}
