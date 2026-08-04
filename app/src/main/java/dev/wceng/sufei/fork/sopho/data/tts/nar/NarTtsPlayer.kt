package dev.wceng.sufei.fork.sopho.data.tts.nar

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Plays NAR TTS audio on Android.
 *
 * Wraps NarOnnxEngine + AudioTrack for coroutine-friendly playback. [speak]
 * suspends for the full duration of playback so [isPlaying] tracks the real
 * playing state (true from synthesis start until the audio drains, an error, or
 * [stop]) rather than flipping false the instant the async AudioTrack starts.
 */
@Singleton
class NarTtsPlayer @Inject constructor(
    private val engine: NarOnnxEngine,
) {
    private val tag = "NarTtsPlayer"

    private val _isPlaying = MutableStateFlow(false)
    val isPlaying: StateFlow<Boolean> = _isPlaying.asStateFlow()

    private val lock = Any()
    private var audioTrack: AudioTrack? = null

    /**
     * Synthesize and play Chinese text. No-op if already playing.
     * Runs inference on Dispatchers.Default; suspends until playback finishes,
     * is cancelled, or [stop] is called.
     */
    suspend fun speak(text: String) {
        if (_isPlaying.value) return
        _isPlaying.value = true
        try {
            val audio = withContext(Dispatchers.Default) { engine.synthesize(text) }
            playAndAwait(audio)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.e(tag, "TTS failed", e)
        } finally {
            releaseTrack()
            _isPlaying.value = false
        }
    }

    fun stop() {
        releaseTrack()
        _isPlaying.value = false
    }

    private fun releaseTrack() {
        synchronized(lock) {
            audioTrack?.let {
                runCatching { it.pause() }
                runCatching { it.stop() }
                it.release()
            }
            audioTrack = null
        }
    }

    /**
     * Play [samples] and suspend until they drain. Uses a MODE_STATIC track and
     * polls the playback head (more reliable across OEMs than the marker
     * callback), with a duration-based guard so a stalled head can't hang the
     * coroutine forever. Bails out immediately once [stop] releases the track.
     */
    private suspend fun playAndAwait(samples: FloatArray) {
        if (samples.isEmpty()) return
        val sampleRate = engine.sampleRate
        val track = buildTrack(samples, sampleRate)
        synchronized(lock) { audioTrack = track }

        track.write(samples, 0, samples.size, AudioTrack.WRITE_BLOCKING)
        track.play()

        val guardPolls = (samples.size * 1000L / sampleRate + 1000L) / POLL_MS
        var polls = 0L
        while (currentCoroutineContext().isActive && polls < guardPolls) {
            // stop() nulls/replaces audioTrack; bail before touching a released track.
            val stillActive = synchronized(lock) { audioTrack === track }
            if (!stillActive) break
            if (track.playbackHeadPosition >= samples.size) break
            delay(POLL_MS)
            polls++
        }
    }

    private fun buildTrack(samples: FloatArray, sampleRate: Int): AudioTrack {
        val bufferSize = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_FLOAT,
        ).coerceAtLeast(samples.size * Float.SIZE_BYTES)

        return AudioTrack(
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
    }

    private companion object {
        const val POLL_MS = 50L
    }
}
