package dev.wceng.sufei.fork.sopho.data.tts.nar

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxValue
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer

/**
 * FastSpeech 2 + HiFi-GAN NAR TTS engine.
 *
 * Pipeline: text → G2P → phone IDs → FS2 ONNX → mel → (denorm) → HiFi-GAN → waveform
 *
 * @param modelDir directory containing FS2 ONNX, HiFiGAN ONNX, phone_id_map.txt
 * @param cpuThreads inference thread count
 * @param useSuFeiModel if true, loads custom SuFei FS2 (normalized mel output)
 */
class NarOnnxEngine(
    private val modelDir: File,
    private val cpuThreads: Int = 4,
    private val useSuFeiModel: Boolean = false,
) : Closeable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val sessionOptions = OrtSession.SessionOptions().apply {
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        setIntraOpNumThreads(cpuThreads.coerceAtLeast(1))
        setInterOpNumThreads(1)
    }

    private val fs2Session: OrtSession = createSession(
        if (useSuFeiModel) "fastspeech2_sufei.onnx" else "fastspeech2_csmsc.onnx"
    )
    private val hifiganSession: OrtSession = createSession("hifigan_csmsc.onnx")
    private val phoneIdMap: Map<String, Int> = loadPhoneIdMap()
    private val normStats: NormStats? = if (useSuFeiModel) loadNormStats() else null

    val sampleRate: Int = 24000

    /**
     * Synthesize speech from Chinese text.
     *
     * @param text Chinese text (may contain punctuation: ，。？！；：)
     * @return PCM float array at 24kHz mono
     */
    fun synthesize(text: String): FloatArray {
        val phoneIds = g2p(text)
        android.util.Log.d("NarOnnxEngine", "G2P: ${text.take(20)} → ${phoneIds.size} ids (suFei=$useSuFeiModel)")
        require(phoneIds.isNotEmpty()) { "No valid phonemes from text: $text" }
        val mel = runFastSpeech2(phoneIds)
        android.util.Log.d("NarOnnxEngine", "FS2: mel=${mel.size}x${if (mel.isNotEmpty()) mel[0].size else 0}")
        val melForVocoder = if (normStats != null) denormalizeMel(mel) else mel
        val audio = runHifiGan(melForVocoder)
        android.util.Log.d("NarOnnxEngine", "HiFiGAN: ${audio.size} samples")
        return audio
    }

    private fun g2p(text: String): IntArray {
        val phones = ChineseG2p.textToPhones(text)
        return phones.mapNotNull { phoneIdMap[it] }.toIntArray()
    }

    private fun runFastSpeech2(phoneIds: IntArray): Array<FloatArray> {
        val longIds = phoneIds.map { it.toLong() }.toLongArray()
        val input = OnnxTensor.createTensor(
            env,
            java.nio.LongBuffer.wrap(longIds),
            longArrayOf(longIds.size.toLong()),
        )
        input.use {
            val outputs = fs2Session.run(mapOf("text" to it))
            outputs.use { result ->
                @Suppress("UNCHECKED_CAST")
                val mel2d = result.requiredValue(0).value as Array<FloatArray>
                return mel2d
            }
        }
    }

    private fun runHifiGan(mel: Array<FloatArray>): FloatArray {
        val frames = mel.size
        val melBins = mel[0].size
        val flat = FloatArray(frames * melBins)
        var idx = 0
        for (frame in mel) {
            for (v in frame) {
                flat[idx++] = v
            }
        }
        val input = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(flat),
            longArrayOf(frames.toLong(), melBins.toLong()),
        )
        input.use {
            val outputs = hifiganSession.run(mapOf("logmel" to it))
            outputs.use { result ->
                val outputValue = result.requiredValue(0)
                android.util.Log.d("NarOnnxEngine", "HiFiGAN output type: ${outputValue.value?.javaClass?.simpleName}")
                // Output shape is [None, 1] — flatten everything
                val flatOutput = flattenToFloat(outputValue.value)
                return flatOutput
            }
        }
    }

    private fun createSession(fileName: String): OrtSession {
        val file = File(modelDir, fileName)
        require(file.isFile) { "Missing ONNX model: ${file.absolutePath}" }
        return env.createSession(file.absolutePath, sessionOptions)
    }

    private fun loadPhoneIdMap(): Map<String, Int> {
        val file = File(modelDir, "phone_id_map.txt")
        require(file.isFile) { "Missing phone_id_map.txt: ${file.absolutePath}" }
        return file.readLines().mapNotNull { line ->
            val parts = line.trim().split(" ")
            if (parts.size == 2) parts[0] to parts[1].toInt() else null
        }.toMap()
    }

    override fun close() {
        hifiganSession.close()
        fs2Session.close()
        sessionOptions.close()
    }

    // fork-sopho: mel denormalization for custom SuFei FS2 model
    private fun denormalizeMel(mel: Array<FloatArray>): Array<FloatArray> {
        val stats = normStats ?: return mel
        return mel.map { frame ->
            FloatArray(frame.size) { i -> frame[i] * stats.melStd[i] + stats.melMean[i] }
        }.toTypedArray()
    }

    private fun loadNormStats(): NormStats? {
        val file = File(modelDir, "norm_stats.npz")
        if (!file.isFile) return null
        java.util.zip.ZipFile(file).use { zf ->
            val melMean = readNpyFloatArray(zf.getInputStream(zf.getEntry("mel_mean.npy")))
            val melStd = readNpyFloatArray(zf.getInputStream(zf.getEntry("mel_std.npy")))
            android.util.Log.d("NarOnnxEngine", "Loaded norm_stats: mel_mean[${melMean.size}]")
            return NormStats(melMean, melStd)
        }
    }

    private fun readNpyFloatArray(stream: java.io.InputStream): FloatArray {
        val bytes = stream.readBytes()
        val headerLen = (bytes[8].toInt() and 0xFF) or ((bytes[9].toInt() and 0xFF) shl 8)
        val dataOffset = 10 + headerLen
        val nElements = (bytes.size - dataOffset) / 4
        val result = FloatArray(nElements)
        val bb = java.nio.ByteBuffer.wrap(bytes, dataOffset, nElements * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until nElements) result[i] = bb.float
        return result
    }

    private data class NormStats(val melMean: FloatArray, val melStd: FloatArray)
}

private fun OrtSession.Result.requiredValue(index: Int): OnnxValue =
    get(index) ?: throw IllegalStateException("Missing ONNX output at index $index")

private fun flattenToFloat(value: Any?): FloatArray {
    when (value) {
        is FloatArray -> return value
        is Array<*> -> {
            val result = mutableListOf<Float>()
            for (item in value) {
                val nested = flattenToFloat(item)
                result.addAll(nested.toList())
            }
            return result.toFloatArray()
        }
        is Number -> return floatArrayOf(value.toFloat())
        else -> throw IllegalStateException("Cannot convert ${value?.javaClass} to FloatArray")
    }
}
