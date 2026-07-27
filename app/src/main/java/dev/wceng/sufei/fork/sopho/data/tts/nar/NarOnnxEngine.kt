package dev.wceng.sufei.fork.sopho.data.tts.nar

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxValue
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer
import java.nio.IntBuffer

/**
 * FastSpeech 2 + HiFi-GAN NAR TTS engine.
 *
 * Pipeline: text → G2P → phone IDs → FS2 ONNX → mel → HiFi-GAN → waveform
 *
 * @param modelDir directory containing fastspeech2_csmsc.onnx, hifigan_csmsc.onnx, phone_id_map.txt
 * @param cpuThreads inference thread count
 */
class NarOnnxEngine(
    private val modelDir: File,
    private val cpuThreads: Int = 4,
) : Closeable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val sessionOptions = OrtSession.SessionOptions().apply {
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        setIntraOpNumThreads(cpuThreads.coerceAtLeast(1))
        setInterOpNumThreads(1)
    }

    private val fs2Session: OrtSession = createSession("fastspeech2_csmsc.onnx")
    private val hifiganSession: OrtSession = createSession("hifigan_csmsc.onnx")
    private val phoneIdMap: Map<String, Int> = loadPhoneIdMap()

    val sampleRate: Int = 24000

    /**
     * Synthesize speech from Chinese text.
     *
     * @param text Chinese text (may contain punctuation: ，。？！；：)
     * @return PCM float array at 24kHz mono
     */
    fun synthesize(text: String): FloatArray {
        val phoneIds = g2p(text)
        require(phoneIds.isNotEmpty()) { "No valid phonemes from text: $text" }
        val mel = runFastSpeech2(phoneIds)
        return runHifiGan(mel)
    }

    private fun g2p(text: String): IntArray {
        val phones = ChineseG2p.textToPhones(text)
        return phones.mapNotNull { phoneIdMap[it] }.toIntArray()
    }

    private fun runFastSpeech2(phoneIds: IntArray): Array<FloatArray> {
        val input = OnnxTensor.createTensor(
            env,
            IntBuffer.wrap(phoneIds),
            longArrayOf(phoneIds.size.toLong()),
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
                @Suppress("UNCHECKED_CAST")
                val batch = result.requiredValue(0).value as Array<*>
                val channel = batch[0] as FloatArray
                return channel
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
}

private fun OrtSession.Result.requiredValue(index: Int): OnnxValue =
    get(index) ?: throw IllegalStateException("Missing ONNX output at index $index")
