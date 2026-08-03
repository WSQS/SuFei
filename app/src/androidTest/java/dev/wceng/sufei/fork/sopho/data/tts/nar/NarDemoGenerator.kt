package dev.wceng.sufei.fork.sopho.data.tts.nar

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Generates demo audio files on device for user listening.
 * Output: /sdcard/SuFei/nar_demo/
 */
@RunWith(AndroidJUnit4::class)
class NarDemoGenerator {

    private lateinit var context: Context
    private var modelDir: File? = null

    @Before
    fun setup() {
        context = ApplicationProvider.getApplicationContext()
        ChineseG2p.init(context)
        // getExternalFilesDir works for the test process on Android 13
        val extDir = File(context.getExternalFilesDir(null), "models/nar")
        modelDir = if (extDir.resolve("fastspeech2_csmsc.onnx").isFile) extDir else null
        if (modelDir == null) {
            // Fall back to /sdcard path (won't work on Android 13, but try)
            val sdDir = File("/sdcard/SuFei/models/nar")
            if (sdDir.resolve("fastspeech2_csmsc.onnx").isFile) modelDir = sdDir
        }
    }

    @Test
    fun generateDemos() {
        val dir = modelDir ?: run {
            println("SKIP: models not found")
            return
        }

        val outputDir = File(context.getExternalFilesDir(null), "nar_demo")
        outputDir.mkdirs()

        val engine = NarOnnxEngine(dir, cpuThreads = 4)
        engine.use {
            val poems = listOf(
                "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。" to "chunxiao",
                "床前明月光，疑是地上霜。举头望明月，低头思故乡。" to "jingyesi",
                "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒���雪。" to "jiangxue",
                "空山不见人，但闻人语响。返景入深林，复照青苔上。" to "luzhai",
            )

            for ((text, name) in poems) {
                val audio = it.synthesize(text)
                val wavFile = File(outputDir, "$name.wav")
                writePcmToWav(wavFile, audio, 24000)
                println("Generated: ${wavFile.absolutePath} (${audio.size} samples, ${"%.1f".format(audio.size / 24000.0)}s)")
            }
        }
        println("All demos saved to: ${outputDir.absolutePath}")
    }

    private fun writePcmToWav(file: File, pcm: FloatArray, sampleRate: Int) {
        val byteRate = sampleRate * 2 // 16-bit mono
        val dataSize = pcm.size * 2
        val buffer = java.io.ByteArrayOutputStream()
        val out = java.io.DataOutputStream(buffer)

        // RIFF header
        out.writeBytes("RIFF")
        out.writeInt(Integer.reverseBytes(36 + dataSize))
        out.writeBytes("WAVE")

        // fmt chunk
        out.writeBytes("fmt ")
        out.writeInt(Integer.reverseBytes(16))
        out.writeShort(java.lang.Short.reverseBytes(1).toInt()) // PCM
        out.writeShort(java.lang.Short.reverseBytes(1).toInt()) // mono
        out.writeInt(Integer.reverseBytes(sampleRate))
        out.writeInt(Integer.reverseBytes(byteRate))
        out.writeShort(java.lang.Short.reverseBytes(2).toInt()) // block align
        out.writeShort(java.lang.Short.reverseBytes(16).toInt()) // bits per sample

        // data chunk
        out.writeBytes("data")
        out.writeInt(Integer.reverseBytes(dataSize))
        for (sample in pcm) {
            val clamped = sample.coerceIn(-1f, 1f)
            out.writeShort(java.lang.Short.reverseBytes((clamped * Short.MAX_VALUE).toInt().toShort()).toInt())
        }

        file.writeBytes(buffer.toByteArray())
    }
}
