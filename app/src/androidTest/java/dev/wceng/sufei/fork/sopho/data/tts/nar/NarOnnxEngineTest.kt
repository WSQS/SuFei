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
 * Instrumented test for NarOnnxEngine.
 *
 * Prerequisites: ONNX models at /sdcard/SuFei/models/nar/
 * Run: ./gradlew connectedAndroidTest --tests "*.NarOnnxEngineTest"
 */
@RunWith(AndroidJUnit4::class)
class NarOnnxEngineTest {

    private lateinit var context: Context
    private var modelDir: File? = null

    @Before
    fun setup() {
        context = ApplicationProvider.getApplicationContext()
        val candidates = listOf(
            File(context.getExternalFilesDir(null), "models/nar"),
            File(android.os.Environment.getExternalStorageDirectory(), "SuFei/models/nar"),
            File(context.filesDir, "models/nar"),
        )
        modelDir = candidates.firstOrNull { it.resolve("fastspeech2_csmsc.onnx").isFile }
    }

    @Test
    fun testG2pProducesValidPhones() {
        val phones = ChineseG2p.textToPhones("春眠不觉晓")
        assertTrue("G2P should produce phones", phones.isNotEmpty())
        assertTrue("Should end with <eos>", phones.last() == "<eos>")
        // 春 → ch + uen1 (at least 2 phones for one char)
        assertTrue("Should produce at least 10 phones for 5 chars",
            phones.size >= 10)
    }

    @Test
    fun testG2pHandlesPunctuation() {
        val phones = ChineseG2p.textToPhones("春眠不觉晓，处处闻啼鸟。")
        assertTrue("Should contain punctuation phones",
            phones.contains("，") || phones.contains("。"))
    }

    @Test
    fun testSynthesizeIfModelsAvailable() {
        val dir = modelDir ?: run {
            println("SKIP: models not found. Push to /sdcard/SuFei/models/nar/")
            return
        }

        val engine = NarOnnxEngine(dir, cpuThreads = 4)
        engine.use {
            val audio = it.synthesize("春眠不觉晓")
            println("Audio samples: ${audio.size}")
            assertTrue("Audio should not be empty", audio.isNotEmpty())
            assertTrue("Audio should have reasonable length (>1000 samples), got ${audio.size}",
                audio.size > 1000)
        }
    }

    @Test
    fun testSampleRate() {
        val dir = modelDir ?: return
        val engine = NarOnnxEngine(dir)
        assertTrue("Sample rate should be 24000", engine.sampleRate == 24000)
        engine.close()
    }
}
