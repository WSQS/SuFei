package dev.wceng.sufei.fork.sopho.data.tts.nar

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * On-device E2E test for the deployed m3_v6 NAR TTS pipeline.
 *
 * Exercises the real shipping path end-to-end: [NarModelAssets.ensureExtracted]
 * materializes the CI-embedded model from `assets/models/nar/` into filesDir,
 * then [NarOnnxEngine] with `useSuFeiModel = true` runs text → G2P → FastSpeech2 →
 * (mel denorm) → HiFi-GAN → waveform. No manual model push required.
 *
 * Run: ./gradlew connectedDebugAndroidTest --tests "*.NarOnnxEngineTest"
 */
@RunWith(AndroidJUnit4::class)
class NarOnnxEngineTest {

    private lateinit var context: Context
    private var modelDir: File? = null

    @Before
    fun setup() {
        context = ApplicationProvider.getApplicationContext()
        ChineseG2p.init(context)
        // The actual deployed extraction path (assets → filesDir). Null only if this
        // build embeds no model, in which case the model-dependent tests skip.
        modelDir = NarModelAssets.ensureExtracted(context)
    }

    @Test
    fun testG2pProducesValidPhones() {
        val phones = ChineseG2p.textToPhones("春眠不觉晓")
        assertTrue("G2P should produce phones", phones.isNotEmpty())
        assertTrue("Should end with <eos>", phones.last() == "<eos>")
        assertTrue("Should produce at least 10 phones for 5 chars", phones.size >= 10)
    }

    @Test
    fun testG2pHandlesPunctuation() {
        val phones = ChineseG2p.textToPhones("春眠不觉晓，处处闻啼鸟。")
        assertTrue("Should contain punctuation phones",
            phones.contains("，") || phones.contains("。"))
    }

    @Test
    fun testEmbeddedModelExtracts() {
        val dir = modelDir ?: run {
            println("SKIP: no embedded model in this build")
            return
        }
        for (f in listOf("fastspeech2_sufei.onnx", "hifigan_csmsc.onnx", "phone_id_map.txt", "norm_stats.npz")) {
            assertTrue("extracted $f must exist", File(dir, f).let { it.isFile && it.length() > 0 })
        }
        val fs2 = File(dir, "fastspeech2_sufei.onnx")
        val sha = java.security.MessageDigest.getInstance("SHA-256").digest(fs2.readBytes())
            .joinToString("") { "%02x".format(it) }
        println("on-device fastspeech2_sufei.onnx sha256=$sha size=${fs2.length()}")
        android.util.Log.d("NarOnnxEngine", "extracted fastspeech2_sufei.onnx sha256=$sha")
    }

    @Test
    fun testSynthesizeDeployedModelE2E() {
        val dir = modelDir ?: run {
            println("SKIP: no embedded model in this build")
            return
        }
        // Deployed narText shape: "{title}，{dynasty}{author}。{content}".
        val text = "登鹳雀楼，唐王之涣。白日依山尽，黄河入海流。欲穷千里目，更上一层楼。"
        NarOnnxEngine(dir, cpuThreads = 4, useSuFeiModel = true).use { engine ->
            assertTrue("Sample rate should be 24000", engine.sampleRate == 24000)
            val audio = engine.synthesize(text)
            println("E2E synth: ${audio.size} samples (${"%.2f".format(audio.size / 24000.0)}s) for ${text.length} chars")
            assertNotNull(audio)
            // A 24-char poem must yield well over a second of speech; a few-hundred-sample
            // result would signal a broken length regulator or truncated pipeline.
            assertTrue("Expected > 1s of audio, got ${audio.size} samples", audio.size > 24000)
            // Guard against NaN/inf leaking from the vocoder and clipping.
            val peak = audio.maxOf { kotlin.math.abs(it) }
            assertTrue("Audio peak must be finite and non-silent, got $peak", peak.isFinite() && peak > 1e-3f)
        }
    }
}
