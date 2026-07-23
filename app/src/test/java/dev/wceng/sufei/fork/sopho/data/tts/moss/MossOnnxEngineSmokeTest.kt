package dev.wceng.sufei.fork.sopho.data.tts.moss

import org.junit.Assert
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MossOnnxEngineSmokeTest {

    private val outputDir = File("build/tts-samples").apply { mkdirs() }

    private fun modelRoot(): File? {
        val candidates = listOf(
            File(System.getProperty("user.home"), ".cache/moss-models/MOSS-TTS-Nano-100M-ONNX"),
            File(System.getenv("LOCALAPPDATA") ?: "", "moss-models/MOSS-TTS-Nano-100M-ONNX"),
            File("moss-models/MOSS-TTS-Nano-100M-ONNX"),
            File(System.getProperty("MOSS_MODEL_DIR", System.getenv("MOSS_MODEL_DIR") ?: ""), "MOSS-TTS-Nano-100M-ONNX"),
            File(System.getProperty("MOSS_MODEL_DIR", System.getenv("MOSS_MODEL_DIR") ?: "")),
        )
        return candidates.firstOrNull { dir ->
            dir.isDirectory && File(dir, "browser_poc_manifest.json").isFile
        }
    }

    @Test
    fun synthesizeChinese_nonEmpty() {
        val root = modelRoot()
        assumeTrue("MOSS model not found, skipping", root != null)
        println("MOSS model root: ${root!!.absolutePath}")
        println("Files in root: ${root.listFiles()?.map { it.name }}")
        val engine = try {
            MossOnnxEngine(root, cpuThreads = 2)
        } catch (e: Throwable) {
            println("ENGINE INIT FAILED: ${e::class.java.name}: ${e.message}")
            e.printStackTrace()
            throw e
        }
        try {
            val pcm = engine.synthesize(
                textTokenIds = MossDemoPrompts.CHINESE_TOKEN_IDS,
                voice = "Junhao",
                maxFrames = 160,
            )
            Assert.assertTrue("PCM output should not be empty", pcm.isNotEmpty())
            Assert.assertTrue("PCM output should have meaningful length (got ${pcm.size})", pcm.size > 1000)
            writeWav(pcm, engine.sampleRate, File(outputDir, "chinese_demo.wav"))
        } finally {
            engine.close()
        }
    }

    @Test
    fun synthesizeEnglish_nonEmpty() {
        val root = modelRoot()
        assumeTrue("MOSS model not found, skipping", root != null)
        val engine = MossOnnxEngine(root!!, cpuThreads = 2)
        try {
            val pcm = engine.synthesize(
                textTokenIds = MossDemoPrompts.ENGLISH_TOKEN_IDS,
                voice = "Junhao",
                maxFrames = 160,
            )
            Assert.assertTrue("PCM output should not be empty", pcm.isNotEmpty())
            Assert.assertTrue("PCM output should have meaningful length (got ${pcm.size})", pcm.size > 1000)
            writeWav(pcm, engine.sampleRate, File(outputDir, "english_demo.wav"))
        } finally {
            engine.close()
        }
    }

    private fun writeWav(samples: FloatArray, sampleRate: Int, outputFile: File) {
        val channels = 1
        val dataSize = samples.size * 2
        val fileSize = 44 + dataSize
        val buffer = ByteBuffer.allocate(fileSize).order(ByteOrder.LITTLE_ENDIAN)
        buffer.put("RIFF".toByteArray(Charsets.US_ASCII))
        buffer.putInt(fileSize - 8)
        buffer.put("WAVE".toByteArray(Charsets.US_ASCII))
        buffer.put("fmt ".toByteArray(Charsets.US_ASCII))
        buffer.putInt(16)
        buffer.putShort(1.toShort())
        buffer.putShort(channels.toShort())
        buffer.putInt(sampleRate)
        buffer.putInt(sampleRate * channels * 2)
        buffer.putShort((channels * 2).toShort())
        buffer.putShort(16.toShort())
        buffer.put("data".toByteArray(Charsets.US_ASCII))
        buffer.putInt(dataSize)
        for (sample in samples) {
            buffer.putShort((sample.coerceIn(-1f, 1f) * 32767f).toInt().toShort())
        }
        outputFile.writeBytes(buffer.array())
    }
}
