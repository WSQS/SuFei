package dev.wceng.sufei.fork.sopho.data.tts.nar

import android.content.Context
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import dev.wceng.sufei.fork.sopho.data.tts.nar.NarOnnxEngine
import java.io.File
import javax.inject.Singleton

/**
 * Hilt module providing the NAR TTS engine.
 *
 * fork-sopho: Prefers the custom SuFei FS2 (35MB) if available,
 * falls back to PaddleSpeech FS2 (142MB).
 *
 * Model files expected in: externalFilesDir/models/nar/ or /sdcard/SuFei/models/nar/
 *
 * SuFei custom model files:
 *   - fastspeech2_sufei.onnx (~35MB)
 *   - hifigan_csmsc.onnx (~50MB, shared)
 *   - phone_id_map.txt (3KB, shared)
 *   - norm_stats.npz (1KB, mel denormalization)
 *
 * PaddleSpeech fallback files:
 *   - fastspeech2_csmsc.onnx (142MB)
 *   - hifigan_csmsc.onnx (50MB)
 *   - phone_id_map.txt (3KB)
 */
@Module
@InstallIn(SingletonComponent::class)
object NarTtsModule {

    @Provides
    @Singleton
    fun provideNarOnnxEngine(
        @ApplicationContext context: Context,
    ): NarOnnxEngine {
        ChineseG2p.init(context)

        val candidates = listOf(
            java.io.File(context.getExternalFilesDir(null), "models/nar"),
            java.io.File(android.os.Environment.getExternalStorageDirectory(), "SuFei/models/nar"),
            java.io.File(context.filesDir, "models/nar"),
        )

        // fork-sopho: prefer SuFei custom model
        val sufeiDir = candidates.firstOrNull {
            it.resolve("fastspeech2_sufei.onnx").isFile
        }
        if (sufeiDir != null) {
            return NarOnnxEngine(sufeiDir, cpuThreads = 4, useSuFeiModel = true)
        }

        // Fallback to PaddleSpeech
        val modelDir = candidates.firstOrNull {
            it.resolve("fastspeech2_csmsc.onnx").isFile
        } ?: context.filesDir.resolve("models/nar")
        return NarOnnxEngine(modelDir, cpuThreads = 4)
    }
}
