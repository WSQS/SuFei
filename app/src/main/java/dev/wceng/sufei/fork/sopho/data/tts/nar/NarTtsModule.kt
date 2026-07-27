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
 * Model files expected in: /sdcard/SuFei/models/nar/
 * For production: consider split APK or download-on-first-launch.
 *
 * Required files:
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
        val candidates = listOf(
            java.io.File(context.getExternalFilesDir(null), "models/nar"),
            java.io.File(android.os.Environment.getExternalStorageDirectory(), "SuFei/models/nar"),
            java.io.File(context.filesDir, "models/nar"),
        )
        val modelDir = candidates.firstOrNull {
            it.resolve("fastspeech2_csmsc.onnx").isFile
        } ?: context.filesDir.resolve("models/nar")
        return NarOnnxEngine(modelDir, cpuThreads = 4)
    }
}
