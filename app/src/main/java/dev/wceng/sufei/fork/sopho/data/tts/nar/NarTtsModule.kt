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
 * Model files are loaded from app-accessible locations only: the CI-embedded
 * model extracted to filesDir/models/nar/, or externalFilesDir/models/nar/ where
 * a developer may `adb push` an override. A public path like /sdcard/SuFei is
 * intentionally not used — onnxruntime opens the file directly and scoped storage
 * denies that (EACCES / errno 13).
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

        // fork-sopho: materialize the CI-embedded model (assets/models/nar) to
        // filesDir on first launch, so the filesystem-path loader below finds it.
        NarModelAssets.ensureExtracted(context)

        // App-accessible dirs only, dev override first then the embedded model.
        // Public external storage (/sdcard/SuFei) is excluded: onnxruntime's native
        // loader opens the path directly and scoped storage denies it (EACCES).
        val candidates = listOf(
            File(context.getExternalFilesDir(null), "models/nar"),
            File(context.filesDir, "models/nar"),
        )

        // Try each candidate that has the model file, using the first that actually
        // loads — a present-but-unreadable file must not abort the whole engine.
        fun tryLoad(fileName: String, useSuFei: Boolean): NarOnnxEngine? {
            for (dir in candidates.filter { it.resolve(fileName).isFile }) {
                try {
                    return NarOnnxEngine(dir, cpuThreads = 4, useSuFeiModel = useSuFei)
                } catch (e: Exception) {
                    android.util.Log.w("NarTtsModule", "Model load failed at ${dir.absolutePath}", e)
                }
            }
            return null
        }

        // Prefer the custom SuFei FS2 (m3_v6); fall back to PaddleSpeech FS2.
        tryLoad("fastspeech2_sufei.onnx", useSuFei = true)?.let { return it }
        tryLoad("fastspeech2_csmsc.onnx", useSuFei = false)?.let { return it }

        // Nothing loaded — construct against the embedded dir so NarOnnxEngine
        // surfaces a clear error when synthesis is attempted.
        return NarOnnxEngine(File(context.filesDir, "models/nar"), cpuThreads = 4, useSuFeiModel = true)
    }
}
