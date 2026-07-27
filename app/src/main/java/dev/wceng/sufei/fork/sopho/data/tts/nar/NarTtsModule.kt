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
 * Model files are expected in: /sdcard/SuFei/models/nar/
 * (or app-internal filesDir/models/nar/ for production)
 */
@Module
@InstallIn(SingletonComponent::class)
object NarTtsModule {

    @Provides
    @Singleton
    fun provideNarOnnxEngine(
        @ApplicationContext context: Context,
    ): NarOnnxEngine {
        val modelDir = File(context.filesDir, "models/nar")
        return NarOnnxEngine(modelDir, cpuThreads = 4)
    }
}
