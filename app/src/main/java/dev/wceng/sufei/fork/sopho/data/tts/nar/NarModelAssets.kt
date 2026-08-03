package dev.wceng.sufei.fork.sopho.data.tts.nar

import android.content.Context
import android.util.Log
import java.io.File
import java.io.IOException

/**
 * First-launch extraction of the bundled NAR TTS model.
 *
 * The 50 MB fp16 model (m3_v6) is embedded into the APK under
 * `assets/models/nar/` at CI/CD build time (see the `fetchTtsModels` Gradle
 * task) and is *not* tracked in git. Since [NarOnnxEngine] loads models by
 * filesystem path via `OrtEnvironment.createSession(path)`, the assets must be
 * materialized to a real directory before first use. This copies them once to
 * `filesDir/models/nar/` — the lowest-priority candidate dir in [NarTtsModule],
 * so a dev-pushed model on external/sdcard storage still overrides it.
 *
 * Re-extraction is triggered whenever the shipped `SHA256SUMS.txt` changes
 * (i.e. an app update ships a new model), detected via a `.asset_version`
 * marker file written alongside the extracted models.
 */
object NarModelAssets {

    private const val TAG = "NarModelAssets"
    private const val ASSET_DIR = "models/nar"
    private const val TARGET_SUBPATH = "models/nar"
    private const val MARKER_NAME = ".asset_version"

    /** Files [NarOnnxEngine] requires to load the SuFei custom model. */
    private val REQUIRED = listOf(
        "fastspeech2_sufei.onnx",
        "hifigan_csmsc.onnx",
        "phone_id_map.txt",
        "norm_stats.npz",
    )

    /**
     * Ensure the bundled model is extracted to `filesDir/models/nar/`.
     *
     * @return the extracted directory if the model is available, or `null` when
     *   this build embeds no model and none is present on disk (callers then
     *   fall back to the PaddleSpeech model or disable NAR TTS).
     */
    fun ensureExtracted(context: Context): File? {
        val target = File(context.filesDir, TARGET_SUBPATH)

        val bundled = try {
            context.assets.list(ASSET_DIR)?.toList().orEmpty()
        } catch (e: IOException) {
            Log.w(TAG, "Failed to list assets/$ASSET_DIR", e)
            emptyList()
        }

        // No embedded model in this build → use whatever is already on disk.
        if (bundled.isEmpty()) {
            return if (isComplete(target)) target else null
        }

        val assetVersion = readAssetText("$ASSET_DIR/SHA256SUMS.txt", context)
            ?: bundled.sorted().joinToString(",")  // fallback marker if no checksum file
        val marker = File(target, MARKER_NAME)
        val currentVersion = marker.takeIf { it.isFile }?.readText()

        if (isComplete(target) && currentVersion == assetVersion) {
            return target
        }

        Log.i(TAG, "Extracting NAR model to ${target.absolutePath} (version changed: ${currentVersion != assetVersion})")
        return try {
            extract(context, bundled, target)
            marker.writeText(assetVersion)
            if (isComplete(target)) target else {
                Log.e(TAG, "Extraction incomplete — missing required files in ${target.absolutePath}")
                null
            }
        } catch (e: IOException) {
            Log.e(TAG, "Failed to extract NAR model", e)
            null
        }
    }

    private fun extract(context: Context, names: List<String>, target: File) {
        target.mkdirs()
        for (name in names) {
            val dst = File(target, name)
            val tmp = File(target, "$name.part")
            context.assets.open("$ASSET_DIR/$name").use { input ->
                tmp.outputStream().use { output -> input.copyTo(output, bufferSize = 1 shl 16) }
            }
            // Atomic-ish swap so a crash mid-copy never leaves a truncated final file.
            if (dst.exists()) dst.delete()
            if (!tmp.renameTo(dst)) {
                tmp.copyTo(dst, overwrite = true)
                tmp.delete()
            }
        }
    }

    private fun isComplete(dir: File): Boolean =
        REQUIRED.all { File(dir, it).let { f -> f.isFile && f.length() > 0 } }

    private fun readAssetText(path: String, context: Context): String? = try {
        context.assets.open(path).bufferedReader().use { it.readText() }
    } catch (e: IOException) {
        null
    }
}
