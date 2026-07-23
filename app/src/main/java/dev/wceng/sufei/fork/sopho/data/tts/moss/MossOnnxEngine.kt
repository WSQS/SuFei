package dev.wceng.sufei.fork.sopho.data.tts.moss

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxTensorLike
import ai.onnxruntime.OnnxValue
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONArray
import org.json.JSONObject
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.IntBuffer
import kotlin.math.min

class MossOnnxEngine(
    private val modelRoot: File,
    private val cpuThreads: Int = 2,
) : Closeable {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val manifestPath = resolveManifestPath(modelRoot)
    private val manifestDir = manifestPath.parentFile ?: modelRoot
    private val manifest = ModelManifest.fromJson(readJson(manifestPath))
    private val ttsMetaPath = resolveManifestRelativePath(manifest.modelFiles.ttsMeta)
    private val codecMetaPath = resolveManifestRelativePath(manifest.modelFiles.codecMeta)
    private val ttsMeta = TtsMeta.fromJson(readJson(ttsMetaPath))
    private val codecMeta = CodecMeta.fromJson(readJson(codecMetaPath))
    private val ttsDir = ttsMetaPath.parentFile ?: manifestDir
    private val codecDir = codecMetaPath.parentFile ?: manifestDir
    private val sessionOptions = OrtSession.SessionOptions().apply {
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        setIntraOpNumThreads(cpuThreads.coerceAtLeast(1))
        setInterOpNumThreads(1)
    }
    private val prefillSession = createSession(File(ttsDir, ttsMeta.files.prefill))
    private val decodeSession = createSession(File(ttsDir, ttsMeta.files.decodeStep))
    private val localFixedFrameSession = createSession(File(ttsDir, ttsMeta.files.localFixedSampledFrame))
    private val codecDecodeSession = createSession(File(codecDir, codecMeta.files.decodeFull))

    val sampleRate: Int get() = codecMeta.codecConfig.sampleRate

    fun synthesize(
        textTokenIds: IntArray,
        voice: String = "Junhao",
        maxFrames: Int = 375,
        seed: Long = 1234L,
    ): FloatArray {
        require(textTokenIds.isNotEmpty()) { "textTokenIds must not be empty" }
        val inputRows = buildInputRows(textTokenIds, voice)
        val prefillResult = runPrefill(inputRows)
        val audioTokens = runDecode(prefillResult, maxFrames, seed)
        return decodeAudioTokens(audioTokens)
    }

    override fun close() {
        codecDecodeSession.close()
        localFixedFrameSession.close()
        decodeSession.close()
        prefillSession.close()
        sessionOptions.close()
    }

    private fun createSession(modelFile: File): OrtSession {
        require(modelFile.isFile) { "Missing ONNX file: ${modelFile.absolutePath}" }
        return env.createSession(modelFile.absolutePath, sessionOptions)
    }

    private fun resolveManifestRelativePath(relativePath: String): File {
        val direct = File(manifestDir, relativePath).canonicalFile
        if (direct.exists()) return direct
        val alias = relativePath
            .replace("MOSS-TTS-Nano-ONNX-CPU", "MOSS-TTS-Nano-100M-ONNX")
            .replace("MOSS-Audio-Tokenizer-Nano-ONNX-CPU", "MOSS-Audio-Tokenizer-Nano-ONNX")
        return File(manifestDir, alias).canonicalFile
    }

    private fun buildInputRows(textTokenIds: IntArray, voice: String): InputRows {
        val cfg = manifest.ttsConfig
        val rowWidth = cfg.nVq + 1
        val promptAudioCodes = selectBuiltinVoicePromptAudioCodes(voice)
        val prefixTokens = manifest.promptTemplates.userPromptPrefixTokenIds + cfg.audioStartTokenId
        val suffixTokens = intArrayOf(cfg.audioEndTokenId) +
            manifest.promptTemplates.userPromptAfterReferenceTokenIds +
            textTokenIds +
            manifest.promptTemplates.assistantPromptPrefixTokenIds +
            intArrayOf(cfg.audioStartTokenId)
        val rows = ArrayList<IntArray>()
        rows += buildTextRows(prefixTokens, cfg, rowWidth)
        rows += buildAudioRows(promptAudioCodes, cfg, rowWidth)
        rows += buildTextRows(suffixTokens, cfg, rowWidth)
        return InputRows(rows.toTypedArray(), IntArray(rows.size) { 1 })
    }

    private fun buildTextRows(tokens: IntArray, cfg: TtsConfig, rowWidth: Int): List<IntArray> {
        return tokens.map { token ->
            IntArray(rowWidth) { index -> if (index == 0) token else cfg.audioPadTokenId }
        }
    }

    private fun buildAudioRows(audioCodes: List<IntArray>, cfg: TtsConfig, rowWidth: Int): List<IntArray> {
        return audioCodes.map { codeRow ->
            IntArray(rowWidth) { index ->
                when {
                    index == 0 -> cfg.audioUserSlotTokenId
                    index - 1 < min(codeRow.size, cfg.nVq) -> codeRow[index - 1]
                    else -> cfg.audioPadTokenId
                }
            }
        }
    }

    private fun selectBuiltinVoicePromptAudioCodes(voice: String): List<IntArray> {
        val selected = manifest.builtinVoices.firstOrNull {
            it.voice == voice && it.promptAudioCodes.isNotEmpty()
        } ?: manifest.builtinVoices.firstOrNull { it.promptAudioCodes.isNotEmpty() }
        return selected?.promptAudioCodes
            ?: error("No builtin voice prompt_audio_codes found")
    }

    private fun runPrefill(inputRows: InputRows): PrefillResult {
        val seqLen = inputRows.inputIds.size
        val rowWidth = inputRows.inputIds[0].size
        val inputIdsFlat = IntArray(seqLen * rowWidth)
        var offset = 0
        for (row in inputRows.inputIds) for (value in row) inputIdsFlat[offset++] = value
        OnnxTensor.createTensor(env, IntBuffer.wrap(inputIdsFlat), longArrayOf(1, seqLen.toLong(), rowWidth.toLong())).use { ids ->
            OnnxTensor.createTensor(env, IntBuffer.wrap(inputRows.attentionMask), longArrayOf(1, seqLen.toLong())).use { mask ->
                val outputs = prefillSession.run(mapOf("input_ids" to ids, "attention_mask" to mask))
                return PrefillResult(extractLastHiddenTensor(outputs.requiredTensor("global_hidden")), seqLen, outputs)
            }
        }
    }

    private fun runDecode(prefillResult: PrefillResult, maxFrames: Int, seed: Long): List<IntArray> {
        val cfg = manifest.ttsConfig
        val audioTokens = ArrayList<IntArray>()
        val rowWidth = cfg.nVq + 1
        val cappedMaxFrames = maxFrames.coerceAtMost(manifest.generationDefaults.maxNewFrames)
        val previousTokenSets = Array(cfg.nVq) { HashSet<Int>() }
        val decodePastInputNames = ttsMeta.onnx.decodeInputNames.drop(2)
        val decodePresentOutputNames = ttsMeta.onnx.decodeOutputNames.drop(1)
        val random = java.util.Random(seed)
        var pastValidLengths = prefillResult.pastValidLengths
        var globalHidden = prefillResult.globalHidden
        var pastResult: OrtSession.Result? = prefillResult.pastResult
        try {
            for (step in 0 until cappedMaxFrames) {
                val frameResult = runLocalFixedSampledFrame(globalHidden, previousTokenSets, random)
                if (!frameResult.shouldContinue) break
                val audioRow = IntArray(rowWidth) { index ->
                    if (index == 0) cfg.audioAssistantSlotTokenId else cfg.audioPadTokenId
                }
                for (q in 0 until cfg.nVq) {
                    previousTokenSets[q].add(frameResult.frame[q])
                    audioRow[q + 1] = frameResult.frame[q]
                }
                audioTokens += frameResult.frame
                OnnxTensor.createTensor(env, IntBuffer.wrap(audioRow), longArrayOf(1, 1, rowWidth.toLong())).use { inputTensor ->
                    OnnxTensor.createTensor(env, IntBuffer.wrap(intArrayOf(pastValidLengths)), longArrayOf(1)).use { pastTensor ->
                        val feeds = linkedMapOf<String, OnnxTensorLike>("input_ids" to inputTensor, "past_valid_lengths" to pastTensor)
                        val prev = pastResult ?: error("Missing KV cache")
                        for (i in decodePastInputNames.indices) feeds[decodePastInputNames[i]] = prev.requiredTensor(decodePresentOutputNames[i])
                        val outputs = decodeSession.run(feeds)
                        val nextHidden = extractLastHiddenTensor(outputs.requiredTensor("global_hidden"))
                        globalHidden.close(); prev.close()
                        pastResult = outputs; globalHidden = nextHidden
                        pastValidLengths += 1
                    }
                }
            }
        } finally {
            globalHidden.close(); pastResult?.close()
        }
        return audioTokens
    }

    private fun runLocalFixedSampledFrame(globalHidden: OnnxTensor, previousTokenSets: Array<HashSet<Int>>, random: java.util.Random): LocalFrameResult {
        val cfg = manifest.ttsConfig
        val codebookSize = cfg.audioCodebookSizes.firstOrNull() ?: 1024
        val seenMask = IntArray(cfg.nVq * codebookSize)
        for (ch in previousTokenSets.indices) {
            val off = ch * codebookSize
            for (tid in previousTokenSets[ch]) if (tid in 0 until codebookSize) seenMask[off + tid] = 1
        }
        val asstRand = floatArrayOf(random.nextDouble().coerceIn(1e-6, 1.0 - 1e-6).toFloat())
        val audioRand = FloatArray(cfg.nVq) { random.nextDouble().coerceIn(1e-6, 1.0 - 1e-6).toFloat() }
        OnnxTensor.createTensor(env, IntBuffer.wrap(seenMask), longArrayOf(1, cfg.nVq.toLong(), codebookSize.toLong())).use { seen ->
            OnnxTensor.createTensor(env, FloatBuffer.wrap(asstRand), longArrayOf(1)).use { ar ->
                OnnxTensor.createTensor(env, FloatBuffer.wrap(audioRand), longArrayOf(1, cfg.nVq.toLong())).use { audR ->
                    val outputs = localFixedFrameSession.run(mapOf("global_hidden" to globalHidden, "repetition_seen_mask" to seen, "assistant_random_u" to ar, "audio_random_u" to audR))
                    outputs.use { return LocalFrameResult(it.requiredTensor("should_continue").scalarInt() > 0, it.requiredTensor("frame_token_ids").intArrayValue()) }
                }
            }
        }
    }

    private fun decodeAudioTokens(audioTokens: List<IntArray>): FloatArray {
        require(audioTokens.isNotEmpty()) { "No audio tokens generated" }
        val numFrames = audioTokens.size
        val numQuantizers = manifest.ttsConfig.nVq
        val flat = IntArray(numFrames * numQuantizers)
        var off = 0
        for (frame in audioTokens) for (q in 0 until numQuantizers) flat[off++] = frame[q]
        OnnxTensor.createTensor(env, IntBuffer.wrap(flat), longArrayOf(1, numFrames.toLong(), numQuantizers.toLong())).use { codes ->
            OnnxTensor.createTensor(env, IntBuffer.wrap(intArrayOf(numFrames)), longArrayOf(1)).use { lengths ->
                val outputs = codecDecodeSession.run(mapOf("audio_codes" to codes, "audio_code_lengths" to lengths))
                outputs.use {
                    val audio = it.requiredTensor("audio").value as Array<*>
                    val batch = audio[0] as Array<*>
                    val channels = batch.map { ch -> ch as FloatArray }
                    val repLen = it.requiredTensor("audio_lengths").scalarInt()
                    val len = min(repLen, channels.minOfOrNull { ch -> ch.size } ?: 0)
                    return FloatArray(len) { si -> channels.sumOf { ch -> ch[si].toDouble() }.toFloat() / channels.size }
                }
            }
        }
    }

    companion object {
        private fun resolveManifestPath(modelRoot: File): File {
            val candidates = listOf(
                File(modelRoot, "browser_poc_manifest.json"),
                File(modelRoot, "MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json"),
            )
            return candidates.firstOrNull { it.isFile } ?: error("browser_poc_manifest.json not found in $modelRoot")
        }
        private fun readJson(file: File): JSONObject {
            require(file.isFile) { "Missing JSON: ${file.absolutePath}" }
            return JSONObject(file.readText(Charsets.UTF_8))
        }
        private fun flattenIntTensorValue(raw: Any?): IntArray {
            val values = ArrayList<Int>()
            fun append(v: Any?) {
                when (v) {
                    is Int -> values += v; is Long -> values += v.toInt()
                    is Short -> values += v.toInt(); is Byte -> values += v.toInt()
                    is IntArray -> values += v.toList(); is LongArray -> v.forEach { values += it.toInt() }
                    is ShortArray -> v.forEach { values += it.toInt() }; is ByteArray -> v.forEach { values += it.toInt() }
                    is Array<*> -> v.forEach { append(it) }
                    null -> Unit
                    else -> error("Unsupported tensor type: ${v.javaClass}")
                }
            }
            append(raw); return values.toIntArray()
        }
        private fun extractLastHiddenTensor(tensor: OnnxTensor): OnnxTensor {
            val hidden = when (tensor.info.shape.size) {
                2 -> { val v = tensor.value as Array<*>; v[0] as FloatArray }
                3 -> { val v = tensor.value as Array<*>; (v[0] as Array<*>).last() as FloatArray }
                else -> error("Unexpected rank: ${tensor.info.shape.size}")
            }
            return OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), FloatBuffer.wrap(hidden.copyOf()), longArrayOf(1, hidden.size.toLong()))
        }
        private fun OrtSession.Result.requiredValue(name: String): OnnxValue = get(name).orElseThrow { IllegalStateException("Missing ONNX output: $name") }
        private fun OrtSession.Result.requiredTensor(name: String): OnnxTensor = requiredValue(name) as OnnxTensor
        private fun OnnxTensor.scalarInt(): Int = flattenIntTensorValue(value).firstOrNull() ?: error("Scalar tensor empty")
        private fun OnnxTensor.intArrayValue(): IntArray = flattenIntTensorValue(value)
    }

    private data class InputRows(val inputIds: Array<IntArray>, val attentionMask: IntArray)
    private data class PrefillResult(val globalHidden: OnnxTensor, val pastValidLengths: Int, val pastResult: OrtSession.Result)
    private data class LocalFrameResult(val shouldContinue: Boolean, val frame: IntArray)
}

private data class ModelManifest(
    val modelFiles: ModelFiles, val ttsConfig: TtsConfig, val promptTemplates: PromptTemplates,
    val generationDefaults: GenerationDefaults, val builtinVoices: List<BuiltinVoice>,
) {
    companion object {
        fun fromJson(json: JSONObject) = ModelManifest(
            modelFiles = ModelFiles.fromJson(json.getJSONObject("model_files")),
            ttsConfig = TtsConfig.fromJson(json.getJSONObject("tts_config")),
            promptTemplates = PromptTemplates.fromJson(json.getJSONObject("prompt_templates")),
            generationDefaults = GenerationDefaults.fromJson(json.optJSONObject("generation_defaults")),
            builtinVoices = json.optJSONArray("builtin_voices")?.let { voices ->
                List(voices.length()) { BuiltinVoice.fromJson(voices.getJSONObject(it)) }
            } ?: emptyList(),
        )
    }
}
private data class ModelFiles(val ttsMeta: String, val codecMeta: String) {
    companion object { fun fromJson(json: JSONObject) = ModelFiles(json.getString("tts_meta"), json.getString("codec_meta")) }
}
private data class TtsConfig(
    val nVq: Int, val audioPadTokenId: Int, val audioStartTokenId: Int, val audioEndTokenId: Int,
    val audioUserSlotTokenId: Int, val audioAssistantSlotTokenId: Int, val audioCodebookSizes: IntArray,
) {
    companion object {
        fun fromJson(json: JSONObject) = TtsConfig(
            nVq = json.getInt("n_vq"), audioPadTokenId = json.getInt("audio_pad_token_id"),
            audioStartTokenId = json.getInt("audio_start_token_id"), audioEndTokenId = json.getInt("audio_end_token_id"),
            audioUserSlotTokenId = json.optInt("audio_user_slot_token_id", 8),
            audioAssistantSlotTokenId = json.getInt("audio_assistant_slot_token_id"),
            audioCodebookSizes = json.getJSONArray("audio_codebook_sizes").let { arr -> IntArray(arr.length()) { arr.getInt(it) } },
        )
    }
}
private data class PromptTemplates(val userPromptPrefixTokenIds: IntArray, val userPromptAfterReferenceTokenIds: IntArray, val assistantPromptPrefixTokenIds: IntArray) {
    companion object {
        fun fromJson(json: JSONObject) = PromptTemplates(
            userPromptPrefixTokenIds = json.getJSONArray("user_prompt_prefix_token_ids").toIA(),
            userPromptAfterReferenceTokenIds = json.getJSONArray("user_prompt_after_reference_token_ids").toIA(),
            assistantPromptPrefixTokenIds = json.getJSONArray("assistant_prompt_prefix_token_ids").toIA(),
        )
        private fun JSONArray.toIA() = IntArray(length()) { getInt(it) }
    }
}
private data class GenerationDefaults(val maxNewFrames: Int = 375) {
    companion object { fun fromJson(json: JSONObject?) = GenerationDefaults(json?.optInt("max_new_frames", 375) ?: 375) }
}
private data class BuiltinVoice(val voice: String, val promptAudioCodes: List<IntArray>) {
    companion object {
        fun fromJson(json: JSONObject) = BuiltinVoice(
            voice = json.optString("voice", ""),
            promptAudioCodes = json.optJSONArray("prompt_audio_codes")?.let { outer ->
                List(outer.length()) { IntArray(outer.getJSONArray(it).length()) { i -> outer.getJSONArray(it).getInt(i) } }
            } ?: emptyList(),
        )
    }
}
private data class TtsMeta(val files: TtsFiles, val onnx: TtsOnnxNames) {
    companion object {
        fun fromJson(json: JSONObject) = TtsMeta(
            files = TtsFiles.fromJson(json.getJSONObject("files")),
            onnx = TtsOnnxNames.fromJson(json.getJSONObject("onnx")),
        )
    }
}
private data class TtsFiles(val prefill: String, val decodeStep: String, val localFixedSampledFrame: String) {
    companion object {
        fun fromJson(json: JSONObject) = TtsFiles(json.getString("prefill"), json.getString("decode_step"), json.getString("local_fixed_sampled_frame"))
    }
}
private data class TtsOnnxNames(val decodeInputNames: List<String>, val decodeOutputNames: List<String>) {
    companion object {
        fun fromJson(json: JSONObject) = TtsOnnxNames(
            decodeInputNames = json.getJSONArray("decode_input_names").let { a -> List(a.length()) { a.getString(it) } },
            decodeOutputNames = json.getJSONArray("decode_output_names").let { a -> List(a.length()) { a.getString(it) } },
        )
    }
}
private data class CodecMeta(val files: CodecFiles, val codecConfig: CodecConfig) {
    companion object {
        fun fromJson(json: JSONObject) = CodecMeta(CodecFiles.fromJson(json.getJSONObject("files")), CodecConfig.fromJson(json.getJSONObject("codec_config")))
    }
}
private data class CodecFiles(val decodeFull: String) {
    companion object { fun fromJson(json: JSONObject) = CodecFiles(json.getString("decode_full")) }
}
private data class CodecConfig(val sampleRate: Int) {
    companion object { fun fromJson(json: JSONObject) = CodecConfig(json.getInt("sample_rate")) }
}