package dev.wceng.sufei.fork.sopho.data.tts.nar

import android.content.Context
import org.json.JSONObject

/**
 * Chinese text → PaddleSpeech-style phoneme conversion (G2P).
 *
 * Uses bundled pinyin dictionary (assets/pinyin_dict.json).
 */

private val INITIALS = setOf(
    "b", "p", "m", "f", "d", "t", "n", "l",
    "g", "k", "h", "j", "q", "x",
    "zh", "ch", "sh", "r", "z", "c", "s",
)

private val FINAL_FIX = mapOf(
    "un" to "uen", "ui" to "uei", "iu" to "iou",
    "ue" to "ve", "ü" to "v", "üe" to "ve", "n" to "en",
)

private val PUNCT_MAP = mapOf(
    "，" to "，", "。" to "。", "？" to "？", "！" to "！",
    "；" to "，", "：" to "，",
    // "·" intentionally omitted → emits no token, matching the m3_v6 dot-token
    // fix (CosyVoice read 朝代·作者 connected, so '·' must not inject a pause).
)

object ChineseG2p {

    private var pinyinDict: Map<String, String>? = null

    fun init(context: Context) {
        if (pinyinDict != null) return
        val json = context.assets.open("pinyin_dict.json").bufferedReader().use { it.readText() }
        initFromJson(json)
    }

    /**
     * Test seam: load the pinyin dictionary from a raw JSON string, so the G2P
     * can be exercised in JVM unit tests without an Android [Context]. Idempotent.
     */
    internal fun initFromJson(json: String) {
        if (pinyinDict != null) return
        val obj = JSONObject(json)
        pinyinDict = obj.keys().asSequence().associateWith { obj.getString(it) }
    }

    fun textToPhones(text: String): List<String> {
        val phones = mutableListOf<String>()

        for (ch in text) {
            when {
                isChinese(ch) -> {
                    val rawPinyin = pinyinDict?.get(ch.toString())
                    if (!rawPinyin.isNullOrEmpty()) {
                        val py = convertToneMarksToNumbers(rawPinyin.lowercase())
                        processPinyin(py, phones)
                    }
                }
                else -> {
                    val punct = PUNCT_MAP[ch.toString()]
                    if (punct != null) {
                        phones.add(punct)
                    }
                }
            }
        }
        phones.add("<eos>")
        return phones
    }

    private fun isChinese(ch: Char): Boolean = ch.code in 0x4E00..0x9FFF

    /**
     * Convert Unicode tone marks to tone numbers appended at end.
     * e.g. "chūn" → "chun1", "ái" → "ai2", "mǎi" → "mai3", "zài" → "zai4"
     */
    private fun convertToneMarksToNumbers(pinyin: String): String {
        val toneMarks = mapOf(
            'ā' to ('a' to 1), 'ē' to ('e' to 1), 'ī' to ('i' to 1), 'ō' to ('o' to 1), 'ū' to ('u' to 1), 'ǖ' to ('v' to 1),
            'á' to ('a' to 2), 'é' to ('e' to 2), 'í' to ('i' to 2), 'ó' to ('o' to 2), 'ú' to ('u' to 2), 'ǘ' to ('v' to 2),
            'ǎ' to ('a' to 3), 'ě' to ('e' to 3), 'ǐ' to ('i' to 3), 'ǒ' to ('o' to 3), 'ǔ' to ('u' to 3), 'ǚ' to ('v' to 3),
            'à' to ('a' to 4), 'è' to ('e' to 4), 'ì' to ('i' to 4), 'ò' to ('o' to 4), 'ù' to ('u' to 4), 'ǜ' to ('v' to 4),
            'ü' to ('v' to 0),
        )
        val base = StringBuilder()
        var tone = 0
        for (ch in pinyin) {
            val mapped = toneMarks[ch]
            if (mapped != null) {
                base.append(mapped.first)
                if (mapped.second > 0) tone = mapped.second
            } else {
                base.append(ch)
            }
        }
        return base.toString() + (if (tone > 0) tone else 5).toString()
    }

    private fun processPinyin(py: String, phones: MutableList<String>) {
        if (py.isEmpty()) return

        // Extract tone
        val lastChar = py.last()
        val (base, tone) = if (lastChar.isDigit()) {
            py.dropLast(1) to lastChar.toString()
        } else {
            py to "5"  // neutral tone
        }

        // Handle y/w pseudo-initials
        val pyBase = convertPseudoInitial(base)

        // Fix finals
        var fixed = pyBase
        for ((old, new) in FINAL_FIX) {
            if (fixed == old) {
                fixed = new
                break
            }
        }

        // Try to split into initial + final
        var matched = false
        for (initLen in listOf(2, 1)) {
            if (fixed.length > initLen) {
                val candidateInit = fixed.substring(0, initLen)
                var candidateFinal = fixed.substring(initLen)

                if (candidateInit in INITIALS) {
                    // Disambiguate i → i/ii/iii
                    val finalBase = candidateFinal.trimEnd('1', '2', '3', '4', '5', 'r')
                    if (finalBase == "i") {
                        candidateFinal = when {
                            candidateInit in setOf("j", "q", "x") -> candidateFinal
                            candidateInit in setOf("z", "c", "s") -> candidateFinal.replace("i", "ii")
                            candidateInit in setOf("zh", "ch", "sh", "r") -> candidateFinal.replace("i", "iii")
                            else -> candidateFinal
                        }
                    }

                    // Fix final
                    val fb = candidateFinal.trimEnd('1', '2', '3', '4', '5', 'r')
                    if (fb in FINAL_FIX) {
                        val newBase = FINAL_FIX[fb]!!
                        val suffix = candidateFinal.substring(fb.length)
                        candidateFinal = newBase + tone + suffix
                    } else {
                        candidateFinal += tone
                    }

                    phones.add(candidateInit)
                    phones.add(candidateFinal)
                    matched = true
                    break
                }
            }
        }

        if (!matched) {
            // No initial — standalone final
            for ((old, new) in FINAL_FIX) {
                if (fixed == old) {
                    fixed = new
                    break
                }
            }
            phones.add(fixed + tone)
        }
    }

    private fun convertPseudoInitial(base: String): String {
        var result = base
        if (result.startsWith('y')) {
            result = when {
                result.length == 1 -> "i"
                result == "you" -> "iou"
                result == "yue" -> "ve"
                result == "yu" -> "v"
                result == "yuan" -> "van"
                result == "yun" -> "vn"
                result == "yong" -> "iong"
                result[1] in "ae" -> "i" + result.substring(1)
                result[1] == 'i' -> result.substring(1)
                else -> result
            }
        } else if (result.startsWith('w')) {
            result = when {
                result.length == 1 -> "u"
                result == "wu" -> "u"
                else -> "u" + result.substring(1)
            }
        }
        return result
    }
}
