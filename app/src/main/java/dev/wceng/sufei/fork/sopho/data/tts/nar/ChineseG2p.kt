package dev.wceng.sufei.fork.sopho.data.tts.nar

import com.github.promeg.pinyinhelper.Pinyin

/**
 * Chinese text → PaddleSpeech-style phoneme conversion (G2P).
 *
 * Converts Chinese characters to pinyin with tone numbers,
 * then splits into initials/finals matching PaddleSpeech's phone vocabulary.
 *
 * Example: "春眠不觉晓" → [ch, un1, m, ian2, b, u4, j, ue2, x, iao3]
 *
 * Depends on TinyPinyin for character→pinyin lookup.
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
    "；" to "，", "：" to "，", "·" to "。",
)

object ChineseG2p {

    /**
     * Convert Chinese text to PaddleSpeech phone sequence.
     *
     * @return list of phone strings (initials, finals with tone, punctuation, <eos>)
     */
    fun textToPhones(text: String): List<String> {
        val phones = mutableListOf<String>()

        for (ch in text) {
            when {
                isChinese(ch) -> {
                    val py = Pinyin.toPinyin(ch).lowercase()
                    if (py.isNotEmpty()) {
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
