package dev.wceng.sufei.fork.sopho.data.tts.nar

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import java.io.File

/**
 * G2P parity test — proves the on-device Kotlin [ChineseG2p] reproduces the exact
 * phoneme id sequences the m3_v6 model was trained on.
 *
 * Ground truth: `nar/g2p_golden.jsonl` (a test resource) joins each poem's training
 * text with the `phoneme_ids` from the shipped cosyvoice3 training manifests
 * (huggingface.co/datasets/Sopho/sopho-poetry-tts-data). Every golden poem contains
 * a `·`, so this also verifies the m3_v6 dot-token fix (`·` must emit no token).
 *
 * The test runs the real production G2P — text → [ChineseG2p.textToPhones] → map via
 * the real `phone_id_map.txt` — and compares the resulting ids to the golden ids.
 */
class ChineseG2pTest {

    companion object {
        private lateinit var phoneToId: Map<String, Int>
        private lateinit var idToPhone: Map<Int, String>
        private lateinit var golden: List<Golden>

        private data class Golden(val poemId: String, val split: String, val text: String, val ids: IntArray)

        @BeforeClass
        @JvmStatic
        fun setUp() {
            ChineseG2p.initFromJson(readMainAsset("pinyin_dict.json"))

            phoneToId = readResource("/nar/phone_id_map.txt").lineSequence()
                .mapNotNull { line ->
                    val parts = line.trim().split(" ")
                    if (parts.size == 2) parts[0] to parts[1].toInt() else null
                }.toMap()
            idToPhone = phoneToId.entries.associate { (k, v) -> v to k }

            golden = readResource("/nar/g2p_golden.jsonl").lineSequence()
                .filter { it.isNotBlank() }
                .map { line ->
                    val o = JSONObject(line)
                    val arr = o.getJSONArray("phoneme_ids")
                    Golden(o.getString("poem_id"), o.getString("split"), o.getString("text"), arr.toIntArray())
                }.toList()
        }

        private fun JSONArray.toIntArray() = IntArray(length()) { getInt(it) }

        private fun readResource(path: String): String =
            ChineseG2pTest::class.java.getResourceAsStream(path)?.bufferedReader()?.use { it.readText() }
                ?: error("missing test resource: $path")

        /** pinyin_dict.json lives in main/assets (tracked). Locate it relative to the module dir. */
        private fun readMainAsset(name: String): String {
            val candidates = listOf(
                File("src/main/assets/$name"),
                File("app/src/main/assets/$name"),
                File(System.getProperty("user.dir"), "src/main/assets/$name"),
                File(System.getProperty("user.dir"), "app/src/main/assets/$name"),
            )
            val f = candidates.firstOrNull { it.isFile }
                ?: error("cannot find main asset $name; tried ${candidates.map { it.absolutePath }}")
            return f.readText()
        }
    }

    /** Convert a poem's training text to phone ids exactly as the engine does at runtime. */
    private fun idsFor(text: String): IntArray =
        ChineseG2p.textToPhones(text).mapNotNull { phoneToId[it] }.toIntArray()

    @Test
    fun goldenIsLoaded() {
        assertTrue("golden must be non-empty", golden.isNotEmpty())
        assertEquals("<eos> id", 267, phoneToId["<eos>"])
        // Every golden poem contains '·' — this suite is a dot-token stress test.
        assertTrue(golden.all { it.text.contains('·') })
    }

    @Test
    fun dotProducesNoToken() {
        // '·' must not exist in the phone table and must be dropped by G2P.
        assertTrue("'·' must not be a phone", !phoneToId.containsKey("·"))
        val withDot = ChineseG2p.textToPhones("宋代·苏轼")
        val without = ChineseG2p.textToPhones("宋代苏轼")
        assertEquals("· must emit no token (glued == dotted)", without, withDot)
    }

    @Test
    fun phonemeParityAgainstTrainingManifests() {
        data class Acc(var poems: Int = 0, var exact: Int = 0, var pos: Long = 0, var hit: Long = 0)
        val bySplit = linkedMapOf("train" to Acc(), "holdout" to Acc(), "external_unseen" to Acc())
        val diverging = mutableListOf<String>()

        for (g in golden) {
            val got = idsFor(g.text)
            val a = bySplit.getValue(g.split)
            a.poems++
            val exact = got.contentEquals(g.ids)
            if (exact) a.exact++
            val n = minOf(got.size, g.ids.size)
            a.pos += g.ids.size
            for (i in 0 until n) if (got[i] == g.ids[i]) a.hit++
            if (!exact && diverging.size < 20) {
                val at = (0 until n).firstOrNull { got[it] != g.ids[it] } ?: n
                val exp = g.ids.getOrNull(at)?.let { idToPhone[it] } ?: "∅"
                val act = got.getOrNull(at)?.let { idToPhone[it] } ?: "∅"
                diverging.add("  ${g.poemId}[${g.split}] len ${got.size}/${g.ids.size} @$at: exp=$exp got=$act  «${g.text.take(24)}…»")
            }
        }

        println("=== G2P parity vs m3_v6 training manifests (${golden.size} poems, all contain '·') ===")
        for ((split, a) in bySplit) {
            val seq = if (a.poems == 0) 0.0 else a.exact * 100.0 / a.poems
            val ph = if (a.pos == 0L) 0.0 else a.hit * 100.0 / a.pos
            println("%-16s exact-seq %4d/%-4d (%.1f%%)  per-phoneme %.2f%%".format(split, a.exact, a.poems, seq, ph))
        }
        if (diverging.isNotEmpty()) {
            println("diverging poems (expected = training, got = on-device):")
            diverging.forEach(::println)
        }

        // Regression guard: the on-device G2P MUST reproduce the training manifests exactly
        // for every poem in every split (train, holdout, and the external eval set). Any drop
        // here means text→phoneme drift between training and the app — a real correctness bug.
        // (All splits share the m3_v6 dot-token convention: '·' emits no token.)
        for ((split, a) in bySplit) {
            assertEquals("$split must match training manifests byte-for-byte", a.poems, a.exact)
        }
    }
}
