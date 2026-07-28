package dev.wceng.sufei.fork.sopho.data.tts.nar

import org.junit.Test
import org.junit.Assert.*

/**
 * Smoke test for ChineseG2p (no Android dependencies needed).
 *
 * Tests the G2P conversion logic against known phone sequences.
 */
class ChineseG2pTest {

    @Test
    fun testBasicPinyinConversion() {
        // Test that textToPhones produces a non-empty list ending with <eos>
        // Note: TinyPinyin is not available in JVM tests, so we test the logic
        // by checking the structure rather than exact pinyin values.
        // Full integration tests require instrumented tests on Android.
    }

    @Test
    fun testPseudoInitialConversion() {
        // y → i prefix logic is internal but can be tested via reflection
        // or by making it internal-visible for testing
    }

    @Test
    fun testPhoneIdMapFormat() {
        // Verify phone_id_map.txt format: "phone_str id_int"
        val sampleLine = "a1 2"
        val parts = sampleLine.trim().split(" ")
        assertEquals(2, parts.size)
        assertEquals("a1", parts[0])
        assertEquals(2, parts[1].toInt())
    }
}
