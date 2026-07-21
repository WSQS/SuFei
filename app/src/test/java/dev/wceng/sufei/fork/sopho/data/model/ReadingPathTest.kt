package dev.wceng.sufei.fork.sopho.data.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ReadingPathTest {

    private fun path(
        total: Int,
        readPoemIds: Set<String>,
        orderedPoemIds: List<String>,
    ): ReadingPath = ReadingPath(
        id = "test",
        title = "测试选集",
        description = "",
        sourceTag = "tag",
        total = total,
        readPoemIds = readPoemIds,
        currentPoemId = orderedPoemIds.firstOrNull { it !in readPoemIds },
        orderedPoemIds = orderedPoemIds,
    )

    @Test
    fun `progress is zero when total is zero`() {
        val p = path(total = 0, readPoemIds = emptySet(), orderedPoemIds = emptyList())
        assertEquals(0f, p.progress, 0.001f)
    }

    @Test
    fun `progress is fraction when partially read`() {
        val ids = listOf("a", "b", "c", "d")
        val p = path(total = 4, readPoemIds = setOf("a", "b"), orderedPoemIds = ids)
        assertEquals(0.5f, p.progress, 0.001f)
    }

    @Test
    fun `progress is one when all read`() {
        val ids = listOf("a", "b")
        val p = path(total = 2, readPoemIds = setOf("a", "b"), orderedPoemIds = ids)
        assertEquals(1f, p.progress, 0.001f)
        assertTrue(p.isCompleted)
    }

    @Test
    fun `isCompleted false when nothing read`() {
        val p = path(total = 3, readPoemIds = emptySet(), orderedPoemIds = listOf("a", "b", "c"))
        assertFalse(p.isCompleted)
    }

    @Test
    fun `progressState is NotStarted when readCount zero`() {
        val p = path(total = 3, readPoemIds = emptySet(), orderedPoemIds = listOf("a", "b", "c"))
        assertEquals(ProgressState.NotStarted, p.progressState)
    }

    @Test
    fun `progressState is InProgress when partially read`() {
        val ids = listOf("a", "b", "c")
        val p = path(total = 3, readPoemIds = setOf("a"), orderedPoemIds = ids)
        assertEquals(ProgressState.InProgress, p.progressState)
    }

    @Test
    fun `progressState is Completed when all read`() {
        val ids = listOf("a", "b")
        val p = path(total = 2, readPoemIds = setOf("a", "b"), orderedPoemIds = ids)
        assertEquals(ProgressState.Completed, p.progressState)
    }

    @Test
    fun `currentPoemId is first unread`() {
        val ids = listOf("a", "b", "c")
        val p = path(total = 3, readPoemIds = setOf("a"), orderedPoemIds = ids)
        assertEquals("b", p.currentPoemId)
    }

    @Test
    fun `currentPoemId is null when all read`() {
        val ids = listOf("a", "b")
        val p = path(total = 2, readPoemIds = setOf("a", "b"), orderedPoemIds = ids)
        assertNull(p.currentPoemId)
    }
}
