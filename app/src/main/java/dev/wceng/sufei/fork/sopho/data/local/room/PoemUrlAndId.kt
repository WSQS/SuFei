package dev.wceng.sufei.fork.sopho.data.local.room

/**
 * fork-specific: anthology member projection for ordering JOIN.
 *
 * 用 List 而非 Map 是为了保留 SQL 行序，使 fallback 排序确定性
 * （见 ADR-0005）。
 */
data class PoemUrlAndId(
    val sourceUrl: String,
    val id: String,
)
