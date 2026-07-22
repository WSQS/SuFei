package dev.wceng.sufei.fork.sopho.data.model

import kotlinx.serialization.Serializable

/** Single line in the anthology_ordering JSONL files (fork-specific). */
@Serializable
data class AnthologyOrdering(
    val sourceUrl: String,
    val position: Int,
    val volume: String = "",
)
