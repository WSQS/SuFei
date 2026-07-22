package dev.wceng.sufei.fork.sopho.data.model

import kotlinx.serialization.Serializable

@Serializable
data class AnthologyOrdering(
    val sourceUrl: String,
    val position: Int,
    val volume: String = "",
)
