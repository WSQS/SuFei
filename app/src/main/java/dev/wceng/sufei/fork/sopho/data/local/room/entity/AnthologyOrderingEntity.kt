package dev.wceng.sufei.fork.sopho.data.local.room.entity

import androidx.room.Entity

/** Anthology original-book ordering record (fork-specific). */
@Entity(
    tableName = "anthology_ordering",
    primaryKeys = ["pathId", "sourceUrl"]
)
data class AnthologyOrderingEntity(
    val pathId: String,
    val sourceUrl: String,
    val position: Int,
    val volume: String,
)
