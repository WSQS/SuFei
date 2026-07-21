package dev.wceng.sufei.fork.sopho.data.local.room.entity

import androidx.room.Entity

/**
 * 用户在某条 ReadingPath 内的阅读进度记录
 *
 * 复合主键 (pathId, poemId)：同一首诗可同时属于多个路径。
 *
 * @param pathId ReadingPath.id，如 "gushi_19"
 * @param poemId poems 表的外键
 * @param position 该诗在路径中的顺序索引（0-based）；命名为 position 以避免 SQLite 保留字 order
 * @param readAt 读时间戳（毫秒）
 */
@Entity(
    tableName = "reading_progress",
    primaryKeys = ["pathId", "poemId"]
)
data class ReadingProgressEntity(
    val pathId: String,
    val poemId: String,
    val position: Int,
    val readAt: Long,
)
