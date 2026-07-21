package dev.wceng.sufei.fork.sopho.data.local.room.entity

import androidx.room.Entity

/**
 * 阅读进度记录实体（fork-specific）。
 *
 * 一行 = 某用户在某选集内读完某首诗的标记。
 * 复合主键 (pathId, poemId)：同一首诗可同时属于多个选集，各选集中的已读状态相互独立。
 *
 * @param pathId 选集 id，对应 anthologies.id，如 "gushi_19"
 * @param poemId 诗 id，对应 poems.id（逻辑外键，未声明 FK 约束）
 * @param position 该诗在选集中的顺序索引（0-based）。
 *   命名为 position 以避开 SQLite 保留字 order。
 *   V1 来自数据原序，后续可替换为原著顺序
 * @param readAt 首次标记为已读的时间戳（毫秒，epoch）。重新标记会被覆盖
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
