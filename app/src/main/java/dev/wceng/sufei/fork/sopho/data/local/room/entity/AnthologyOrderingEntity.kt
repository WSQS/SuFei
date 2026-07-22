package dev.wceng.sufei.fork.sopho.data.local.room.entity

import androidx.room.Entity

/**
 * 选集原著顺序记录实体（fork-specific）。
 *
 * 一行 = 某选集内某首诗的原著编排位置。
 * 按 (pathId, sourceUrl) 复合主键去重，用 sourceUrl 而非 poemId
 * 作稳定标识（见 ADR-0005）。
 *
 * 由 ForkImportRepository 从 assets/anthology_ordering/ 下的
 * JSONL 文件首次启动时导入；运行时只读。
 *
 * @param pathId 选集 id，对应 anthologies.id，如 "tangshi_300"
 * @param sourceUrl 古诗文网作品页 URL，对应 poems.sourceUrl
 * @param position 该诗在选集内的 1-based 顺序（1 = 第一首）
 * @param volume 卷名（如"五言绝句"）；本期 UI 未使用，供未来分卷展示
 */
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
