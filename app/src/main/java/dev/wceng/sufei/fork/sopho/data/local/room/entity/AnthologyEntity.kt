package dev.wceng.sufei.fork.sopho.data.local.room.entity

import androidx.room.Entity
import androidx.room.PrimaryKey
import dev.wceng.sufei.fork.sopho.data.model.Anthology

/**
 * 选集定义表实体（fork-specific）。
 *
 * 由 [dev.wceng.sufei.fork.sopho.data.repository.ForkImportRepository]
 * 从 assets/anthologies.jsonl 首次启动时导入；运行时只读，用户不修改。
 *
 * 与 [dev.wceng.sufei.fork.sopho.data.model.ReadingProgressEntity] 不同，
 * 这里只存定义，不含用户进度。
 *
 * @param id 选集稳定标识，主键，如 "gushi_19"、"tangshi_300"
 * @param title 显示名，如 "古诗十九首"
 * @param description 选集简介（列表/详情页副标题文案）
 * @param sourceTag 对应 poems 表 tags 字段值，用于反查该选集的成员诗
 */
@Entity(tableName = "anthologies")
data class AnthologyEntity(
    @PrimaryKey
    val id: String,
    val title: String,
    val description: String,
    val sourceTag: String,
)

/** Entity → 领域模型。字段一一对应，无附加逻辑。 */
fun AnthologyEntity.toAnthology() = Anthology(
    id = id,
    title = title,
    description = description,
    sourceTag = sourceTag,
)

/** 领域模型 → Entity（导入写入用）。 */
fun Anthology.toEntity() = AnthologyEntity(
    id = id,
    title = title,
    description = description,
    sourceTag = sourceTag,
)
