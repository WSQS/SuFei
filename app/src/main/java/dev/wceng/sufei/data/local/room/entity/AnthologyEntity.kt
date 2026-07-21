package dev.wceng.sufei.data.local.room.entity

import androidx.room.Entity
import androidx.room.PrimaryKey
import dev.wceng.sufei.data.model.Anthology

/**
 * 选集定义表（fork-specific）
 *
 * 由 ForkImportRepository 从 assets/anthologies.jsonl 首次启动时导入。
 * 运行时只读，用户不修改。
 */
@Entity(tableName = "anthologies")
data class AnthologyEntity(
    @PrimaryKey
    val id: String,
    val title: String,
    val description: String,
    val sourceTag: String,
)

fun AnthologyEntity.toAnthology() = Anthology(
    id = id,
    title = title,
    description = description,
    sourceTag = sourceTag,
)

fun Anthology.toEntity() = AnthologyEntity(
    id = id,
    title = title,
    description = description,
    sourceTag = sourceTag,
)
