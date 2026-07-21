package dev.wceng.sufei.fork.sopho.data.model

import kotlinx.serialization.Serializable

/**
 * 选集（阅读路径）的领域模型，纯数据定义。
 *
 * 选集通过 [sourceTag] 从 poems 表反查成员。
 * V1 用数据原序占位，后续可替换为原著顺序映射。
 *
 * fork-specific：由 ForkImportRepository 从 assets/anthologies.jsonl 首次导入。
 */
@Serializable
data class Anthology(
    val id: String,
    val title: String,
    val description: String,
    val sourceTag: String,
)
