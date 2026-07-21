package dev.wceng.sufei.fork.sopho.data.model

import kotlinx.serialization.Serializable

/**
 * 选集定义的领域模型（纯数据，不含用户进度）。
 *
 * 由 [dev.wceng.sufei.fork.sopho.data.repository.ForkImportRepository]
 * 从 assets/anthologies.jsonl 首次导入；带进度的聚合模型见 [ReadingPath]。
 *
 * @param id 选集稳定标识，如 "gushi_19"、"tangshi_300"
 * @param title 显示名，如 "古诗十九首"
 * @param description 选集简介
 * @param sourceTag 对应 poems.tags 字段值，用于反查成员；
 *   V1 顺序为数据原序，后续可换原著顺序
 */
@Serializable
data class Anthology(
    val id: String,
    val title: String,
    val description: String,
    val sourceTag: String,
)
