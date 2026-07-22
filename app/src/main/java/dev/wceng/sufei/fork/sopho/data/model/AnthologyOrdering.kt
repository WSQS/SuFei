package dev.wceng.sufei.fork.sopho.data.model

import kotlinx.serialization.Serializable

/**
 * anthology_ordering JSONL 单行模型（fork-specific���。
 *
 * 采集自古诗文网作品专题页，记录每首诗在原著中的位置。
 * sourceUrl 用于跟 poems 表 JOIN，不用 poemId 以避免数据源
 * 更换导致顺序失效（见 ADR-0005）。
 *
 * @param sourceUrl 古诗文网作品页 URL，对应 poems.sourceUrl
 * @param position 1-based 顺序
 * @param volume 卷名（如"五言绝句"），可选，供未来分卷展示
 */
@Serializable
data class AnthologyOrdering(
    val sourceUrl: String,
    val position: Int,
    val volume: String = "",
)
