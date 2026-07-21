package dev.wceng.sufei.data.model

/**
 * 阅读路径（选集）的领域模型
 *
 * 一个 ReadingPath 代表一组有顺序的诗词集合，如《古诗十九首》《唐诗三百首》。
 * 选集定义（id / title / sourceTag / orderedPoemIds）内置在代码中，
 * 用户进度（已读集合 / 当前位置）从 Room 读取。
 *
 * @param id 选集稳定标识，如 "gushi_19", "tangshi_300"
 * @param title 显示名，如 "古诗十九首"
 * @param description 选集简介
 * @param sourceTag 对应诗句的 tags 字段值，用于从 poems 表反查成员
 * @param total 总篇数
 * @param readPoemIds 用户已读的 poemId 集合（按 path 维度）
 * @param currentPoemId 下一个该读的 poemId（已读之外的最小序号），null 表示未启动或已读完
 * @param orderedPoemIds 按预定顺序排列的 poemId 列表（V1 用数据原序占位，后续替换为原著顺序）
 */
data class ReadingPath(
    val id: String,
    val title: String,
    val description: String,
    val sourceTag: String,
    val total: Int,
    val readPoemIds: Set<String>,
    val currentPoemId: String?,
    val orderedPoemIds: List<String>,
) {
    val readCount: Int get() = readPoemIds.size
    val progress: Float get() = if (total == 0) 0f else readCount.toFloat() / total
    val isCompleted: Boolean get() = total > 0 && readCount >= total
}

/**
 * 单首诗在路径中的视图：携带路径内位置和已读状态
 */
data class PathItem(
    val poem: Poem,
    val order: Int,
    val isRead: Boolean,
    val readAt: Long?,
)
