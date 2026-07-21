package dev.wceng.sufei.fork.sopho.data.model

import dev.wceng.sufei.data.model.Poem

/**
 * 阅读路径（选集 + 用户进度）的聚合领域模型。
 *
 * 一个 ReadingPath 代表一组有顺序的诗词集合，如《古诗十九首》《唐诗三百首》。
 * - 定义侧（id / title / description / sourceTag）：来自 anthologies 表
 *   （assets/anthologies.jsonl 首次导入）
 * - 成员顺序（orderedPoemIds）：V1 由 sourceTag 反查 poems 表，按数据原序占位；
 *   后续可替换为原著顺序映射
 * - 进度侧（readPoemIds / currentPoemId）：来自 reading_progress 表
 *
 * @param id 选集稳定标识，如 "gushi_19", "tangshi_300"
 * @param title 显示名，如 "古诗十九首"
 * @param description 选集简介
 * @param sourceTag 对应诗句的 tags 字段值，用于从 poems 表反查成员
 * @param total 总篇数（= orderedPoemIds.size）
 * @param readPoemIds 本选集中已读的 poemId 集合。
 *   已读状态绑定在"选集+诗"组合上，同一首诗在不同选集里相互独立
 * @param currentPoemId 下一个该读的 poemId（已读之外的最小序号），
 *   null 表示未启动或已读完
 * @param orderedPoemIds 按路径顺序排列的 poemId 列表
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

    /**
     * 进度文案的三种语义状态，供 UI 层映射到字符串资源。
     * 放在 model 里集中定义，避免多个 Composable 各自判断。
     */
    val progressState: ProgressState get() = when {
        isCompleted -> ProgressState.Completed
        readCount == 0 -> ProgressState.NotStarted
        else -> ProgressState.InProgress
    }
}

enum class ProgressState { NotStarted, InProgress, Completed }

/**
 * 单首诗在路径中的视图：携带路径内位置和已读状态。
 */
data class PathItem(
    val poem: Poem,
    val order: Int,
    val isRead: Boolean,
    val readAt: Long?,
)
