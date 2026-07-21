package dev.wceng.sufei.fork.sopho.data.repository

import dev.wceng.sufei.fork.sopho.data.model.PathItem
import dev.wceng.sufei.fork.sopho.data.model.ReadingPath
import kotlinx.coroutines.flow.Flow

/**
 * 阅读路径（选集）仓库（fork-specific）。
 *
 * 职责：把「选集定义 + poems 表成员反查 + 用户阅读进度」三个数据源
 * 组合成 [ReadingPath] / [PathItem] 领域模型。
 *
 * 不修改既有 PoemRepository，职责独立。
 */
interface ReadingPathRepository {

    /** 观察所有内置选集及其进度（选集列表页）。 */
    fun observeAllPaths(): Flow<List<ReadingPath>>

    /** 观察单个选集（详情页头部）；pathId 不存在时发射 null。 */
    fun observePath(pathId: String): Flow<ReadingPath?>

    /** 观察某选集内全部诗，按路径顺序排列（详情页列表）。 */
    fun observePathItems(pathId: String): Flow<List<PathItem>>

    /**
     * 标记某首诗为已读。
     *
     * 若 pathId / poemId 无效，或该诗不在选集成员内，则静默忽略。
     */
    suspend fun markRead(pathId: String, poemId: String)

    /** 取消某首诗在该选集内的已读标记。 */
    suspend fun markUnread(pathId: String, poemId: String)
}
