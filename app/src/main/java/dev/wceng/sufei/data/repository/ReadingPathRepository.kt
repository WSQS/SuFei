package dev.wceng.sufei.data.repository

import dev.wceng.sufei.data.model.PathItem
import dev.wceng.sufei.data.model.ReadingPath
import kotlinx.coroutines.flow.Flow

/**
 * 阅读路径（选集）仓库 (fork-specific)
 *
 * 职责：把内置选集定义 + poems 表成员反查 + 用户阅读进度 三个数据源
 * 组合成 [ReadingPath] / [PathItem] 领域模型。
 *
 * 不修改既有 [PoemRepository]，职责独立。
 */
interface ReadingPathRepository {

    /** 观察所有内置选集及其进度（用于选集列表页） */
    fun observeAllPaths(): Flow<List<ReadingPath>>

    /** 观察单个选集（用于详情页头） */
    fun observePath(pathId: String): Flow<ReadingPath?>

    /** 观察某选集内全部诗，按路径顺序排列（用于详情页列表） */
    fun observePathItems(pathId: String): Flow<List<PathItem>>

    /** 标记某首诗为已读，并推进路径当前位置 */
    suspend fun markRead(pathId: String, poemId: String)

    /** 取消某首诗的已读标记 */
    suspend fun markUnread(pathId: String, poemId: String)
}
