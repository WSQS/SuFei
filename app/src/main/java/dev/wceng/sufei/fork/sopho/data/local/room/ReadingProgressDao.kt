package dev.wceng.sufei.fork.sopho.data.local.room

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import dev.wceng.sufei.fork.sopho.data.local.room.entity.ReadingProgressEntity
import kotlinx.coroutines.flow.Flow

/**
 * 阅读进度表 DAO（fork-specific）。
 *
 * 一行 = 某选集内某首诗的已读记录；复合主键 (pathId, poemId)，
 * 同一首诗在不同选集中的进度相互独立。
 */
@Dao
interface ReadingProgressDao {

    /** 某选集下全部已读记录，按路径内顺序索引升序。 */
    @Query("SELECT * FROM reading_progress WHERE pathId = :pathId ORDER BY position ASC")
    suspend fun getByPath(pathId: String): List<ReadingProgressEntity>

    /**
     * 观察全表行数，用作"任意进度变更"的脏标志。
     *
     * 任意 upsert/delete 都会改变 COUNT，从而驱动
     * [dev.wceng.sufei.fork.sopho.data.repository.ReadingPathRepository]
     * 重新聚合 ReadingPath / PathItem。
     */
    @Query("SELECT COUNT(*) FROM reading_progress")
    fun observeAnyChange(): Flow<Int>

    /** 标记已读；同 (pathId, poemId) 已存在则覆盖 position / readAt。 */
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(entity: ReadingProgressEntity)

    /** 取消某首诗在该选集内的已读标记。 */
    @Query("DELETE FROM reading_progress WHERE pathId = :pathId AND poemId = :poemId")
    suspend fun delete(pathId: String, poemId: String)
}
