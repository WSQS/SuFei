package dev.wceng.sufei.data.local.room

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import dev.wceng.sufei.data.local.room.entity.ReadingProgressEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface ReadingProgressDao {

    @Query("SELECT * FROM reading_progress WHERE pathId = :pathId ORDER BY position ASC")
    suspend fun getByPath(pathId: String): List<ReadingProgressEntity>

    /** 观察全表计数，用作"任意进度变更"的脏标志 */
    @Query("SELECT COUNT(*) FROM reading_progress")
    fun observeAnyChange(): Flow<Int>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(entity: ReadingProgressEntity)

    @Query("DELETE FROM reading_progress WHERE pathId = :pathId AND poemId = :poemId")
    suspend fun delete(pathId: String, poemId: String)

    @Query("DELETE FROM reading_progress WHERE pathId = :pathId")
    suspend fun deleteByPath(pathId: String)
}
