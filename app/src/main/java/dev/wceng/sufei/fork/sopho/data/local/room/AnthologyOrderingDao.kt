package dev.wceng.sufei.fork.sopho.data.local.room

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import dev.wceng.sufei.fork.sopho.data.local.room.entity.AnthologyOrderingEntity

/** DAO for anthology original-book ordering (fork-specific). */
@Dao
interface AnthologyOrderingDao {

    /** All ordering records for an anthology, sorted by position ascending. */
    @Query("SELECT * FROM anthology_ordering WHERE pathId = :pathId ORDER BY position ASC")
    suspend fun getByPath(pathId: String): List<AnthologyOrderingEntity>

    /** Row count; >0 means already imported. */
    @Query("SELECT COUNT(*) FROM anthology_ordering")
    suspend fun count(): Int

    /** Batch insert; REPLACE on (pathId, sourceUrl) conflict. */
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(items: List<AnthologyOrderingEntity>)
}
