package dev.wceng.sufei.fork.sopho.data.local.room

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import dev.wceng.sufei.fork.sopho.data.local.room.entity.AnthologyOrderingEntity

@Dao
interface AnthologyOrderingDao {

    @Query("SELECT * FROM anthology_ordering WHERE pathId = :pathId ORDER BY position ASC")
    suspend fun getByPath(pathId: String): List<AnthologyOrderingEntity>

    @Query("SELECT COUNT(*) FROM anthology_ordering")
    suspend fun count(): Int

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(items: List<AnthologyOrderingEntity>)
}
