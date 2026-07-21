package dev.wceng.sufei.data.local.room

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import dev.wceng.sufei.data.local.room.entity.AnthologyEntity

@Dao
interface AnthologyDao {

    @Query("SELECT * FROM anthologies ORDER BY id ASC")
    suspend fun getAll(): List<AnthologyEntity>

    @Query("SELECT * FROM anthologies WHERE id = :id")
    suspend fun getById(id: String): AnthologyEntity?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(anthologies: List<AnthologyEntity>)

    @Query("SELECT COUNT(*) FROM anthologies")
    suspend fun count(): Int
}
