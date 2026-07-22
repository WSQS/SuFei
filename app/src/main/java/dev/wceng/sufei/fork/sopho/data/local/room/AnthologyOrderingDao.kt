package dev.wceng.sufei.fork.sopho.data.local.room

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import dev.wceng.sufei.fork.sopho.data.local.room.entity.AnthologyOrderingEntity

/**
 * 选集原著顺序表 DAO（fork-specific）。
 *
 * 数据由 ForkImportRepository 从 assets/anthology_ordering/ 下的
 * JSONL 文件首次启动导入；运行时只读。
 *
 * 查询为 one-shot suspend（非 Flow）：顺序数据静态，进度变化由
 * ReadingProgressDao.observeAnyChange 驱动上层重算。
 */
@Dao
interface AnthologyOrderingDao {

    /** 某选集的全部顺序记录，按原著 position 升序。 */
    @Query("SELECT * FROM anthology_ordering WHERE pathId = :pathId ORDER BY position ASC")
    suspend fun getByPath(pathId: String): List<AnthologyOrderingEntity>

    /** 表行数；>0 表示已导入，import 流程可跳过。 */
    @Query("SELECT COUNT(*) FROM anthology_ordering")
    suspend fun count(): Int

    /** 批量写入；同 (pathId, sourceUrl) 已存在则 REPLACE。 */
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(items: List<AnthologyOrderingEntity>)
}
