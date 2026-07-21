package dev.wceng.sufei.fork.sopho.data.local.room

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import dev.wceng.sufei.fork.sopho.data.local.room.entity.AnthologyEntity

/**
 * 选集定义表 DAO（fork-specific）。
 *
 * 数据由 [dev.wceng.sufei.fork.sopho.data.repository.ForkImportRepository]
 * 从 assets/anthologies.jsonl 首次启动导入；运行时只读，用户不修改。
 *
 * 查询为 one-shot suspend（非 Flow）：选集定义静态，进度变化由
 * [ReadingProgressDao.observeAnyChange] 驱动上层重算。
 */
@Dao
interface AnthologyDao {

    /** 全部选集，按 id 升序（稳定展示顺序）。 */
    @Query("SELECT * FROM anthologies ORDER BY id ASC")
    suspend fun getAll(): List<AnthologyEntity>

    /** 按稳定 id 取单条；不存在时返回 null。 */
    @Query("SELECT * FROM anthologies WHERE id = :id")
    suspend fun getById(id: String): AnthologyEntity?

    /** 批量写入；同 id 已存在则 REPLACE（便于 assets 更新后重灌）。 */
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(anthologies: List<AnthologyEntity>)

    /** 表行数；>0 表示已导入，import 流程可跳过。 */
    @Query("SELECT COUNT(*) FROM anthologies")
    suspend fun count(): Int
}
