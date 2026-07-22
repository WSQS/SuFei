package dev.wceng.sufei.fork.sopho.data.repository

import android.content.Context
import dagger.hilt.android.qualifiers.ApplicationContext
import dev.wceng.sufei.fork.sopho.data.local.room.AnthologyDao
import dev.wceng.sufei.fork.sopho.data.local.room.entity.toEntity
import dev.wceng.sufei.fork.sopho.data.model.Anthology
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import java.io.BufferedReader
import java.io.InputStreamReader
import javax.inject.Inject
import javax.inject.Singleton

/**
 * [ForkImportRepository] 实现：从 assets/anthologies.jsonl 灌入 anthologies 表。
 *
 * 幂等：表非空则跳过。单行 JSON 解析失败静默忽略；assets 缺失不影响启动
 * （研习 Tab 会显示空列表）。
 */
@Singleton
class ForkImportRepositoryImpl @Inject constructor(
    @ApplicationContext private val context: Context,
    private val anthologyDao: AnthologyDao,
) : ForkImportRepository {

    private val json = Json { ignoreUnknownKeys = true }

    override suspend fun startImportIfNeeded() {
        withContext(Dispatchers.IO) {
            if (anthologyDao.count() > 0) return@withContext
            importAnthologies()
        }
    }

    private suspend fun importAnthologies() {
        val toInsert = mutableListOf<Anthology>()
        try {
            context.assets.open("anthologies.jsonl").use { inputStream ->
                BufferedReader(InputStreamReader(inputStream)).useLines { lines ->
                    lines.forEach { line ->
                        if (line.isNotBlank()) {
                            runCatching { json.decodeFromString<Anthology>(line) }
                                .getOrNull()
                                ?.let(toInsert::add)
                        }
                    }
                }
            }
        } catch (_: Exception) {
            // assets 缺失或 IO 失败：不影响启动，选集列表为空
        }
        if (toInsert.isNotEmpty()) {
            anthologyDao.insertAll(toInsert.map { it.toEntity() })
        }
    }
}
