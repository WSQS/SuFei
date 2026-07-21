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
            val inputStream = context.assets.open("anthologies.jsonl")
            BufferedReader(InputStreamReader(inputStream)).useLines { lines ->
                lines.forEach { line ->
                    if (line.isNotBlank()) {
                        runCatching { json.decodeFromString<Anthology>(line) }
                            .getOrNull()
                            ?.let(toInsert::add)
                    }
                }
            }
        } catch (e: Exception) {
            // assets 文件缺失不影响 App 启动，选集功能会显示空列表
        }
        if (toInsert.isNotEmpty()) {
            anthologyDao.insertAll(toInsert.map { it.toEntity() })
        }
    }
}
