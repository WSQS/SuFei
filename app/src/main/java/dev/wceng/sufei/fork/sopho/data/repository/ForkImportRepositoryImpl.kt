package dev.wceng.sufei.fork.sopho.data.repository

import android.content.Context
import dagger.hilt.android.qualifiers.ApplicationContext
import dev.wceng.sufei.fork.sopho.data.local.room.AnthologyDao
import dev.wceng.sufei.fork.sopho.data.local.room.AnthologyOrderingDao
import dev.wceng.sufei.fork.sopho.data.local.room.entity.AnthologyOrderingEntity
import dev.wceng.sufei.fork.sopho.data.local.room.entity.toEntity
import dev.wceng.sufei.fork.sopho.data.model.Anthology
import dev.wceng.sufei.fork.sopho.data.model.AnthologyOrdering
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.CancellationException
import kotlinx.serialization.json.Json
import java.io.BufferedReader
import java.io.InputStreamReader
import javax.inject.Inject
import javax.inject.Singleton

/** Imports fork-specific data from assets into Room tables. */
@Singleton
class ForkImportRepositoryImpl @Inject constructor(
    @ApplicationContext private val context: Context,
    private val anthologyDao: AnthologyDao,
    private val anthologyOrderingDao: AnthologyOrderingDao,
) : ForkImportRepository {

    private val json = Json { ignoreUnknownKeys = true }

    override suspend fun startImportIfNeeded() {
        withContext(Dispatchers.IO) {
            if (anthologyDao.count() == 0) {
                importAnthologies()
            }
            if (anthologyOrderingDao.count() == 0) {
                importOrderings()
            }
        }
    }

    private suspend fun importAnthologies() {
        val toInsert = readJsonl<Anthology>("anthologies.jsonl")
        if (toInsert.isNotEmpty()) {
            anthologyDao.insertAll(toInsert.map { it.toEntity() })
        }
    }

    private suspend fun importOrderings() {
        val definitions = anthologyDao.getAll()
        val toInsert = mutableListOf<AnthologyOrderingEntity>()
        definitions.forEach { def ->
            val items = readJsonl<AnthologyOrdering>("anthology_ordering/${def.id}.jsonl")
            items.forEach { item ->
                toInsert.add(
                    AnthologyOrderingEntity(
                        pathId = def.id,
                        sourceUrl = item.sourceUrl,
                        position = item.position,
                        volume = item.volume,
                    )
                )
            }
        }
        if (toInsert.isNotEmpty()) {
            anthologyOrderingDao.insertAll(toInsert)
        }
    }

    private inline fun <reified T> readJsonl(path: String): List<T> {
        val result = mutableListOf<T>()
        try {
            context.assets.open(path).use { inputStream ->
                BufferedReader(InputStreamReader(inputStream)).useLines { lines ->
                    lines.forEach { line ->
                        if (line.isNotBlank()) {
                            runCatching { json.decodeFromString<T>(line) }
                                .getOrNull()
                                ?.let(result::add)
                        }
                    }
                }
            }
        } catch (ce: CancellationException) {
            throw ce
        } catch (_: Exception) {
        }
        return result
    }
}
