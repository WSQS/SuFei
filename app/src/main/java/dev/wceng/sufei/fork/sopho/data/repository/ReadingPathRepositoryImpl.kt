package dev.wceng.sufei.fork.sopho.data.repository

import dev.wceng.sufei.data.local.room.PoemDao
import dev.wceng.sufei.data.local.room.entity.toPoem
import dev.wceng.sufei.fork.sopho.data.local.room.AnthologyDao
import dev.wceng.sufei.fork.sopho.data.local.room.AnthologyOrderingDao
import dev.wceng.sufei.fork.sopho.data.local.room.ReadingProgressDao
import dev.wceng.sufei.fork.sopho.data.local.room.entity.ReadingProgressEntity
import dev.wceng.sufei.fork.sopho.data.local.room.entity.toAnthology
import dev.wceng.sufei.fork.sopho.data.model.Anthology
import dev.wceng.sufei.fork.sopho.data.model.PathItem
import dev.wceng.sufei.fork.sopho.data.model.ReadingPath
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.map
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ReadingPathRepositoryImpl @Inject constructor(
    private val poemDao: PoemDao,
    private val readingProgressDao: ReadingProgressDao,
    private val anthologyDao: AnthologyDao,
    private val anthologyOrderingDao: AnthologyOrderingDao,
) : ReadingPathRepository {

    private val progressTick: Flow<Unit> = readingProgressDao.observeAnyChange().map { }

    override fun observeAllPaths(): Flow<List<ReadingPath>> = progressTick
        .map {
            anthologyDao.getAll().map { entity ->
                val def = entity.toAnthology()
                val orderedIds = resolveOrderedIds(def)
                val progress = readingProgressDao.getByPath(def.id)
                resolvePath(def, orderedIds, progress)
            }
        }
        .flowOn(Dispatchers.IO)

    override fun observePath(pathId: String): Flow<ReadingPath?> = progressTick
        .map {
            val def = anthologyDao.getById(pathId)?.toAnthology() ?: return@map null
            val orderedIds = resolveOrderedIds(def)
            val progress = readingProgressDao.getByPath(pathId)
            resolvePath(def, orderedIds, progress)
        }
        .flowOn(Dispatchers.IO)

    override fun observePathItems(pathId: String): Flow<List<PathItem>> = progressTick
        .map {
            val def = anthologyDao.getById(pathId)?.toAnthology() ?: return@map emptyList()
            val orderedIds = resolveOrderedIds(def)
            val readMap = readingProgressDao.getByPath(pathId).associateBy { it.poemId }
            orderedIds.mapIndexedNotNull { index, poemId ->
                val entity = poemDao.getPoemById(poemId) ?: return@mapIndexedNotNull null
                PathItem(
                    poem = entity.toPoem(),
                    order = index,
                    isRead = readMap.containsKey(poemId),
                    readAt = readMap[poemId]?.readAt,
                )
            }
        }
        .flowOn(Dispatchers.IO)

    override suspend fun markRead(pathId: String, poemId: String) {
        val def = anthologyDao.getById(pathId)?.toAnthology() ?: return
        val orderedIds = resolveOrderedIds(def)
        val position = orderedIds.indexOf(poemId).takeIf { it >= 0 } ?: return
        readingProgressDao.upsert(
            ReadingProgressEntity(
                pathId = pathId,
                poemId = poemId,
                position = position,
                readAt = System.currentTimeMillis(),
            )
        )
    }

    override suspend fun markUnread(pathId: String, poemId: String) {
        readingProgressDao.delete(pathId, poemId)
    }

    private suspend fun resolveOrderedIds(def: Anthology): List<String> {
        val urlToId = poemDao.getSourceUrlToIdByTag(def.sourceTag)
        if (urlToId.isEmpty()) return emptyList()

        val orderMap = anthologyOrderingDao.getByPath(def.id)
            .associateBy { it.sourceUrl }

        val ordered = orderMap.values.mapNotNull { o ->
            urlToId[o.sourceUrl]?.let { o.position to it }
        }.sortedBy { it.first }.map { it.second }

        if (ordered.size == urlToId.size) return ordered

        val covered = ordered.toHashSet()
        val fallback = urlToId.values.filter { it !in covered }
        return ordered + fallback
    }

    private fun resolvePath(
        def: Anthology,
        orderedIds: List<String>,
        progress: List<ReadingProgressEntity>,
    ): ReadingPath {
        val readIds = progress.map { it.poemId }.toSet()
        val nextIndex = orderedIds.indexOfFirst { it !in readIds }
        val current = if (nextIndex >= 0) orderedIds[nextIndex] else null
        return ReadingPath(
            id = def.id,
            title = def.title,
            description = def.description,
            sourceTag = def.sourceTag,
            total = orderedIds.size,
            readPoemIds = readIds,
            currentPoemId = current,
            orderedPoemIds = orderedIds,
        )
    }
}
