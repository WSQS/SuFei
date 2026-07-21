package dev.wceng.sufei.data.repository

import dev.wceng.sufei.data.local.room.PoemDao
import dev.wceng.sufei.data.local.room.ReadingProgressDao
import dev.wceng.sufei.data.local.room.entity.ReadingProgressEntity
import dev.wceng.sufei.data.local.room.entity.toPoem
import dev.wceng.sufei.data.model.PathItem
import dev.wceng.sufei.data.model.ReadingPath
import dev.wceng.sufei.data.readingpath.AnthologyDefinition
import dev.wceng.sufei.data.readingpath.BuiltInAnthologies
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
) : ReadingPathRepository {

    /**
     * 观察全表计数——任何路径的进度变化都会触发它重新发射，
     * 用作"任意进度变更"的脏标志，驱动所有选集的重算。
     *
     * 注：选集成员查询是静态的（数据导入后不变），
     * 因此不必观察 poems 表，只观察 reading_progress 即可。
     */
    private val progressTick: Flow<Unit> = readingProgressDao.observeAnyChange().map { }

    override fun observeAllPaths(): Flow<List<ReadingPath>> = progressTick
        .map {
            BuiltInAnthologies.all.map { def ->
                resolvePath(def, readingProgressDao.getByPath(def.id))
            }
        }
        .flowOn(Dispatchers.IO)

    override fun observePath(pathId: String): Flow<ReadingPath?> = progressTick
        .map {
            val def = BuiltInAnthologies.byId(pathId) ?: return@map null
            resolvePath(def, readingProgressDao.getByPath(pathId))
        }
        .flowOn(Dispatchers.IO)

    override fun observePathItems(pathId: String): Flow<List<PathItem>> = progressTick
        .map {
            val def = BuiltInAnthologies.byId(pathId) ?: return@map emptyList()
            val orderedIds = poemDao.getPoemIdsByTag(def.sourceTag)
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
        val def = BuiltInAnthologies.byId(pathId) ?: return
        val orderedIds = poemDao.getPoemIdsByTag(def.sourceTag)
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

    private suspend fun resolvePath(
        def: AnthologyDefinition,
        progress: List<ReadingProgressEntity>,
    ): ReadingPath {
        val orderedIds = poemDao.getPoemIdsByTag(def.sourceTag)
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
