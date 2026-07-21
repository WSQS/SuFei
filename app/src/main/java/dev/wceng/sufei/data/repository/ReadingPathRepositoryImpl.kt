package dev.wceng.sufei.data.repository

import dev.wceng.sufei.data.local.room.PoemDao
import dev.wceng.sufei.data.local.room.ReadingProgressDao
import dev.wceng.sufei.data.local.room.entity.ReadingProgressEntity
import dev.wceng.sufei.data.model.PathItem
import dev.wceng.sufei.data.model.Poem
import dev.wceng.sufei.data.model.ReadingPath
import dev.wceng.sufei.data.readingpath.BuiltInAnthologies
import dev.wceng.sufei.data.readingpath.AnthologyDefinition
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
                val progress = readMap[poemId]
                PathItem(
                    poem = Poem(
                        id = entity.id, sourceUrl = entity.sourceUrl,
                        title = entity.title, author = entity.author,
                        dynasty = entity.dynasty, content = entity.content,
                        tags = entity.tags, notes = entity.notes,
                        translation = entity.translation, intro = entity.intro,
                        background = entity.background,
                    ),
                    order = index,
                    isRead = progress != null,
                    readAt = progress?.readAt,
                )
            }
        }
        .flowOn(Dispatchers.IO)

    override suspend fun markRead(pathId: String, poemId: String) {
        val def = BuiltInAnthologies.byId(pathId) ?: return
        val orderedIds = poemDao.getPoemIdsByTag(def.sourceTag)
        val order = orderedIds.indexOf(poemId).takeIf { it >= 0 } ?: return
        readingProgressDao.upsert(
            ReadingProgressEntity(
                pathId = pathId, poemId = poemId, position = order,
                readAt = System.currentTimeMillis(),
            )
        )
    }

    override suspend fun markUnread(pathId: String, poemId: String) {
        readingProgressDao.delete(pathId, poemId)
    }

    override suspend fun resetPath(pathId: String) {
        readingProgressDao.deleteByPath(pathId)
    }

    override suspend fun nextUnread(pathId: String): Pair<String, Int>? {
        val def = BuiltInAnthologies.byId(pathId) ?: return null
        val orderedIds = poemDao.getPoemIdsByTag(def.sourceTag)
        val readIds = readingProgressDao.getByPath(pathId).map { it.poemId }.toSet()
        val nextIndex = orderedIds.indexOfFirst { it !in readIds }
        return if (nextIndex >= 0) orderedIds[nextIndex] to nextIndex else null
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
