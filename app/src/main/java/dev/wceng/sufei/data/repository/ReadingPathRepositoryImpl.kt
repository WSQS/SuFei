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
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.map
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ReadingPathRepositoryImpl @Inject constructor(
    private val poemDao: PoemDao,
    private val readingProgressDao: ReadingProgressDao,
) : ReadingPathRepository {

    override fun observeAllPaths(): Flow<List<ReadingPath>> = flow {
        val definitions = BuiltInAnthologies.all
        val paths = definitions.map { def ->
            resolvePath(def, readingProgressDao.getByPath(def.id))
        }
        emit(paths)
    }.flowOn(Dispatchers.IO)

    override fun observePath(pathId: String): Flow<ReadingPath?> = flow {
        val def = BuiltInAnthologies.byId(pathId) ?: run { emit(null); return@flow }
        emit(resolvePath(def, readingProgressDao.getByPath(pathId)))
    }.flowOn(Dispatchers.IO)

    override fun observePathItems(pathId: String): Flow<List<PathItem>> = flow {
        val def = BuiltInAnthologies.byId(pathId) ?: run { emit(emptyList()); return@flow }
        val orderedIds = poemDao.getPoemIdsByTag(def.sourceTag)  // V1: 数据原序占位
        val readMap = readingProgressDao.getByPath(pathId).associateBy { it.poemId }
        val items = orderedIds.mapIndexedNotNull { index, poemId ->
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
        emit(items)
    }.flowOn(Dispatchers.IO)

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
