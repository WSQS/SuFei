package dev.wceng.sufei.data.readingpath

/**
 * 内置选集定义（fork-specific，独立包避免侵入上游）
 *
 * V1 包含两个经典选集，其成员通过 sourceTag 从 poems 表反查。
 * 顺序映射（orderedPoemIds）在运行时从 DAO 取数后填充——
 * V1 先用数据原序占位，后续可替换为原著顺序映射表。
 *
 * 注：这两个标签存在于诗句的 tags 字段，但未进入 tags.jsonl，
 * 因此无法从 TagDao 查询，必须通过 PoemDao.getPoemIdsByTag 反查。
 */
data class AnthologyDefinition(
    val id: String,
    val title: String,
    val description: String,
    val sourceTag: String,
)

object BuiltInAnthologies {
    val all: List<AnthologyDefinition> = listOf(
        AnthologyDefinition(
            id = "gushi_19",
            title = "古诗十九首",
            description = "汉代无名氏所作，五言诗的源头与典范。情感真挚，语言自然，是诗歌入门的最佳起点。",
            sourceTag = "古诗十九首",
        ),
        AnthologyDefinition(
            id = "tangshi_300",
            title = "唐诗三百首",
            description = "清代蘅塘退士编选，涵盖唐诗各体裁，是流传最广的唐诗选本。",
            sourceTag = "唐诗三百首",
        ),
    )

    fun byId(id: String): AnthologyDefinition? = all.firstOrNull { it.id == id }
}
