package dev.wceng.sufei.fork.sopho.data.repository

/**
 * fork-specific：fork 所需数据的统一导入入口。
 *
 * 与上游 [ImportRepository] 职责平行但独立：
 * - [ImportRepository] 负责上游数据（诗词/作者/标签/词牌）
 * - [ForkImportRepository] 负责 fork 新增数据（选集定义）
 *
 * 由 [dev.wceng.sufei.ui.screens.splash.SplashViewModel] 在上游导入完成后调用。
 * 幂等：数据已存在时直接跳过。
 */
interface ForkImportRepository {
    suspend fun startImportIfNeeded()
}
