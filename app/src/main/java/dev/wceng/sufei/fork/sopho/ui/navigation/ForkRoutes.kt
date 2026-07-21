package dev.wceng.sufei.fork.sopho.ui.navigation

import kotlinx.serialization.Serializable

/**
 * fork-specific routes, isolated from upstream Routes.kt to avoid merge conflicts.
 *
 * These are declared in a separate package so upstream Routes.kt stays untouched.
 * Upstream files reference them via cross-package imports, which double as
 * visible "fork-specific" markers during merge review.
 */

/** 研习 Tab（经典选集列表）。 */
@Serializable
object Study

/**
 * 选集详情（有序诗列表 + 阅读进度）。
 *
 * @param pathId 对应 [dev.wceng.sufei.fork.sopho.data.model.Anthology.id]
 */
@Serializable
data class PathDetail(val pathId: String)
