package dev.wceng.sufei.fork.sopho.ui.navigation

import kotlinx.serialization.Serializable

/**
 * fork-specific routes, isolated from upstream Routes.kt to avoid merge conflicts.
 *
 * These are declared in a separate package so upstream Routes.kt stays untouched.
 * Upstream files reference them via cross-package imports, which double as
 * visible "fork-specific" markers during merge review.
 */

// Study tab (Reading Paths anthology list)
@Serializable
object Study

// Path detail (single anthology content + progress)
@Serializable
data class PathDetail(val pathId: String)
