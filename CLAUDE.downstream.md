# Downstream Fork Notes

This is a fork of the upstream repository. To minimize merge conflicts when syncing with upstream, follow these constraints.

## Working Principles

- **CLAUDE.md is read-only**: Never modify the upstream `CLAUDE.md`. All fork-specific instructions and notes belong in this file (`CLAUDE.downstream.md`).
- **Add, don't modify**: Prefer new files/packages/modules over editing upstream files.
- **Isolate, don't intrude**: Keep fork-specific changes in dedicated directories or via wrapper/composition patterns.
- **Localize environment changes**: Use machine-local config (e.g., global Gradle init scripts) instead of editing shared build files.

## Safe vs Risky Changes

| Safe (low conflict risk) | Risky (high conflict risk) |
|---|---|
| New files / packages / modules | Editing existing upstream files |
| New feature directories | Modifying shared components/utilities |
| Local config (init.gradle, CLAUDE.local.md) | Editing settings.gradle.kts / build.gradle.kts |
| Extension (new ViewModel/Screen) | Refactoring existing code structure |

## Decision Rule

When tempted to edit an upstream file, first ask: can this be achieved via extension (wrapper/inheritance/composition)? If an edit is unavoidable, keep it small and localized, and call it out in the commit message for easier merge resolution later.

## Product Direction: Reading Paths (导读模式)

The fork's core differentiation is shifting from a **tool** (search/retrieval) to a **service** (guided reading). The upstream app assumes users know what to read; the fork addresses the "where do I start" paralysis via curated reading paths.

### Core Decisions

1. **Path unit**: Classic anthologies (经典选集)
   - V1 ships with two: 《古诗十九首》(19 poems) and 《唐诗三百首》(318 poems).
   - Data already exists as tags in assets; reliable and finite.
   - Future: more anthologies (宋词三百首, 千家诗, etc.).

2. **Path ordering**: Original book order (原著顺序)
   - Anthologies are ordered as in their source books (e.g., 唐诗三百首 by 蘅塘退士's 体裁分类).
   - Note: data source order in `poems_*.jsonl` is NOT the original book order; a `order` index field is required.
   - Rationale: authority (centuries-tested), comparability with print editions, learnable literary progression.

3. **Guidance strength**: Focus + roamable (焦点 + 可漫游)
   - Default surface shows only the next poem to read; the full list is one tap away.
   - No hard locks: users can jump to any poem freely.
   - Read poems are auto-marked (progress) but not gated.
   - Goal: eliminate "where to start" anxiety without trapping users.

4. **Home integration**: Plan B (并存)
   - Upstream `HomeScreen` (card stack) is preserved unchanged.
   - A new "Study" (学习) tab is added to the bottom navigation, routing to a new dedicated study surface.
   - Entry point: a new `MainTab.Study` enum value (minimal upstream edits, see "Upstream Edits" below).

### Three-State Reading Model

Introduces a "read" (已读) middle state between unread and favorite:
```
未读 ──open detail──▶ 已读 ──curate──▶ 收���
```
- "Read" is auto-marked on detail open; it is NOT the same as favorite.
- Favorites return to their original meaning: the best of the best.
- Reading a poem inside a path advances path progress.

### Upstream Edits (unavoidable, kept minimal)

Three localized additions, isolated in one commit and labeled in the message:
1. `ui/navigation/Routes.kt` — append `@Serializable object Study`
2. `ui/navigation/MainTab.kt` — add `Study` enum entry + `toRoute()` branch
3. `ui/SuFeiApp.kt` — add `is Study -> MainTab.Study` to the `selectedTab` when-branch

### New Code (all in new packages/files)

- `data/model/ReadingPath.kt` — path domain model
- `data/local/room/entity/ReadingPathEntity.kt` + DAO (new table) — persisted path progress
- `data/repository/ReadingPathRepository.kt` (+Impl) — path queries, progress advancement
- `ui/screens/study/` — new feature directory
  - `StudyScreen.kt` — anthology picker + progress overview
  - `StudyViewModel.kt`
  - `PathDetailScreen.kt` — single anthology: progress bar, ordered list, "next poem" CTA
  - `PathDetailViewModel.kt`
- DetailScreen integration: add a "下一篇" (next) action when entering from a path (optional, V1.1)

## Branch Strategy

- `main` tracks upstream; keep it clean for syncing.
- `dev` is the working trunk for this fork's development.
- Feature work targets `dev` via **Pull Requests** (not direct pushes). Create a feature branch per unit of work, open a PR into `dev`, merge after review.

## Git Commit Convention

Use the **Angular commit convention**. Commit messages in English only.

Format:
```
<type>(<scope>): <subject>

<body>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.

Rules:
- Subject in imperative mood, lowercase, no trailing period, max 72 chars.
- Body explains the "why" and "what", wrapped at 72 chars.
- Only commit when explicitly requested; never commit automatically.
