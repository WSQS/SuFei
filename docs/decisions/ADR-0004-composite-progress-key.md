# ADR-0004 Per-path composite progress key `(pathId, poemId)`

- Status: Accepted
- Date: 2026-07-21
- Related: issue #1, PR #3

## Context

The reading-paths feature needs to record "the user has read poem P within
anthology A". Because the same poem can belong to multiple anthologies
(ADR-0002 accepts this as desirable), the storage shape for progress is a
real decision: it determines whether reading a poem in one anthology marks it
read everywhere, and how reactivity is wired.

## Decision

Store one progress row per `(pathId, poemId)` pair, with those two columns as
a composite primary key. Reading a poem marks it read only within its
anthology; the same poem in another anthology remains unread until visited
there.

## Alternatives considered

- **Single global `readPoemIds` set** (poem-level, path-agnostic). Rejected:
  reading 《静夜思》 in *唐诗三百首* would silently mark it read in every other
  anthology too, which misrepresents the user's actual progress through each
  book and breaks per-path completion percentages.
- **One row per path with a packed `readPoemIds` column** (array/blob).
  Rejected: every mark/unmark rewrites the whole row; hard to index and query;
  fights Room's row-oriented model.
- **Foreign keys with cascade**. Not adopted: `poemId` is a *logical* foreign
  key only. Declaring a hard FK would couple fork tables to upstream schema
  migrations and raise the cost of upstream `poems` schema changes.

## Consequences

- Upside: per-path progress is honest — completing 《唐诗三百首》 doesn't lie
  about 《古诗十九首》.
- Upside: marking/unmarking is a single-row upsert/delete; cheap and easy to
  reason about.
- Upside: reactivity is simple — a table-wide `COUNT(*)` `Flow` acts as a
  dirty flag that drives recomputation of all derived `ReadingPath` /
  `PathItem` flows. This avoids per-row observe queries.
- Cost: storage is proportional to (paths × read-poems); negligible at the
  scale of classical anthologies, but worth revisiting if user-built paths
  ever land (ADR-0002 consequence).
- Cost: no hard FK means a deleted poem could leave an orphan progress row.
  Accepted, since built-in poems are immutable in practice.
