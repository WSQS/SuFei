# ADR-0005 Original-book ordering via external JSONL + sourceUrl join

- Status: Accepted
- Date: 2026-07-22
- Related: issue #5, issue #6, PR #7

## Context

ADR-0002 accepted that anthology members would follow the original book's
editorial order, but noted as accepted debt that V1 used *data order*
(whatever `PoemDao.getPoemIdsByTag` returns) as a placeholder. Issue #5
asked to replace that placeholder with real ordering.

Two sub-decisions had to be made together: how to **acquire** the order
data, and how to **join** it against the `poems` table at runtime.

## Decision

Acquire ordering as machine-readable JSONL under
`assets/anthology_ordering/<anthologyId>.jsonl`, one line per poem, each
carrying `sourceUrl`, `position`, and `volume`. Persist it in a new
`anthology_ordering` Room table keyed by `(pathId, sourceUrl)`. At runtime,
`ReadingPathRepository` joins on `sourceUrl` and sorts by `position`.

The **stable identifier for matching** is `sourceUrl` (the gushiwen.cn poem
page URL), **not** `poemId`. Rationale:

- `poemId` is a UUID assigned at import time; it is meaningless outside this
  app's database and changes if the data source is ever replaced.
- `sourceUrl` is anchored to the data source (gushiwen.cn). The same URL hash
  appears in both the ordering JSONL and the `poems.sourceUrl` column, so the
  join survives re-imports.

## Alternatives considered

- **Order by poemId.** Rejected: a re-import or data-source swap invalidates
  the mapping silently; ordering would be wrong with no error.
- **Order by `(title, author)`.** Rejected as the *primary* key: duplicates
  exist within *唐诗三百首* alone (`隋宫` ×2 by 李商隐, `春思` ×2, `月夜`
  ×2, `凉州词` ×2). Used only as a last-resort fallback during acquisition,
  not at runtime.
- **Store ordering as a JSON blob column on `anthologies`.** Rejected: the
  data is a flat `(pathId, sourceUrl, position, volume)` tuple, not arbitrary
  JSON. A dedicated table is queryable, individually updatable, and
  consistent with how `reading_progress` is modeled (ADR-0004).
- **Hardcode ordering in Kotlin.** Rejected for the same reason as
  ADR-0003: coupling data authoring to code review and recompilation.

## Consequences

- **Acquisition is one-time and auditable.** Adding or correcting an
  anthology's order is a data-only change (edit the JSONL); no code change
  or recompilation is needed, though a new app build is required for
  distribution since the JSONL ships in assets.
- **Fallback is explicit.** When the ordering JSONL does not cover a poem
  (e.g., the 3 tag-gap poems in issue #6), that poem is not lost — it
  appends at the end in data order. This keeps the feature working while
  data issues are fixed independently.
- **`sourceUrl` is now a load-bearing column.** Any future change to how
  `poems.sourceUrl` is populated must preserve the URL-to-ordering
  correspondence, or the join breaks.
- **Cost: one extra table and a one-time import.** Negligible at the scale of
  classical anthologies (hundreds of rows, not thousands).
- **Volume is stored but unused in V1 UI.** It is cheap to carry and avoids
  a future schema migration when卷-level grouping is desired.
