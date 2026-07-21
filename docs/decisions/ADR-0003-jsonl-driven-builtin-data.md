# ADR-0003 JSONL-driven built-in data with Room persistence

- Status: Accepted
- Date: 2026-07-21
- Related: PR #3, commit fb119f9

## Context

Built-in anthologies need to ship with the app and be queryable at runtime
(the reading-paths feature joins anthology definitions against the `poems`
table). Two sub-questions had to be decided at once: how the data is *authored*
and how it is *stored at runtime*.

Earlier in the feature, anthology definitions were Kotlin objects in a
`data.readingpath` package. That made every addition a code change requiring
recompilation and inflated the diff for a purely data-level concern.

## Decision

Author built-in data as JSONL under `app/src/main/assets/` (e.g.
`anthologies.jsonl`), consistent with the upstream import pattern
(`tags.jsonl`, `tunes.jsonl`, `poems_*.jsonl`). On first launch, a
fork-specific import repository reads the JSONL and seeds a Room table
(`anthologies`) that persists the definitions for runtime queries.

Adding a new anthology becomes a data-only change (edit the JSONL); no code
changes or recompilation of feature logic.

## Alternatives considered

- **Hardcoded Kotlin objects** (the V0 approach). Rejected: couples data
  authoring to code review; every addition touches source files and triggers
  rebuilds.
- **Pure runtime, no persistence** (read JSONL on every query). Rejected:
  need to join against Room tables anyway; re-parsing on each query is
  wasteful and breaks the reactive `Flow` model the repository layer uses.
- **Ship a prebuilt SQLite db**. Rejected: conflicts with upstream's
  migration-based schema evolution; would require parallel migration logic
  for the fork tables.

## Consequences

- Upside: authoring is a single-line JSONL edit; reviewers see data changes
  in isolation from logic changes.
- Upside: the persistence layer is parallel to upstream's
  (`ForkImportRepository` vs `ImportRepository`), keeping responsibilities
  disjoint.
- Cost: a one-time import and a Room table to maintain; a schema change to
  anthologies would require a new migration (v10 → v11 was the first).
- Cost: a malformed line is silently skipped to avoid blocking app launch;
  correctness depends on the JSONL being well-formed at author time.
