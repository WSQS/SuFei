# ADR-0002 Anthology as the reading-path unit

- Status: Accepted
- Date: 2026-07-21
- Related: issue #1, PR #3

## Context

The fork's product direction is *guided reading* — leading the user through
classical poetry rather than offering an infinite library to browse. The first
design question was: what is the unit of a "reading path"?

The choices shape data model, UX, and how content gets curated.

## Decision

The **anthology** — a curated, ordered set of poems such as *古诗十九首*
("Nineteen Old Poems") or *唐诗三百首* ("300 Tang Poems") — is the atomic unit
of a reading path. Each anthology has a stable id, a display title, a short
description, and an ordered member list. Users progress path-by-path.

## Alternatives considered

- **Algorithmic recommendation** (e.g. "poems similar to what you read").
  Rejected: opaque, hard to curate, fights the "guided" intent. Better suited
  to an exploration feature than a reading path.
- **User-built playlists**. Rejected as the primary unit: high cold-start
  cost, requires curation UX we don't want to build yet, and most users want
  a trusted editor's selection. Could coexist later as a second path type.
- **Single-poem "paths"** (one poem at a time). Rejected: loses the notion of
  *progress through a collection*, which is the whole point.

## Consequences

- Upside: curation is human and finite; progress has clear start/end
  (`isCompleted`, `progress`), which the UI leans on.
- Cost (accepted debt): V1 derives anthology membership by reverse-looking-up
  the `tags` field on the `poems` table, which yields *data order* rather than
  the original book's order. A later ADR or commit can swap in an explicit
  ordered-id mapping without changing the model.
- Cost: same poem may appear in multiple anthologies; this forces the progress
  key design (see ADR-0004) but is otherwise desirable.
