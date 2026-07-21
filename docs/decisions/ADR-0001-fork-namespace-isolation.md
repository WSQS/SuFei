# ADR-0001 Fork namespace isolation under `dev.wceng.sufei.fork.sopho`

- Status: Accepted
- Date: 2026-07-21
- Related: PR #3

## Context

This repository is a downstream fork of an active upstream project. Features
developed in the fork must stay mergeable with upstream over time. Early fork
commits edited upstream files directly (appending a route here, adding an `is
Study` branch there), which worked for one feature but does not scale:

- Each upstream file edit is a future merge-conflict seed.
- There is no easy way, during review, to tell fork-specific code apart from
  upstream code once it lands in a shared file.
- Renames or moves by upstream silently break fork additions that were
  inlined into upstream modules.

A structural rule was needed so fork additions can be merged predictably
without giving up the ability to extend upstream behavior.

## Decision

All fork-specific code lives under the dedicated namespace
`dev.wceng.sufei.fork.sopho.*`, organized by layer (`ui`, `data`, `di`,
`domain`). Upstream files may import fork symbols but must not host fork
implementation.

When an upstream file must reference fork code, the
`import dev.wceng.sufei.fork.sopho...` line doubles as a visible
fork-specific marker during review and merge.

## Alternatives considered

- **Inline fork additions into upstream packages** (`dev.wceng.sufei.*`).
  Rejected: maximum merge pain, no visual distinction, rename/move conflicts
  on every upstream refactor.
- **Separate Gradle module** (`:fork`). Rejected for now: heavyweight for the
  current size; cross-module Hilt/Room wiring adds cost without proportional
  isolation benefit. Worth revisiting if the fork grows substantially.
- **Feature branches only, no namespace rule**. Rejected: branching strategy
  and code organization are orthogonal; branches don't prevent inline edits.

## Consequences

- Upside: upstream merges touch disjoint files; conflicts are rare and, when
  they happen, localized to the import line or small registration sites
  (`ScreensModule`, `AppDatabase`, `MainTab`).
- Upside: every fork addition is greppable via the namespace prefix.
- Cost: unavoidable small upstream edits remain (Hilt registration, Room
  entity/migration registration, tab enum entry). These are kept to one to
  three lines, localized, and marked `// fork-specific` in the commit.
- Cost: cross-package imports from upstream into fork are necessary and
  accepted as the visible cost of isolation.
