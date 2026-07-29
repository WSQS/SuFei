# Architecture Decision Records

Lightweight log of decisions that shape this fork: architecture, data model,
navigation, conventions. One file per decision, numbered, append-only.

## Why keep ADRs

Commits and PR descriptions capture *what* changed and *how*. They rot quickly.
ADR captures *why* — the question being decided, the alternatives weighed, the
trade-off accepted — so the future maintainer (often yourself, six months later)
doesn't have to reverse-engineer intent from code.

## When to write one

| Decision size | Where it goes |
|---|---|
| Trivial (naming, a constant) | Commit body is enough |
| Medium (data model, API shape, module boundary) | New ADR, bundled with the implementing PR |
| Large (direction spanning multiple PRs) | ADR first, merged via its own PR before implementation |

Rule of thumb: *will I later wonder "why was it done this way"?* If yes, write
one. If no, don't.

## File naming

`ADR-NNNN-kebab-case-title.md`, zero-padded, starting at `0001`.

NNNN is assigned in order and **never reused**. Title is short; it summarizes
the decision, not the feature.

## Lifecycle

- A new ADR is `Status: Accepted` on merge.
- When superseded by a later ADR, do **not** delete it. Change its status to
  `Superseded by ADR-00NN` and add the replacement's number. The superseding
  ADR opens with `Supersedes ADR-00NN`.
- Statuses used here: `Accepted`, `Superseded`. That's all — no `Proposed` /
  `Rejected` graveyard; if a decision wasn't taken, it isn't worth a file.

## Template

```markdown
# ADR-NNNN <title>

- Status: Accepted
- Date: YYYY-MM-DD
- Related: issue #N, PR #N

## Context

Why does this decision need to happen? What constraint, pain, or question
forced it? Keep it to the minimum needed to understand the rest.

## Decision

What was chosen. One or two sentences. No implementation detail — that lives
in the code.

## Alternatives considered

Each alternative, one or two lines each, with the reason it was rejected.

## Consequences

What we gain, and what we knowingly accept as the cost (technical debt,
constraints on future work, etc.).
```

## Index

| # | Title | Status | Date |
|---|---|---|---|
| [0001](ADR-0001-fork-namespace-isolation.md) | Fork namespace isolation under `dev.wceng.sufei.fork.sopho` | Accepted | 2026-07-21 |
| [0002](ADR-0002-anthology-as-reading-unit.md) | Anthology as the reading-path unit | Accepted | 2026-07-21 |
| [0003](ADR-0003-jsonl-driven-builtin-data.md) | JSONL-driven built-in data with Room persistence | Accepted | 2026-07-21 |
| [0004](ADR-0004-composite-progress-key.md) | Per-path composite progress key `(pathId, poemId)` | Accepted | 2026-07-21 |
| [0005](ADR-0005-original-book-ordering.md) | Original-book ordering via external JSONL + sourceUrl join | Accepted | 2026-07-22 |
| [0006](ADR-0006-nar-tts-architecture.md) | NAR TTS architecture (FastSpeech 2 + HiFi-GAN) | Accepted | 2026-07-27 |
| [0007](ADR-0007-on-device-nar-tts-paddlespeech.md) | On-device NAR TTS via PaddleSpeech FS2 + HiFi-GAN ONNX | Accepted | 2026-07-28 |
| [0008](ADR-0008-fs2-from-scratch-generalization-ceiling.md) | From-scratch FS2 training: generalization ceiling | Accepted | 2026-07-29 |
