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

## Fork Code Isolation

All fork-specific code lives under the dedicated namespace `dev.wceng.sufei.fork.sopho`. This keeps fork additions physically and logically separated from upstream code.

Package structure convention:
```
dev.wceng.sufei.fork.sopho/
├── ui/             # fork screens, components, navigation
├── data/           # fork repositories, datasources
├── di/             # fork Hilt modules (if any)
└── domain/         # fork usecases, models
```

Rules:
- New files go under `dev.wceng.sufei.fork.sopho.*`, not under `dev.wceng.sufei.*`.
- Cross-package imports from upstream code into fork code are expected and fine.
- When an upstream file must reference fork code, the `import dev.wceng.sufei.fork.sopho...` line itself serves as a visible "fork-specific" marker during merge review.
- Routes, models, and other stateless definitions that upstream might also add should ALWAYS be isolated to fork packages to avoid rename/move conflicts.

## Branch Strategy

- `main` tracks upstream; keep it clean for syncing.
- `dev` is the working trunk for this fork's development.
- Feature work targets `dev` via **Pull Requests** (not direct pushes). Create a feature branch per unit of work, open a PR into `dev`, merge after review.
- **Direct commits to `dev` are forbidden** unless the user explicitly allows it in a specific case. When unsure, always default to the PR workflow.

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
- **PR merge requires explicit user approval**. Never auto-merge; always ask first.
- **Direct commits to `dev` require explicit user approval per case**. Default to PR workflow when unsure.
