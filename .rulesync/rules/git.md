---
paths:
  - '**/*'
---
# Git Conventions

## Branch naming

- `feature/<short-description>` — new features/enhancements (e.g. `feature/v100-fp16-patch`). Branch from `main`, PR back to `main`.
- `bugfix/<short-description>` — bug fixes found before or after a release (e.g. `bugfix/cuda-graph-oom`). Branch from `main`.
- `hotfix/<short-description>` — urgent fixes for a production-impacting regression requiring an out-of-band patch release. Branch from `main`; cherry-pick onto the relevant `vX.Y.Z` release branch/tag if one exists.
- Reference the related GitHub issue number in the branch name when one exists (e.g. `bugfix/59286-lora-adapter-name`), but it isn't mandatory — a short descriptive slug is fine on its own.

**This fork specifically:** all V100/Volta (Compute Capability 7.0) compatibility work lives on the long-running `feature/v100-fp16-patch` branch (see CLAUDE.md "Repository context"). Topic branches for that work should branch from, and PR back into, `feature/v100-fp16-patch`, not `main`.

Creating an implementation branch **requires a git worktree** (`git checkout -b` alone is not enough). See [git-worktree.md](./git-worktree.md) for layout and steps.

## Commit messages

vLLM uses a bracketed-tag prefix, not Conventional Commits:

```
[Category][Subcategory] Description
```

- Categories mirror the area touched, e.g. `[Bugfix]`, `[Feature]`, `[Kernel]`, `[Platform]`, `[CI]`, `[Doc]`, `[V100]` (this fork's Volta patch set). Stack tags when useful: `[Fix][V100][Build]`.
- GitHub appends `(#<PR-number>)` automatically on squash-merge — don't add it by hand before the PR exists.
- Write commit messages in English.
- Every commit needs a DCO `Signed-off-by:` trailer (see [`docs/contributing/README.md`](../../docs/contributing/README.md) "DCO and Signed-off-by"). AI-assisted commits additionally need a `Co-authored-by:` trailer naming the agent (see CLAUDE.md).

## Staging

Don't use `git add -A`. Stage changed files explicitly.

```bash
git diff --name-only                  # review changed files
git add path/to/file1 path/to/file2   # stage explicitly
```

## PR conventions

- Title: `[Category] Short summary of the change` (same tag convention as commit messages).
- Base branch: `main` for general changes; `feature/v100-fp16-patch` for V100/Volta work on this fork.
- Body follows [`.github/PULL_REQUEST_TEMPLATE.md`](../../.github/PULL_REQUEST_TEMPLATE.md): **Purpose** / **Test Plan** / **Test Result**, plus the "Essential Elements of an Effective PR Description" checklist at the bottom.
- Link the GitHub issue(s) the PR resolves under **Purpose**, if any.
- Describe the change as a bullet list when it touches multiple areas.
- Paste the actual lint/test commands run and their output into **Test Result** — don't tick a checklist box without evidence.
- For AI-assisted PRs, follow CLAUDE.md's Contribution Policy: a human must review every changed line, duplicate-work checks (`gh issue view`, `gh pr list --search ...`) must be run first, and the description must state that AI assistance was used.

## Syncing `feature/v100-fp16-patch` with upstream `main`

This fork tracks `vllm-project/vllm` through the `upstream` remote. Periodically merge `upstream/main` into `feature/v100-fp16-patch` to pick up upstream fixes without losing the Volta patch history:

```bash
git fetch upstream
git checkout feature/v100-fp16-patch
git merge upstream/main -m "Merge upstream/main into feature/v100-fp16-patch"
git push origin feature/v100-fp16-patch   # or: git push origin HEAD:for-v100-fp16-patch
                                           # if the remote branch name differs locally
```

- **Always merge, never rebase or squash** this branch against upstream: rebasing would rewrite history that downstream users of this fork may already have based work on, and squashing would bury the V100 commit history that CLAUDE.md explicitly calls out as meaningful (e.g. the typing/`from __future__ import annotations` fixes that keep custom-op schema inference working under PyTorch 2.6).
- Resolve conflicts file by file rather than wholesale, and pay particular attention to `vllm/platforms/cuda.py`, `csrc/v100_adapter/`, and anything matching the `v100-csrc-adapter` rule's `applyTo` globs — those are the files most likely to need genuine V100-specific reconciliation rather than a trivial auto-merge.
- **Never force-push `main` or `feature/v100-fp16-patch`.** Both are long-lived branches; other clones of this fork (including `private-origin`) may be based on them.
- **Never click the "Delete branch" button** after merging a PR whose head is `main` or `feature/v100-fp16-patch` — unlike short-lived `feature/<x>`/`bugfix/<x>` topic branches, these are not disposable.
