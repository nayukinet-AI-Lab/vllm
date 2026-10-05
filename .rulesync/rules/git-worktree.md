---
paths:
  - '**/*'
---
# Git Worktrees (implementation branch workflow)

## Hard rules (NEVER / ALWAYS)

When starting implementation work (feature / bugfix / hotfix):

- **NEVER**: run `git checkout -b <branch>` / `git switch -c <branch>` /
  `git branch <branch> && git checkout <branch>` in the main working tree.
- **ALWAYS**: create the branch and its worktree together:
  `git worktree add -b <branch> ../vllm-worktrees/<dir> origin/main`
  (use `feature/v100-fp16-patch` as the base ref instead of `origin/main` for V100/Volta work — see [git.md](./git.md)).
- **ALWAYS**: `cd` into the new worktree immediately, and run every subsequent
  Edit / Write / Bash command from inside it.

This applies to AI coding agents (Claude Code, Cursor, Codex CLI, etc.) too. Before an agent
makes its **first** file change via `Edit`/`Write`, it must run the preflight below and
state the result in the conversation.

## Preflight (run before starting any implementation)

```bash
# 1. Confirm whether you're in a worktree or the main checkout
git rev-parse --show-toplevel
pwd

# 2. List existing worktrees
git worktree list
```

Decision:

- `pwd` is under `.../vllm-worktrees/<...>` → OK, proceed with implementation.
- `pwd` is the main checkout (directly under `.../vllm`) and the task is
  ticket-driven implementation work → **STOP**. Create a worktree per
  "Creating a worktree" below, `cd` into it, then resume. Never create or
  commit to a feature branch in the main checkout without the user's explicit
  permission.

## Background and rationale

Why worktrees are required here:

- Lets you check `main` (or `feature/v100-fp16-patch`) or switch to another
  task without disturbing an in-progress build.
- Several branches of GPU/kernel work can proceed in parallel without one
  `setup.py develop` / incremental CMake build clobbering another's compiled
  `.so` artifacts.
- Each worktree gets its own `.venv`, `build/`, and `.deps/` anyway — the
  editable install hardcodes the absolute checkout path (`vllm.egg-link`,
  `easy-install.pth`, the `__editable__` finder, and the venv's own script
  shebangs all point at the exact path they were created under), so sharing a
  `.venv` across worktrees silently resolves imports against the wrong tree.

## Placement

Worktrees live **next to** the repo (sibling directory), under
`../vllm-worktrees/` — placing them inside the repo has side effects on IDE
indexing, ripgrep, and any bind-mounted Docker build context.

- Path: `../vllm-worktrees/<branch-name>`
- If the branch name contains a slash, convert it to a hyphen for the
  directory name
  - e.g. branch `feature/kv-cache-fix` → directory `feature-kv-cache-fix`

## Creating a worktree

```bash
# Fetch the latest refs first
git fetch origin upstream

# Create the branch and worktree together
# (base off origin/main for general work, or feature/v100-fp16-patch for
# Volta-specific work)
git worktree add -b feature/kv-cache-fix \
  ../vllm-worktrees/feature-kv-cache-fix \
  origin/main

# Move into the working directory
cd ../vllm-worktrees/feature-kv-cache-fix
```

## Per-worktree setup

- **Python environment**: each worktree needs its own `.venv` (see CLAUDE.md
  "Environment setup"); venvs aren't shareable across worktrees because of the
  absolute-path issue above.
  ```bash
  uv venv --python 3.12
  source .venv/bin/activate
  VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
  ```
- **Incremental C++/CUDA builds**: regenerate `CMakeUserPresets.json` per
  worktree (`python tools/generate_cmake_presets.py`) rather than copying one
  from another worktree — it embeds the worktree's absolute path. See
  [`docs/contributing/incremental_build.md`](../../docs/contributing/incremental_build.md).
  For V100/Volta work, build with `TORCH_CUDA_ARCH_LIST="7.0"` as documented
  in CLAUDE.md.
- **Shared caches** (safe to share — both already live under `$HOME`): the
  Hugging Face cache (`~/.cache/huggingface`) and `uv`'s package cache. No
  per-worktree copying needed for these.
- **`.deps/`, `build/`, `bin/`**: not shareable; each worktree builds its own.
  Expect the first build in a new worktree to take as long as a fresh clone.

## Cleanup

After the PR merges, or to discard the work, remove the worktree:

```bash
git worktree remove ../vllm-worktrees/feature-kv-cache-fix
git branch -d feature/kv-cache-fix
git worktree prune
```

`git worktree remove` fails if there are uncommitted changes. Only pass
`--force` when deliberately discarding the work.

## Exceptions (when it's OK to skip the worktree)

Work directly in the main working tree only when **all** of the following hold:

1. The change is a **single-file, ≤5-line** typo/comment/docs fix that doesn't
   alter code behavior.
2. The user has **explicitly said worktree-free is fine** in the current
   conversation.
3. A Claude Code Agent (`isolation: "worktree"`) is handling it as an
   **automatically isolated** short-lived task, or is reusing an existing
   worktree of the same name.

If none of the above applies, create a worktree — even if it "looks like a
quick fix" or "should only take one commit." When in doubt, create one.

## See also

- Branch naming and commit conventions: [git.md](./git.md)
- Incremental C++/CUDA build setup:
  [`docs/contributing/incremental_build.md`](../../docs/contributing/incremental_build.md)
