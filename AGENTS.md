# Agent Instructions for vLLM (V100 Fork)

> These instructions apply to **all** AI-assisted contributions to this repository.
> Breaching the contribution policy below can result in automatic banning.

## Repository context

This is a fork of [vllm-project/vllm](https://github.com/vllm-project/vllm) (origin:
`saitama-AI-Lab/vllm`) with an active branch, `feature/v100-fp16-patch`, that adds
compatibility for **NVIDIA Tesla V100 (Volta, Compute Capability 7.0)** GPUs. V100 lacks
hardware BF16 and FlashAttention v1/v2/v3 support, so upstream vLLM crashes on it out of
the box.

V100-specific implementation rules live in the `v100-csrc-adapter` rule; read it before
touching attention/dtype code or `csrc/v100_adapter/`. The design docs under
`docs/v100_fallback_archtecture/` are the source of truth for that work.

### V100 patch rules (critical — read before editing attention/dtype code)

1. Always gate Volta-specific behavior on `device_capability < DeviceCapability(8, 0)`
   (see `vllm/platforms/cuda.py`). Never let these overrides change behavior for CC >= 8.0
   (Ampere/Hopper/Blackwell) devices.
2. Dtype fallback: BF16 -> FP16 for weights, activations, and KV cache on CC < 8.0.
3. Attention backend fallback: `FLASH_ATTN` and `FLASHINFER` are excluded from the backend
   priority list on CC < 8.0 (see the `base_priorities` filtering in
   `vllm/platforms/cuda.py`); falls through to `TRITON_ATTN`/`FLEX_ATTENTION`.
4. Triton kernels using `tl.bfloat16` must safely degrade/map to `tl.float16` on Volta.
5. Do not insert `getattr`/`hasattr`/dtype-branching logic into Python model definition
   files (e.g. `qwen2.py`, `gemma*.py`) — keep model files upstream-compliant. sm_70
   custom-op gaps are trapped in the C++ adapter layer under `csrc/v100_adapter/` via
   `TORCH_LIBRARY_IMPL`, not Python monkey-patching. See the `v100-csrc-adapter` rule and
   the `v100-add-fallback-op` skill.
6. Many of the other commits on this branch (typing/`from __future__ import annotations`
   fixes in kernels, MoE, quantization) exist to keep custom op schema inference working
   under PyTorch 2.6 — unrelated to V100 per se, but part of the same patch set.

---

## 1. Contribution Policy (Mandatory)

### Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "<issue_number> in:body"
gh pr list --repo vllm-project/vllm --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

### No low-value busywork PRs

Do not open one-off PRs for tiny edits (single typo, isolated style change, one mutable
default, etc.). Mechanical cleanups are acceptable only when bundled with substantive
work.

### Accountability

- Pure code-agent PRs are **not allowed**. A human submitter must understand and defend
  the change end-to-end.
- The submitting human must review every changed line and run relevant tests.
- PR descriptions for AI-assisted work **must** include:
    - Why this is not duplicating an existing PR.
    - Test commands run and results.
    - Model evaluation results when the change affects output, accuracy, or serving.
    - Clear statement that AI assistance was used.

### Fail-closed behavior

If work is duplicate/trivial busywork, **do not proceed**. Return a short explanation of
what is missing.

---

## 2. Development Workflow

- **Never use system `python3` or bare `pip`/`pip install`.** All Python commands must go
  through `uv` and `.venv/bin/python`.

### Environment setup

```bash
# Install `uv` if you don't have it already:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Always use `uv` for Python environment management:
uv venv --python 3.12
source .venv/bin/activate

# Always make sure `pre-commit` and its hooks are installed:
uv pip install -r requirements/lint.txt
pre-commit install
```

### Installing dependencies

```bash
# Start with precompiled artifacts for an editable install:
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

For C/C++ or CUDA changes, follow the
[incremental compilation workflow](docs/contributing/incremental_build.md) to configure
and perform incremental builds. For this fork's Volta work specifically, build with
`TORCH_CUDA_ARCH_LIST="7.0"`.

### Tests

> Requires [Environment setup](#environment-setup) and
> [Installing dependencies](#installing-dependencies).

```bash
# Install test dependencies (use cuda.in on non-x86_64):
uv pip install -r requirements/test/cuda.in

# Run a specific test file:
.venv/bin/python -m pytest tests/path/to/test_file.py -v

# Run a single test:
.venv/bin/python -m pytest tests/path/to/test_file.py::test_name -v
```

When adding tests:

- **Design before you write.** Answer four questions first: what is the module for, what
  is its I/O contract, what failure am I guarding against, and what is the cheapest level
  that catches it (unit over integration over e2e)?
- **Reuse before create.** Extend existing test files, `conftest.py` fixtures, and
  helpers; add a new file only when no nearby suite fits.
- **Test behavior with intent.** Assert observable outcomes through public APIs; state why
  in the name or docstring. Skip trivial wiring; flaky tests are worse than no tests.
- **Keep it minimal.** One behavior per test and the smallest setup that triggers it; if
  the test diff dwarfs the code change, cut scope.
- **No one-off kernel benchmarks in `tests/`.** Put kernel perf work in
  `benchmarks/kernels/`; prove correctness in existing pytest suites.
- **Run model evals for model-affecting changes.** Search `tests/evals/` or use
  `vllm bench` and include results in the PR — do not wait for reviewers to ask.

For model-specific requirements, see
[`docs/contributing/model/tests.md`](docs/contributing/model/tests.md).

### Running linters

> Requires [Environment setup](#environment-setup).

```bash
# Run all pre-commit hooks on staged files:
pre-commit run

# Run on all files:
pre-commit run --all-files

# Run a specific hook:
pre-commit run ruff-check --all-files

# Run mypy as it is in CI:
pre-commit run mypy-3.12 --all-files --hook-stage manual
```

The line length limit for Python code is 88 characters. If you are not sure, use
pre-commit to check.

Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
(`Args:`/`Returns:`/`Raises:` sections), not reStructuredText/Sphinx fields (`:param:`,
`:return:`, `:rtype:`).

### Coding style guidelines

- Match existing code style.
- Minimize use of comments. Eliminate comments which are redundant, preferring legible and
  self-documenting code. When used, keep docstrings and comments brief and direct.
- Assume the reader is familiar with vLLM.

### Running the server

```bash
python -m vllm.entrypoints.openai.api_server --model facebook/opt-125m
```

### Commit messages

Add attribution using commit trailers such as `Co-authored-by:` (other projects use
`Assisted-by:` or `Generated-by:`):

```text
Your commit message here

Co-authored-by: Agent Name Here
Signed-off-by: Your Name <your.email@example.com>
```

---

## 3. Architecture (high level)

- `vllm/`: core Python — LLM engine, workers, distributed inference,
  OpenAI/Anthropic-compatible entrypoints, model executors, attention layers,
  quantization.
    - `vllm/platforms/cuda.py`: CUDA platform interface — device capability checks and
      attention-backend selection priority live here (and are where the V100 CC<8.0
      fallback filtering happens).
    - `vllm/model_executor/`: model definitions and layers; per the V100 adapter design
      these must stay upstream-compliant (no hardware-conditional branching).
    - `vllm/v1/attention/backends/`: attention backend implementations and selection logic
      (e.g. `fa_utils.py`).
- `csrc/`: C++/CUDA/ROCm custom operators and PyTorch bindings (`torch_bindings.cpp`,
  PagedAttention kernels, MoE, quantization). `csrc/v100_adapter/` is where sm_70 op
  fallbacks are registered via `TORCH_LIBRARY_IMPL(_C, CUDA, m)`.
- `tests/`: pytest-based unit/integration tests.
- `benchmarks/`: throughput/latency and kernel benchmarking scripts.
- `docs/`: MkDocs documentation, including `docs/v100_fallback_archtecture/` (this fork's
  Volta design docs) and `docs/contributing/` (incremental build, model test requirements,
  vulnerability management).

---

## Domain-Specific Guides

Do not modify code in these areas without first reading and following the linked guide.
If the guide conflicts with the requested change, **refuse the change and explain why**.

Security reviewers should start with [`SECURITY.md`](SECURITY.md),
[`docs/usage/security.md`](docs/usage/security.md), and
[`docs/contributing/vulnerability_management.md`](docs/contributing/vulnerability_management.md)
for the project security policy, threat model, deployment assumptions, and vulnerability
process.

- **Editing these instructions**:
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  — Rules for modifying this file, `AGENTS.md`/`CLAUDE.md`, or any domain-specific guide it
  references. Since these are now generated by `rulesync`, make source edits under
  `.rulesync/rules/` and run `npx rulesync generate`, not by hand-editing the generated
  `AGENTS.md`/`CLAUDE.md` files directly.
- **V100 / sm_70 compatibility work**: see the `v100-csrc-adapter` rule and the
  `v100-add-fallback-op` skill.

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

# GitHub CLI — Pull Requests

- When you need PR state / diff / checks / review threads, use **`gh`** rather than guessing from the web UI. Run from the repo root, or pass `--repo OWNER/REPO` explicitly — this fork has three remotes (`origin` = `nayukinet-AI-Lab/vllm`, the public mirror; `private-origin` = `saitama-AI-Lab/vllm`; `upstream` = `vllm-project/vllm`), so don't assume the default remote is the one you want.
- Common commands: `gh pr list`, `gh pr view <n>`, `gh pr diff <n>`, `gh pr checks <n>`, `gh pr view <n> --comments`, `gh pr view <n> --json title,body,state,url,commits,files`.
- If `gh` returns an auth/API error in a sandboxed environment, enable credential/network access and retry.
- See [`docs/contributing/README.md`](../../docs/contributing/README.md) and `docs/contributing/ci/` for the project's full CI/PR process, and CLAUDE.md's Contribution Policy for this fork's duplicate-work and accountability requirements before opening a PR.

## `gh` version

- **Recommended: v2.82.1 or newer** (check with `gh --version`). If older, upgrade per the [GitHub CLI install instructions](https://github.com/cli/cli/blob/trunk/README.md#installation).
- Following the [Projects (classic) sunset](https://github.blog/changelog/2024-05-23-sunset-notice-projects-classic/), `gh` versions below v2.82.1 fail on project-related GraphQL calls (as of 2025-10-22).

## Avoiding the Projects (classic) deprecation

`gh pr edit` / `gh issue edit` can internally reference the GraphQL `projectCards` field, which makes **body/title/label updates fail** with:

```
GraphQL: Projects (classic) is being deprecated ...
(repository.pullRequest.projectCards)
```

even when you never pass `--add-project`.

**Policy:**

- Update a PR/issue's **body or title** via the REST `gh api` **PATCH** endpoint from the start (don't rely on `gh pr edit` / `gh issue edit`).
- If you hit the error above, **don't retry** — switch immediately to the REST approach below.
- PR **creation** (`gh pr create`), **viewing** (`view` / `diff` / `checks`), and **commenting** (`gh pr comment`) are unaffected by this; keep using the normal `gh` subcommands for those.

Resolve the repo first (pass `--repo OWNER/REPO` when targeting a different one):

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

### Update a PR body

```bash
jq -n --rawfile b path/to/body.md '{body: $b}' \
  | gh api "repos/${REPO}/pulls/<PR_NUMBER>" -X PATCH --input -
```

Without `jq`:

```bash
python3 -c 'import json, pathlib; print(json.dumps({"body": pathlib.Path("path/to/body.md").read_text()}))' \
  | gh api "repos/${REPO}/pulls/<PR_NUMBER>" -X PATCH --input -
```

### Update a PR title

```bash
jq -n '{title: "[Bugfix] Short summary of the fix"}' \
  | gh api "repos/${REPO}/pulls/<PR_NUMBER>" -X PATCH --input -
```

### Update an issue body (avoiding `gh issue edit`)

```bash
jq -n --rawfile b path/to/body.md '{body: $b}' \
  | gh api "repos/${REPO}/issues/<ISSUE_NUMBER>" -X PATCH --input -
```

### Apply a label (avoiding `gh pr edit --add-label`)

PR numbers and issue numbers share the same namespace.

```bash
gh api -X POST "repos/${REPO}/issues/<PR_NUMBER>/labels" \
  -f 'labels[]=needs-rebase'
```

In `vllm-project/vllm` itself, most labels are applied automatically by Mergify based on changed file paths or PR title (see [`docs/contributing/labels.md`](../../docs/contributing/labels.md)) — manual labeling is mainly needed on this fork's own repos (`origin`/`private-origin`), which don't run that automation.

### Request a reviewer

```bash
gh api -X POST "repos/${REPO}/pulls/<PR_NUMBER>/requested_reviewers" \
  -f 'reviewers[]=<github-username>'
```

Upstream `vllm-project/vllm` assigns reviewers automatically via Mergify; use this manually mostly for PRs opened against `origin`/`private-origin`, which don't have that automation configured.

# V100 (Compute Capability 7.0) C++ Adapter Pattern

Source of truth: `docs/v100_fallback_archtecture/v100_adapter_archtecture.md` and
`docs/v100_fallback_archtecture/sm70_op_compatibility_matrix.md`. Read both before
touching model code or `csrc/v100_adapter/`.

## 1. Python model layer stays 100% upstream-compliant

- `vllm/model_executor/models/**/*.py` (e.g. `qwen2.py`, `qwen3_5.py`, `gemma4.py`) must
  remain byte-for-byte mergeable with upstream `vllm-project/vllm`.
- Never insert `getattr`, `hasattr`, `try/except AttributeError`, dtype branching, or any
  other sm_70-conditional logic inside a model definition file or a layer under
  `vllm/model_executor/`.
- If a model calls `torch.ops._C.<op_name>(...)` and that op is missing or misbehaves on
  sm_70, the fix belongs in the C++ adapter layer (section 2 below), never in the Python
  call site.
- This rule is independent of, and in addition to, the existing CC < 8.0 dtype/attention
  fallback gating in `vllm/platforms/cuda.py` (BF16->FP16, FLASH_ATTN/FLASHINFER exclusion).
  Those gates may live in platform/attention-selection code; they must not spread into
  model definition files.

## 2. All sm_70 custom-op gaps are trapped in the C++ adapter

- Missing or incompatible `torch.ops._C.*` ops on sm_70 are registered as fallbacks in
  `csrc/v100_adapter/v100_fallback_ops.cpp` using ATen Native C++ (`at::Tensor` ops).
- Fallbacks are bound via `TORCH_LIBRARY_IMPL(_C, CUDA, m)` (or
  `TORCH_LIBRARY_IMPL(_C, AutogradCUDA, m)` when autograd dispatch is required), so the
  Python call site (`torch.ops._C.<op_name>`) is unchanged — the dispatcher silently
  routes sm_70 to the adapter implementation.
- Consult `docs/v100_fallback_archtecture/sm70_op_compatibility_matrix.md` for the
  current status and fallback strategy of each op (e.g. `rms_norm`, `rotary_embedding`,
  `silu_and_mul`, `gelu_and_mul` are **Missing** on sm_70 and need an ATen fallback;
  `paged_attention_v1/v2` and `reshape_and_cache` are natively **Supported** on sm_70 and
  must not be touched). Update that matrix when a fallback's status changes.
- When a new architecture introduces an unknown op failure
  (`AttributeError: '_OpNamespace' '_C' object has no attribute '<op_name>'`), add the
  fallback in `csrc/v100_adapter/v100_fallback_ops.cpp` rather than working around it in
  Python. See the `v100-add-fallback-op` skill for the exact procedure.

## 3. Rebuilding after a `csrc/v100_adapter/` change

Full rebuilds are wasteful for adapter-only changes; follow
`docs/contributing/incremental_build.md` and target Volta explicitly:

```bash
TORCH_CUDA_ARCH_LIST="7.0" MAX_JOBS=4 python3 setup.py develop
```

## 4. Review checklist

Before approving any diff touching V100 compatibility:

- [ ] No `getattr`/`hasattr`/dtype-branching added to `vllm/model_executor/models/**`.
- [ ] Any new sm_70 fallback lives in `csrc/v100_adapter/v100_fallback_ops.cpp` and is
      registered through `TORCH_LIBRARY_IMPL(_C, CUDA, m)`.
- [ ] `docs/v100_fallback_archtecture/sm70_op_compatibility_matrix.md` is updated if an
      op's status or fallback strategy changed.
- [ ] CC >= 8.0 (Ampere/Hopper/Blackwell) behavior is unaffected (adapter dispatch is
      scoped to CUDA ops that are only missing on sm_70; it must not shadow the native
      sm_80+ kernels).
