---
paths:
  - '**/*'
---
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
