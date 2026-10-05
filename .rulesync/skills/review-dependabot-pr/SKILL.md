---
name: review-dependabot-pr
description: |
  Risk-review a Dependabot dependency-update PR from multiple angles.
  Reports on breaking changes and codebase impact. Uses gh pr checkout to
  fetch the PR branch for reproducibility.
  Does not modify application code or commit/push (different role from the
  address-pr-feedback skill).
  Takes a PR number (e.g. 1234) as its argument. Posting a comment on the PR
  requires explicit instruction in the same prompt (e.g. "comment on the PR").
---
# Dependabot PR Risk Review

Review PR **#$ARGUMENTS**. In the command examples below, substitute
`$ARGUMENTS` with the PR number.

## Input (posting a PR comment)

- By default, only report back in chat.
- Only post a comment to the GitHub PR when the **same prompt** includes one
  of: `post comment` / `comment on pr` / `post to pr`.

---

## Step 0: Confirm this is a Dependabot PR

```bash
gh pr view $ARGUMENTS --json author,title,body,url,baseRefName,headRefName,files
```

- Check that `author.login` looks like Dependabot (contains `dependabot`,
  e.g. `dependabot[bot]`).
- If not, **stop** and ask the user whether to continue as a generic
  dependency-PR review instead.

---

## Step 1: Check the working tree (required before checkout)

```bash
git status
```

- If there are uncommitted changes, **do not run `gh pr checkout`**. Ask the
  user to stash/commit or work from a separate clone, then resume once clean.

---

## Step 2: Check out the PR branch (required)

Always check out the actual PR head for reproducibility.

```bash
PREV_BRANCH=$(git branch --show-current)
gh pr checkout $ARGUMENTS
git branch --show-current
gh pr view $ARGUMENTS --json headRefName -q .headRefName
```

**Source of truth:** the on-disk lockfile/manifest files after checkout
(`requirements/*.txt`, `pyproject.toml`, `.github/workflows/*.yml`, etc.).

---

## Step 3: Diff and CI

```bash
BASE=$(gh pr view $ARGUMENTS --json baseRefName -q .baseRefName)
git fetch origin "$BASE" 2>/dev/null || true
git merge-base HEAD "origin/$BASE" | xargs -I{} git diff {}...HEAD
```

Also useful:

```bash
gh pr diff $ARGUMENTS
gh pr checks $ARGUMENTS
gh pr view $ARGUMENTS --json title,body,commits,files,statusCheckRollup
```

Use [`.github/dependabot.yml`](../../.github/dependabot.yml) as the map of
what's covered — this repo only configures two ecosystems: `github-actions`
(root) and `pip` (root, `requirements/*.txt` + `pyproject.toml`). Note that
`dependabot.yml` explicitly **ignores** `torch`, `torchvision`, `xformers`,
`lm-format-enforcer`, `compressed-tensors`, `ray[cgraph]`, and `lm-eval` —
any PR that *does* touch one of these despite the ignore list, or touches
`requirements/cuda.txt` / `requirements/build/cuda.txt` /
`pyproject.toml`'s `[build-system]` torch pin, needs extra scrutiny on this
fork specifically, since those pins are manually maintained to keep the
V100/Volta build working (see CLAUDE.md "Development Workflow" and the
`v100-torch-version-matrix` constraints) and a routine bump can silently
break sm_70 support. Run `gh` from the repo root (or pass `--repo
OWNER/REPO`).

---

## Step 4: Risk report (chat)

Report using the following **heading order**. Flag weak evidence explicitly.

### 1. Summary of the update

Changed paths, ecosystem, size of the semver jump, whether it's a grouped
update.

### 2. Breaking changes / compatibility

Major-version bumps, removed/deprecated APIs, Python version requirements,
CUDA/driver version requirements, transitive dependency conflicts (e.g. with
pinned `torch`/`numpy`/`numba` versions — see CLAUDE.md's documented
numpy/numba conflict history). For grouped updates, call out **the
highest-risk package** explicitly.

### 3. Security / supply chain

Only make concrete claims when backed by evidence (advisory, release notes).
Otherwise mark as unknown.

### 4. CI/CD

Results from `gh pr checks`. For `github-actions` updates, name the affected
workflow(s) and their blast radius.

### 5. Codebase impact

For each updated package, grep for actual usage (imports, config, tool
integration) and classify as **likely used / not used / needs verification**
(e.g. `torch`, `triton`, `transformers`, `pytest` plugins, `ruff`, `mypy`,
`fastapi`/`starlette`/`pydantic` in the API server, `ray`, `boto3`).

### 6. Residual risk / recommended action

Additional verification to do before merging (tests, manual check, a V100
smoke test if the pin affects the Volta build).

---

## Step 5: Comment on GitHub (opt-in only)

Skip this step if the **same prompt** doesn't contain one of the **Input
(posting a PR comment)** keywords.

If it does:

1. Write the report body to `tmp/review-dependabot-pr-comment.md` and post it
   (the file can be deleted afterward):

   ```bash
   gh pr comment $ARGUMENTS --body-file tmp/review-dependabot-pr-comment.md
   ```

2. Include the posted comment's URL in the final report.

---

## When the PR number isn't known

```bash
gh pr list --state open --json number,title,author --jq '.[] | select((.author.login // "") | test("dependabot"; "i")) | "\(.number)\t\(.title)"'
```

---

## After the review

- Optional: `git checkout "$PREV_BRANCH"` (if `PREV_BRANCH` is valid). If you
  don't switch back, note the current branch in the final report.

---

## On error

- `gh` auth/API error → retry, or point the user at `gh auth login`.
- Dirty working tree → don't check out the PR.
- `gh pr checkout` failure → report stderr, consider retrying after
  `git fetch`.

---

## Final report

- Dependabot confirmation result
- Current branch after checkout
- PR URL (`gh pr view $ARGUMENTS --json url -q .url`)
- Risk level (low / medium / high) and why
- Link to the posted PR comment, if one was posted
