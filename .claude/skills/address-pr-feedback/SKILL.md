---
name: address-pr-feedback
description: |
  Read PR review feedback and act on it.
  Classify review comments into fix / push-back / question, then handle each.
  Takes a PR number (e.g. 1234) as its argument.
---
# Address PR Feedback

Address the review feedback on PR #$ARGUMENTS.

---

## Step 1: Read the review

1. Fetch the PR:
   ```bash
   gh pr view $ARGUMENTS --json title,body,reviews,comments
   ```
2. Fetch review comments:
   ```bash
   gh pr view $ARGUMENTS --comments
   ```
3. Fetch inline code comments (**always paginate to get all of them**):
   ```bash
   gh api --paginate \
     "repos/:owner/:repo/pulls/$ARGUMENTS/comments?per_page=100"
   ```
   - `gh api` defaults to 30 items per page, so PRs with more than 30 comments
     silently drop some unless you pass both `--paginate` and `per_page=100`
     (common on PRs with multiple reviewers or dense threads).
   - After fetching, check the count with `jq 'length'` and eyeball it
     against the actual comment count on the PR page.
4. Also fetch review threads (reply chains) to see what's already resolved
   and whether a reply already exists:
   ```bash
   gh api graphql -f query='
     query($owner:String!,$repo:String!,$pr:Int!) {
       repository(owner:$owner,name:$repo) {
         pullRequest(number:$pr) {
           reviewThreads(first:100) {
             nodes {
               id isResolved isOutdated
               comments(first:50) { nodes { id author{login} body } }
             }
           }
         }
       }
     }
   ' -F owner=:owner -F repo=:repo -F pr=$ARGUMENTS
   ```

---

**Comment IDs are needed for posting replies in Step 6 — keep each comment's
`id` (the `id` field from the `.../comments` response) alongside its
classification.**

---

## Step 2: Classify feedback

Sort each review comment into one of these buckets and show the list to the
user:

### Fix
A valid point. Address it with a code change.
- Bug reports
- Security issues
- Clear code-quality problems
- Requests for additional test coverage

### Push back
A technically unnecessary or inappropriate request. Explain the reasoning in
a PR comment.
- Stylistic preferences that conflict with this project's existing
  conventions (e.g. the bracketed-tag commit style in `git.md`, or the
  "no `getattr`/`hasattr` in model files" rule in `v100-csrc-adapter.md`)
- Optimization requests with no measurable performance impact
- Requests that fall outside the current PR's scope

### Question
Needs more information. Check with the user.
- Ambiguous spec interpretation
- Judgment calls on product/API-compatibility tradeoffs

**Present the classification to the user and get confirmation before acting
on it.**

> **Claude Code only (skip this for Codex CLI):**
> 1. **Switch to `/model sonnet` to reduce token usage.**
> 2. Use the pr-review-toolkit plugin's agents (comment-analyzer,
>    pr-test-analyzer, silent-failure-hunter, type-design-analyzer,
>    code-reviewer, code-simplifier) as needed to raise review quality.
> 3. **Switch back to `/model opus` once the review pass is done.**

---

## Step 3: Apply fixes

Address each "Fix" comment one at a time:

1. Open the file(s) involved and make the change.
2. Run the quality checks:
   ```bash
   source .venv/bin/activate
   pre-commit run --all-files
   .venv/bin/python -m pytest tests/path/to/relevant_test_file.py -v
   ```
   If the change touches V100/Volta code (`vllm/platforms/cuda.py`,
   `csrc/v100_adapter/`, model files under `vllm/model_executor/models/`),
   also run the review checklist in the `v100-csrc-adapter` rule before
   moving on.
3. Fix anything the checks flag.

---

## Updating the PR description (after addressing feedback)

When adding test results or a summary of changes to the PR body:

- **Don't use `gh pr edit`.** Use the REST PATCH approach from the
  **github-cli-pr** rule (applied project-wide across tools/memories) —
  `jq -n --rawfile` + `gh api repos/.../pulls/<n> -X PATCH`.
- Inline replies and top-level comments still use `gh api` (replies) /
  `gh pr comment` as usual.

---

## Step 4: Post push-back comments

For each comment classified as "Push back":

1. Lay out the technical reasoning.
2. Post it with `gh pr comment $ARGUMENTS --body "..."`.
3. Keep the tone polite and constructive.

---

## Step 5: Commit & push

1. Review changed files and stage them explicitly:
   ```bash
   git diff --name-only
   git add path/to/changed/file1 path/to/changed/file2
   ```
   **Don't use `git add -A`.**
2. Commit using this project's bracketed-tag convention (see
   [git.md](../../rules/git.md)) and include a DCO sign-off:
   ```bash
   git commit -s -m "[Bugfix] Address PR review feedback"
   ```
3. Push: `git push`

---

## Step 6: Reply to addressed comments

For each comment classified as "Fix", post a reply summarizing the fix
commit, so reviewers can track resolution per-comment.

1. Build the URL of the commit just pushed:
   ```bash
   COMMIT_SHA=$(git rev-parse HEAD)
   REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
   COMMIT_URL="https://github.com/${REPO}/pull/$ARGUMENTS/commits/${COMMIT_SHA}"
   ```

2. Reply to each "Fix" comment ID noted in Step 2:
   ```bash
   gh api -X POST \
     repos/:owner/:repo/pulls/$ARGUMENTS/comments/<COMMENT_ID>/replies \
     -f body="$(cat <<EOF
   Thanks for flagging this — addressed by <one or two sentence summary>.

   ${COMMIT_URL}
   EOF
   )"
   ```

3. Reply body structure:
   - Open with a brief thanks.
   - Summarize what changed and how, in one or two sentences.
   - Always attach the fix commit's URL.
   - Optionally add a one-line pointer to the relevant code (file + line).

4. **Only use this for inline review comments** (comments anchored to a
   path/line). Top-level PR comments (issue comments) don't support the
   replies API — use `gh pr comment` from Step 4 for those.

5. Once all replies are posted, report the count to the user.

---

## Final report

- Number of comments fixed (and how many got a reply posted)
- Number of comments pushed back on
- Items left open as questions
- Lint/test results
