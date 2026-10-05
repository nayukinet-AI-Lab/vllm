Please also reference the following rules as needed. The list below is provided in TOON format, and `@` stands for the project root directory.

rules[4]:
  - path: @.claude/rules/git-worktree.md
  - path: @.claude/rules/git.md
  - path: @.claude/rules/github-cli-pr.md
  - path: @.claude/rules/v100-csrc-adapter.md
    applyTo[3]: vllm/model_executor/models/**/*.py,csrc/v100_adapter/**/*,docs/v100_fallback_archtecture/**/*

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
