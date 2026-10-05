# V100 (sm_70) C++ adapter tests

Fork-local test suite for the Volta compatibility work. It verifies the
ATen-native fallbacks in `csrc/v100_adapter/v100_fallback_ops.cpp` — the ops that
are missing on NVIDIA Tesla V100 (Compute Capability 7.0) and trapped in the C++
adapter layer (see `docs/v100_fallback_archtecture/` and the `v100-csrc-adapter`
rule).

## Why it lives in `test_v100/` (not `tests/`)

Kept separate from the upstream `tests/` tree so that merging from
`vllm-project/vllm` never conflicts with it, and so this fork's V100 patch can be
re-verified in one command after every upstream pull.

## What it does

- Selects a pre-Ampere (CC < 8.0) CUDA device, preferring a real V100.
- JIT-compiles the **real** adapter source with `-DVLLM_V100_ADAPTER` via
  `torch.utils.cpp_extension` (plus a standalone `_C` schema file), so **no full
  vLLM build is required**. The adapter is pure C++ calling ATen ops whose CUDA
  kernels already ship in the installed libtorch (cu124 supports sm_70).
- Calls each `torch.ops._C.<op>` and compares against an fp32 PyTorch reference.
- If a built `vllm._C` already exposes the ops (a real sm_70 build), it uses them
  directly instead of JIT-compiling.

The suite **skips cleanly** when no CC < 8.0 GPU is present, so it is safe to run
in CI on any hardware.

## Running

```bash
# From the repo root, in the project venv (per CLAUDE.md: no system python/pip):
.venv/bin/python -m pytest test_v100/ -v

# Show the JIT compile command / ninja output:
V100_ADAPTER_VERBOSE=1 .venv/bin/python -m pytest test_v100/ -v -s
```

Requirements: `torch`, `pytest`, and `ninja` (the suite auto-shims
`python -m ninja` onto `PATH` if the `ninja` executable is not found). A
C++17 host compiler is needed for the JIT compile.

## Coverage

| op | checks |
| :-- | :-- |
| `rms_norm` | with and without `weight` |
| `fused_add_rms_norm` | in-place `residual` (pre-norm sum) and `input` (normalized) |
| `silu_and_mul` | `silu(gate) * up` |
| `gelu_and_mul` | `gelu(gate, 'none') * up` |
| `gelu_tanh_and_mul` | `gelu(gate, 'tanh') * up` |
| `silu_and_mul_with_clamp` | default and non-default `alpha`/`beta` clamping |

When adding a new fallback op to the adapter, add its schema to
`_c_op_schemas.cpp`, its name to `ADAPTER_OPS` in `conftest.py`, and a focused
test here.
