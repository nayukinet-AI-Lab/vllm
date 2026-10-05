---
root: false
targets: ["*"]
description: "V100 (sm_70) C++ Adapter Pattern: keep Python model code upstream-compliant and trap missing torch.ops._C custom ops in csrc/v100_adapter/ via TORCH_LIBRARY_IMPL"
globs: ["vllm/model_executor/models/**/*.py", "csrc/v100_adapter/**/*", "docs/v100_fallback_archtecture/**/*"]
---

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
