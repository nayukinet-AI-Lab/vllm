---
name: v100-add-fallback-op
description: >-
  Register an ATen-native C++ fallback in csrc/v100_adapter/ for a torch.ops._C
  custom op that is missing or unsupported on NVIDIA V100 (sm_70 / Compute
  Capability 7.0), bind it via TORCH_LIBRARY_IMPL, and rebuild for Volta. Use
  this when a model run on V100 crashes with
  "AttributeError: '_OpNamespace' '_C' object has no attribute '<op_name>'" or
  when adding support for a new model architecture on this fork.
targets: ["*"]
---

# Add a V100 (sm_70) fallback for a missing `torch.ops._C` custom op

Follow this procedure whenever a model fails on a V100 (Compute Capability 7.0 / sm_70)
GPU because a C++ custom op does not exist or does not support sm_70. Background and
rationale: `docs/v100_fallback_archtecture/v100_adapter_archtecture.md` and
`docs/v100_fallback_archtecture/sm70_op_compatibility_matrix.md`.

**Hard constraint:** never fix this by editing `vllm/model_executor/models/**/*.py` or by
adding `getattr`/`hasattr`/dtype-branching in Python. The fix always lives in
`csrc/v100_adapter/`. See the `v100-csrc-adapter` rule.

## Steps

1. **Identify the missing operator from the stack trace.**
   Look for:
   ```
   AttributeError: '_OpNamespace' '_C' object has no attribute '<op_name>'
   ```
   `<op_name>` is the op to add a fallback for.

2. **Check whether it is already tracked.**
   Open `docs/v100_fallback_archtecture/sm70_op_compatibility_matrix.md` and search for
   `<op_name>`. If it is listed as **Supported** on sm_70, the bug is elsewhere (do not
   add a redundant fallback — investigate the real failure instead). If it is **Missing**
   or **Unsupported** and already has a documented strategy, implement that strategy. If
   it is not listed yet, add a new row describing the op, its native target compute
   capability, its sm_70 status (`Missing`), and the fallback implementation you are about
   to write.

   Useful greps:
   ```bash
   grep -rn "TORCH_LIBRARY_IMPL(_C" csrc/
   grep -rn "m.impl(" csrc/
   grep -rn "torch.ops._C.<op_name>" vllm/
   ```

3. **Implement the ATen-native fallback in C++.**
   Open `csrc/v100_adapter/v100_fallback_ops.cpp` (create it, following the existing
   `csrc/` binding conventions, if this is the first fallback in the file) and implement
   the op using plain `at::Tensor` operations — no sm_80+ intrinsics, no Tensor Core
   PTX. Favor existing ATen primitives that already have sm_70-compatible CUDA kernels,
   e.g.:
   - `rms_norm` -> `input * at::rsqrt(var + eps) * weight`
   - `fused_add_rms_norm` -> `(x + residual)` followed by the norm, updating in place
   - `rotary_embedding` -> complex-tensor rotation or sin/cos lookup
   - `silu_and_mul` -> `at::silu(x1) * x2`
   - `silu_and_mul_with_clamp` -> `at::clamp(at::silu(x1) * x2, min, max)`
   - `gelu_and_mul` -> `at::gelu(x1) * x2`
   - `gelu_tanh_and_mul` -> `at::gelu(x1, "tanh") * x2`
   - `get_cuda_view_from_cpu_tensor` -> `cpu_tensor.to(at::kCUDA, /*non_blocking=*/true)`

   Match the existing op's signature exactly (same arg types/order, same output
   tensor(s)/in-place semantics) so the Python call site needs no change.

4. **Register the fallback for CUDA dispatch.**
   Bind the implementation under the `_C` library, scoped to the CUDA dispatch key (and
   `AutogradCUDA` only if the op must support autograd):
   ```cpp
   TORCH_LIBRARY_IMPL(_C, CUDA, m) {
     m.impl("<op_name>", &v100_adapter::<op_name>_fallback);
   }
   ```
   Do not register under a dispatch key broader than `CUDA`/`AutogradCUDA` — the fallback
   must not shadow the native sm_80+ kernel for the same op name on Ampere/Hopper/
   Blackwell. If the registration needs to be sm_70-specific rather than CUDA-wide,
   gate it at build time (`TORCH_CUDA_ARCH_LIST` / `__CUDA_ARCH__`) rather than at
   dispatch time.

5. **Rebuild for Volta.**
   ```bash
   TORCH_CUDA_ARCH_LIST="7.0" MAX_JOBS=4 python3 setup.py develop
   ```
   For iterative work, prefer the incremental build flow in
   `docs/contributing/incremental_build.md` instead of a full rebuild when only
   `csrc/v100_adapter/` changed.

6. **Verify.**
   - Re-run the failing model path and confirm the `AttributeError` is gone and output is
     numerically sane (compare against a CC >= 8.0 run of the same model/prompt if one is
     available).
   - Run/extend the relevant test under `tests/` to cover the fallback — correctness only,
     per this repo's test policy (kernel perf work belongs in `benchmarks/kernels/`).
   - Confirm CC >= 8.0 behavior is unchanged (the fallback is CUDA-dispatch-scoped and
     must not run on Ampere/Hopper/Blackwell).

7. **Update tracking docs.**
   Update the row for `<op_name>` in
   `docs/v100_fallback_archtecture/sm70_op_compatibility_matrix.md` to reflect the final
   fallback strategy actually implemented.
