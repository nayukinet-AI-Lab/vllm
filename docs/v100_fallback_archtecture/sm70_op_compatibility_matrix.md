# sm_70 (V100) vs sm_75+ Custom Ops Compatibility Matrix

This document tracks all C++ custom operators exposed via `torch.ops._C` in vLLM, their minimum hardware requirements, and the C++ Adapter fallback strategy for Compute Capability 7.0 (sm_70).

---

## 1. Custom Ops Matrix & Fallback Strategy

| Custom Op Name (`torch.ops._C.*`) | Native Target Compute Capability | sm_70 Status | Fallback | sm_70 Fallback Implementation (`csrc/v100_adapter/`) |
| :--- | :--- | :--- | :--- | :--- |
| `rms_norm` | sm_80+ | **Missing** | ✅ Implemented | ATen C++ Native: `input * rsqrt(mean(input^2) + eps) * weight` (fp32 intermediate) |
| `fused_add_rms_norm` | sm_80+ | **Missing** | ✅ Implemented | ATen C++ Native: `residual := x + residual`; `input := rms_norm(residual) * weight` (in-place) |
| `silu_and_mul` | sm_80+ | **Missing** | ✅ Implemented | ATen C++ Native: `at::silu(gate) * up` |
| `silu_and_mul_with_clamp` | sm_80+ | **Missing** | ✅ Implemented | ATen C++ Native: `(gate.clamp_max(limit) * sigmoid(alpha * gate)) * (up.clamp(±limit) + beta)` |
| `gelu_and_mul` | sm_80+ | **Missing** | ✅ Implemented | ATen C++ Native: `at::gelu(gate) * up` |
| `gelu_tanh_and_mul` | sm_80+ | **Missing** | ✅ Implemented | ATen C++ Native: `at::gelu(gate, "tanh") * up` |
| `rotary_embedding` | sm_80+ | **Missing** | ⏳ Planned | ATen C++ Native: Complex tensor rotation / Sin-Cos lookup |
| `get_cuda_view_from_cpu_tensor` | sm_80+ (UVA) | **Missing** | ⏳ Planned | Standard CUDA Copy: `cpu_tensor.to(at::kCUDA, non_blocking=true)` (note: copy, not a zero-copy view) |
| `topk_topp_sampler` (FlashInfer) | sm_80+ | **Unsupported** | ⏳ Planned | Native PyTorch Sampler fallback |
| `paged_attention_v1/v2` | sm_70+ | **Supported** | — (native) | Native CUDA Kernel (`sm_70` compiled) — must not be touched |
| `reshape_and_cache` | sm_70+ | **Supported** | — (native) | Native CUDA Kernel (`sm_70` compiled) — must not be touched |

> **Implemented fallbacks** live in `csrc/v100_adapter/v100_fallback_ops.cpp`,
> registered via `TORCH_LIBRARY_IMPL(_C, CUDA, m)` under the `#ifdef
> VLLM_V100_ADAPTER` guard. CMake defines `VLLM_V100_ADAPTER` only for CUDA
> builds whose target architectures are all CC < 8.0; the matching native CUDA
> registrations in `csrc/libtorch_stable/torch_bindings.cpp` are then gated out
> with `#ifndef VLLM_V100_ADAPTER` so exactly one CUDA kernel is registered per
> op.
>
> The `_C_stable_libtorch` / `_moe_C_stable_libtorch` extension targets (disabled
> during earlier trial-and-error) are re-enabled in both `CMakeLists.txt` and
> `setup.py` (`ext_modules`). A source build for Volta is therefore:
>
> ```bash
> TORCH_CUDA_ARCH_LIST="7.0" MAX_JOBS=4 python setup.py develop
> ```
>
> sm_80+-only kernels are kept out of a Volta build automatically: the heavy
> CUTLASS/machete/marlin/fp4/w4a8 families are arch-gated via
> `cuda_archs_loose_intersection`, and the sm_80+ external projects (flash-attn,
> deepgemm, fmha_sm100, flashmla, flashkda, qutlass, tml_fa4) are gated by
> `_HAS_AMPERE_OR_NEWER`. Any unconditionally-listed base source that still fails
> to compile for sm_70 can be excluded via the `VLLM_SM70_EXCLUDED_SRCS` /
> `VLLM_SM70_MOE_EXCLUDED_SRCS` hooks in `CMakeLists.txt` (pair each exclusion
> with an `#ifndef VLLM_V100_ADAPTER` guard on its central registration).

---

## 2. Automated Grep Guidelines for AI Agents

To inspect the codebase for newly added or missing C++ custom ops across vLLM versions:

### 2.1 Find all C++ Op Registrations in `csrc/`
Run the following command to list all operators currently exposed to `torch.ops._C`:
```bash
grep -rn "TORCH_LIBRARY_IMPL(_C" csrc/
grep -rn "m.impl(" csrc/
```

### 2.2 Find all Python-side C++ Op Invocations
Run the following command to find where Python expects `_C` ops:
```bash
grep -rn "torch.ops._C." vllm/
```

### 2.3 Check Available Ops at Runtime in Python
Run this snippet inside your environment to inspect bound operators:
```python
import torch
import vllm._C

ops = [attr for attr in dir(torch.ops._C) if not attr.startswith("_")]
print("Currently registered vllm._C ops:", ops)
```
