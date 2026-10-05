# sm_70 (V100) vs sm_75+ Custom Ops Compatibility Matrix

This document tracks all C++ custom operators exposed via `torch.ops._C` in vLLM, their minimum hardware requirements, and the C++ Adapter fallback strategy for Compute Capability 7.0 (sm_70).

---

## 1. Custom Ops Matrix & Fallback Strategy

| Custom Op Name (`torch.ops._C.*`) | Native Target Compute Capability | sm_70 Status | sm_70 Fallback Implementation (`csrc/v100_adapter/`) |
| :--- | :--- | :--- | :--- |
| `rms_norm` | sm_80+ | **Missing** | ATen C++ Native: `input * rsqrt(var + eps) * weight` |
| `fused_add_rms_norm` | sm_80+ | **Missing** | ATen C++ Native: `(x + residual)` norm & in-place update |
| `rotary_embedding` | sm_80+ | **Missing** | ATen C++ Native: Complex tensor rotation / Sin-Cos lookup |
| `silu_and_mul` | sm_80+ | **Missing** | ATen C++ Native: `at::silu(x1) * x2` |
| `silu_and_mul_with_clamp` | sm_80+ | **Missing** | ATen C++ Native: `at::clamp(at::silu(x1) * x2, min, max)` |
| `gelu_and_mul` | sm_80+ | **Missing** | ATen C++ Native: `at::gelu(x1) * x2` |
| `gelu_tanh_and_mul` | sm_80+ | **Missing** | ATen C++ Native: `at::gelu(x1, "tanh") * x2` |
| `get_cuda_view_from_cpu_tensor` | sm_80+ (UVA) | **Missing** | Standard CUDA Copy: `cpu_tensor.to(at::kCUDA, non_blocking=true)` |
| `topk_topp_sampler` (FlashInfer) | sm_80+ | **Unsupported** | Native PyTorch Sampler fallback |
| `paged_attention_v1/v2` | sm_70+ | **Supported** | Native CUDA Kernel (`sm_70` compiled) |
| `reshape_and_cache` | sm_70+ | **Supported** | Native CUDA Kernel (`sm_70` compiled) |

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
