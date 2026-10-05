# V100 (Compute Capability 7.0) C++ Adapter Architecture Design

## 1. Overview & Purpose
This document outlines the architectural design for supporting modern LLMs (e.g., Qwen2.5/3.5, Gemma-4) on NVIDIA Tesla V100 GPUs (sm_70 / Compute Capability 7.0) within the vLLM execution environment.

Due to the absence of Ampere/Hopper-specific hardware features (e.g., FlashAttention-2/3, FP8/FP4 Tensor Cores, sm_80+ PTX instructions) in sm_70, vLLM's native C++ extensions (`vllm._C`) fail to bind or execute.

To avoid repetitive, brittle Python-level monkey patching (`vllm/model_executor/` and `vllm/kernels/`), this repository adopts an **Invasive-Free C++ Adapter Pattern**. All hardware incompatibilities and missing custom operators are encapsulated within the C++ layer (`csrc/`).

---

## 2. Core Architectural Principles

### 2.1 Open-Closed Principle (OCP)
- **Model Layer (Python):** Python model definition files (e.g., `qwen2.py`, `qwen3_5.py`, `gemma4.py`) must remain **100% upstream compliant**. Do NOT insert `getattr`, `hasattr`, or Python fallback logic inside model definition classes.
- **Dispatch Layer (C++):** All fallback mechanisms for sm_70 must be trapped inside `csrc/` using PyTorch's OpNamespace registration system (`TORCH_LIBRARY_IMPL`).

### 2.2 Layer Responsibilities

```
+-------------------------------------------------------------+
| Python Layer (vllm/model_executor/models/)                 |
|  - Qwen2.5 / Qwen3.5 / Gemma-4 Model Definitions            |
|  - Directly calls torch.ops._C.<op_name>(...)                |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| PyTorch Operator Registry (TORCH_LIBRARY_IMPL / torch::ops) |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| C++ Adapter Layer (csrc/v100_adapter/)                      |
|  - Traps missing sm_80+ custom ops                          |
|  - Dispatches to ATen Native C++ / PyTorch C++ Fallbacks     |
+-------------------------------------------------------------+
```

---

## 3. Extension & Maintenance Guidelines for AI Agents (Gemini / Claude / Sonnet)

When a new architecture (e.g., Qwen3.5, Gemma-4) introduces unknown C++ custom ops or instructions not supported on sm_70, follow these steps:

1. **Do NOT modify Python model definitions.**
2. **Identify the missing operator:** Check the stack trace for `AttributeError: '_OpNamespace' '_C' object has no attribute '<op_name>'`.
3. **Register the fallback in C++:**
   - Open `csrc/v100_adapter/v100_fallback_ops.cpp`.
   - Implement an ATen C++ fallback function using `at::Tensor` operations.
   - Register it under `TORCH_LIBRARY_IMPL(_C, CUDA, m)` or `TORCH_LIBRARY_IMPL(_C, AutogradCUDA, m)`.
4. **Rebuild the C++ extension:**
   ```bash
   TORCH_CUDA_ARCH_LIST="7.0" MAX_JOBS=4 python3 setup.py develop
   ```
   