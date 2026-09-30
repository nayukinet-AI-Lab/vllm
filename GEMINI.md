# vLLM Project Instructions

## Project Overview

**vLLM** is a high-throughput and memory-efficient LLM inference and serving engine. 
It provides state-of-the-art serving throughput, efficient memory management with **PagedAttention**, continuous batching, chunked prefill, prefix caching, and extensive quantization/hardware support.

- **Languages:** Python, C++, CUDA/HIP, Rust, CMake
- **Frameworks:** PyTorch, Triton, Hugging Face Transformers
- **Key Architecture Components:**
  - `vllm/`: Core Python codebase including the LLM engine, workers, distributed inference, entrypoints (OpenAI-compatible API server, CLI), model executors, attention layers, and quantization support.
  - `csrc/`: C++ and CUDA/ROCm custom operators, extensions, and PyTorch bindings (`torch_bindings.cpp`, flash attention, PagedAttention cache kernels, MoE, quantization).
  - `tests/`: Comprehensive test suite (`pytest`-based unit and integration tests).
  - `benchmarks/`: Performance benchmarking tools and throughput/latency scripts.
  - `docs/`: MkDocs documentation and guides.

---

## Building and Running

### Installation & Environment Setup
It is recommended to use `uv` or `pip` for installation in editable mode during development:

```bash
# Install in editable mode with development dependencies
pip install -e .
# Or using uv
uv pip install -e .
```

### Running the API Server
vLLM includes an OpenAI-compatible API server:

```bash
python -m vllm.entrypoints.openai.api_server --model facebook/opt-125m
```

### Running Tests
Tests are implemented using `pytest`. Run tests with:

```bash
pytest tests/
```

---

## Development Conventions & Standards

- **Code Style & Linting:** 
  - The project uses `ruff` for linting and code formatting, configured in `pyproject.toml`.
  - Run lint checks via `ruff check .`.
- **Pre-commit Hooks:** 
  - Install pre-commit hooks before contributing: `pre-commit install`.
- **C++ & CUDA Extensions:**
  - Custom C++/CUDA operators are built via `setup.py`, `CMakeLists.txt`, and setuptools-rust/setuptools-cpp. Ensure appropriate CUDA toolkit versions are installed when compiling GPU-accelerated code.
- **Testing Requirements:**
  - Always add or update unit tests when implementing new features or fixing bugs in `tests/`.

---

# 4. Target Specifics: NVIDIA Tesla V100 (Volta / CC 7.0) Patch Requirements

## 4.1 Hardware Constraints & Goals
- **Target Hardware:** NVIDIA Tesla V100 (Volta architecture / Compute Capability 7.0).
- **Key Limitations:** Lacks native hardware support for `bfloat16` (BF16) and `FlashAttention v1/v2/v3`.
- **Primary Goal:** Modify vLLM code to allow executing modern LLM architectures (e.g., Gemma 2, Qwen 2.5 series) on CC 7.0 devices without runtime panics or hardcoded BF16 crashes.

## 4.2 Implementation Rules for Gemini CLI / Coding Agents
1. **Device Capability Check:**
   - Always verify CUDA device capability via `torch.cuda.get_device_capability()[0] < 8` when selecting attention backends or default model dtypes.
2. **Dtype Fallback (BF16 -> FP16):**
   - Automatically fall back from `torch.bfloat16` to `torch.float16` for model weights, layer activations, and KV cache when Compute Capability < 8.0.
3. **Attention Backend Fallback:**
   - Bypass `FLASH_ATTN` selection when running on CC < 8.0 devices.
   - Force automatic fallback to `XFORMERS` or vLLM's native `PAGED_ATTN` (Triton / Standard PyTorch Attention).
4. **Triton Kernel Type Safety:**
   - Inspect Triton kernels (`.py` files containing `@triton.jit` or `tl.bfloat16`) to ensure `tl.bfloat16` references safely degrade or map to `tl.float16` for Volta GPUs.
5. **Non-Destructive Conditionals:**
   - ALWAYS wrap Volta/V100 overrides with explicit `if device_capability < (8, 0):` checks so performance on Ampere/Hopper/Blackwell (CC >= 8.0) devices remains completely untouched.

## 4.3 Git Workspace & Branch Context
- **Active Working Branch:** `feature/v100-fp16-patch`
- **Origin Repository:** `saitama-AI-Lab/vllm`
- **Upstream Repository:** `vllm-project/vllm`