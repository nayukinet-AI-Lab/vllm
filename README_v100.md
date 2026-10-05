# V100 (sm_70) C++ Adapter Layer — Implementation Notes

This document describes the **C++ adapter layer** that makes vLLM run on
NVIDIA Tesla V100 (Volta / Compute Capability 7.0 / sm_70) GPUs, and how it is
wired into the build system.

**Status (updated 2026-10-05)**: the adapter is built and loaded as part of
the real `_C_stable_libtorch` / `_moe_C_stable_libtorch` extensions, and has
been verified end to end on real V100 hardware across three architecture
families: `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen3.5-0.8B`, and
`google/gemma-4-e2b-it` (see §6, §7). Before this, the stable extension was
always skip-gated on `torch 2.6`, so the adapter had never actually been
compiled into a real build.

Related documents:

- Design rationale: `docs/v100_fallback_archtecture/v100_adapter_archtecture.md`
- Op compatibility matrix: `docs/v100_fallback_archtecture/sm70_op_compatibility_matrix.md`
- Operating rule: `.claude/rules/v100-csrc-adapter.md`
- Fallback-addition procedure: `.claude/skills/v100-add-fallback-op/SKILL.md`

---

## 1. Background and the problem

V100 (sm_70) lacks the hardware features that Ampere-and-later (CC >= 8.0)
GPUs have (FlashAttention-2/3, FP8/FP4 Tensor Cores, sm_80+ PTX instructions,
hardware BF16, etc.). Because of this, some of vLLM's native C++ custom ops
don't work on sm_70, and model execution fails with errors like:

```
AttributeError: '_OpNamespace' '_C' object has no attribute '<op_name>'
```

Working around this by editing the Python model definitions (monkey-patching)
is a maintainability dead end, so this fork uses an **invasive-free C++
adapter pattern** instead:

- The Python model layer (`vllm/model_executor/models/**`) stays 100%
  upstream-compliant and is never touched.
- Every op that's missing or unsupported on sm_70 is trapped entirely in the
  C++ layer (`csrc/v100_adapter/`) via `STABLE_TORCH_LIBRARY_IMPL` (libtorch
  stable ABI, see §2.2) and falls back to an ATen-native implementation.

---

## 2. Design decisions

### 2.1 The double-registration problem and the build-time gate

The target ops (`rms_norm`, `silu_and_mul`, etc.) are **already registered**
for the CUDA dispatch key inside
`csrc/libtorch_stable/torch_bindings.cpp`'s
`STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops)` block (and the native
`layernorm_kernels.cu` etc. have no sm_80-specific dependency, so they do
compile for sm_70 too).

So if the adapter naively registered the same op under the same dispatch key
(`_C::rms_norm` / `CUDA`), PyTorch's dispatcher would raise a **double
registration error** and the extension would fail to load — breaking every
GPU, not just sm_70.

The fix is a **build-time gate** (the `VLLM_V100_ADAPTER` macro):

- CMake defines `VLLM_V100_ADAPTER` only when **every** target CUDA
  architecture is CC < 8.0.
- The adapter's registration is wrapped in `#ifdef VLLM_V100_ADAPTER`.
- The matching native registration is wrapped in `#ifndef VLLM_V100_ADAPTER`.

So on an sm_70-only build, the native registration is dropped and the ATen
fallback is registered instead; on a CC >= 8.0 build, the native registration
stays and the adapter never even compiles. **Exactly one CUDA kernel per op**
is registered either way.

### 2.2 Implementing against the libtorch stable ABI (rewritten 2026-10-05)

`csrc/v100_adapter/v100_fallback_ops.cpp` compiles as part of the
`_C_stable_libtorch` target, which is built with
`-DPy_LIMITED_API=3 -DTORCH_TARGET_VERSION=...`. That means legacy ATen
headers (`torch/all.h`, `torch/library.h`, `at::Tensor`) are unusable —
they `#error` out under `TORCH_TARGET_VERSION`. The original implementation
was written against the legacy API, but since the stable extension was always
skip-gated on `torch 2.6`, this incompatibility had never actually been
caught until the first real `_C_stable_libtorch` build was attempted.

The implementation was rewritten from scratch against
`torch::stable::Tensor` / `torch/csrc/stable/library.h`
(`STABLE_TORCH_LIBRARY_IMPL`, `TORCH_BOX`). ATen ops are called through three
mechanisms, in order of preference:

1. Pre-made wrappers in `torch::stable::ops.h` (`narrow`, `copy_`, `to`,
   `full`, `sum`, etc.).
2. Direct dispatcher-stack calls (`torch_call_dispatcher("aten::op", ...)`)
   for ops with no `Scalar`-typed argument (`silu`, `sigmoid`, `rsqrt`,
   `gelu`, etc.).
3. Codegen'd C shim functions
   (`torch/csrc/inductor/aoti_torch/generated/c_shim_cuda.h`'s
   `aoti_torch_cuda_<op>`) for ops with a `Scalar`-typed argument. `Scalar`
   isn't yet generically boxable through the `StableIValue` stack (see the
   `fill_`/`full` comments in `torch::stable::ops.h`), but these shim
   functions take the Scalar as a plain `double`. `clamp`/`clamp_max` have no
   shim at all, so a constant tensor is built with `torch::stable::full(...)`
   and `aten::minimum`/`aten::maximum` (Tensor-only, no Scalar) are used
   instead.

In addition, where a single existing composite op could stand in for a
hand-rolled pow/mean/rsqrt chain, it was preferred:
`rms_norm`/`fused_add_rms_norm` now delegate to
`aoti_torch_cuda__fused_rms_norm` in one call (verified numerically identical
to the manual fp32 formula).

---

## 3. Files changed

| File | Change | Content |
| :-- | :-- | :-- |
| `csrc/v100_adapter/v100_fallback_ops.cpp` | new → rewritten | ATen fallback implementation on the stable ABI (`torch::stable::Tensor`) + `STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops)` registration |
| `csrc/libtorch_stable/torch_bindings.cpp` | changed | Native registration of the 6 target ops gated `#ifndef VLLM_V100_ADAPTER`. Also gates the DeepGEMM-only `silu_and_mul_quant` / `persistent_masked_m_silu_mul_quant` the same way (§5.3) |
| `CMakeLists.txt` | changed | `VLLM_V100_ADAPTER` gate, adapter source addition, re-enables the two stable extension targets, registers the actual excluded files in `VLLM_SM70_EXCLUDED_SRCS` |
| `csrc/libtorch_stable/cuda_vec_utils.cuh` | changed | Added `__CUDA_ARCH__ >= 800` guards to 3 BF16 packed-conversion functions (§5.3) |
| `csrc/libtorch_stable/moe/topk_softmax_kernels.cu`<br>`csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.cu` | changed | Same guard added at the BF16 packed-conversion call site in the MoE routing kernels |
| `csrc/libtorch_stable/quantization/activation_kernels.cu` | changed | Fixed a `constexpr` BF16 construction that didn't compile under `__CUDA_ARCH__ < 800`. The whole file is excluded via `VLLM_SM70_EXCLUDED_SRCS` (§5.3) |
| `pyproject.toml` / `requirements/cuda.txt` / `requirements/build/cuda.txt` | changed | torch pin updated to `2.11.0+cu126` (§5.0) |
| `test_v100/conftest.py` | changed | Fixed a bug where the real built `vllm._C_stable_libtorch` was never imported before checking for it |
| `docs/v100_fallback_archtecture/sm70_op_compatibility_matrix.md` | changed | Status update for implemented/excluded ops |
| `test_v100/` | existing | CI numerical-verification suite (8/8 passing against the real build) |

---

## 4. Adapter implementation (`csrc/v100_adapter/v100_fallback_ops.cpp`)

The adapter implements ATen fallbacks for the following 6 ops, which the
compatibility matrix marks as **Missing**, and they're now verified on real
hardware (§6). Numerics match the native kernels by computing in **fp32
intermediate precision** and casting back to the output dtype.

| Op | Fallback implementation |
| :-- | :-- |
| `rms_norm` | Calls `aten::_fused_rms_norm` directly (verified numerically identical to `input * rsqrt(mean(input^2, -1) + eps) [* weight]`) |
| `fused_add_rms_norm` | `residual := input + residual` (via `aten::add.Tensor`); `input := rms_norm(residual) [* weight]` (reuses the `rms_norm` implementation above, in place) |
| `silu_and_mul` | `silu(gate) * up` |
| `silu_and_mul_with_clamp` | `(gate.clamp_max(limit) * sigmoid(alpha*gate)) * (up.clamp(±limit) + beta)` (clamp implemented via `full()` + `minimum`/`maximum`, §2.2) |
| `gelu_and_mul` | `gelu(gate, 'none') * up` |
| `gelu_tanh_and_mul` | `gelu(gate, 'tanh') * up` |

- Gated-activation inputs are laid out as `[..., 2*d]`, split into the first
  half (`gate`) and second half (`up`).
- Each function's signature matches its `_C` op schema exactly, so the Python
  call site is unchanged.
- Registration is `STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops)` wrapped in
  `#ifdef VLLM_V100_ADAPTER`.

Not yet implemented (future work): `rotary_embedding`,
`get_cuda_view_from_cpu_tensor`, `topk_topp_sampler` — listed as "⏳ Planned"
in the matrix. `paged_attention_v1/v2` and `reshape_and_cache` are natively
supported on sm_70 already and **must not be touched**.

---

## 5. Build-system integration (`CMakeLists.txt` / `setup.py`)

### 5.0 Required torch version: `2.11.0+cu126`

Actually building `_C_stable_libtorch` / `_moe_C_stable_libtorch` requires a
torch build that satisfies **both** of the following conditions. They must be
checked independently.

1. **sm_70 (Volta) must be bundled.** cu126 wheels keep sm_70 through at
   least torch 2.14.1. **cu128/cu129 wheels dropped sm_70 starting torch
   2.11** (a cuDNN bump forced it out). cu13x wheels never had sm_70 to begin
   with.
2. **The stable-ABI headers the adapter actually uses must be present.** The
   CMake gate only checks for the existence of
   `torch/csrc/stable/library.h` (present from torch 2.7+), but the real code
   also needs `torch/csrc/stable/ops.h` and
   `torch/headeronly/util/Exception.h`. These are **not yet present in torch
   2.8** and only show up from torch 2.9 onward.

The intersection of both constraints is `torch==2.11.0+cu126`. All three
pins — `pyproject.toml` (`[build-system].requires` and
`[tool.uv].find-links`), `requirements/cuda.txt`, and
`requirements/build/cuda.txt` — must be updated together (they interact with
`[tool.uv]`'s `no-build-isolation-package = ["torch"]`; an inconsistent pin
breaks `uv`'s build-isolation bypass and the resolve fails).

If torch is upgraded again in the future, re-verify both conditions
independently (`torch.cuda.get_arch_list()` and
`find .venv/.../torch/include/torch/csrc/stable -name ops.h`).

### 5.1 The `VLLM_V100_ADAPTER` gate

Right after `VLLM_STABLE_EXT_SRC` is defined, a check turns on
`VLLM_V100_ADAPTER_ENABLE` only when every entry in `CUDA_ARCHS` is CC < 8.0.
When enabled:

- The adapter source is appended to `VLLM_STABLE_EXT_SRC`.
- `set_source_files_properties(...)` applies
  `COMPILE_DEFINITIONS "VLLM_V100_ADAPTER"` to both the adapter and
  `torch_bindings.cpp`.

The check correctly handles suffixed arch strings too (`9.0a`, `10.0f`,
`12.0f`, etc.) — only `7.0`/`7.5` turns it ON; anything including 8.0+ turns
it OFF.

### 5.2 Re-enabling the stable extension targets

The `define_extension_target(_C_stable_libtorch ...)` and
`define_extension_target(_moe_C_stable_libtorch ...)` blocks (including their
`target_compile_definitions` and ROCm link blocks), which had been commented
out during earlier trial-and-error to dodge build errors, have been restored,
and both extensions have been restored to `setup.py`'s `ext_modules` (without
this, `build_ext` never issues the `--target`, so fixing the CMake side alone
doesn't produce a build).

**On 2026-10-05, this was actually built, linked, and loaded successfully
using torch 2.11.0+cu126 from §5.0.** The resulting shared objects are
`vllm/_C_stable_libtorch.abi3.so` and `vllm/_moe_C_stable_libtorch.abi3.so`;
after `import vllm._C_stable_libtorch`, `torch.ops._C.rms_norm` etc. resolve
correctly.

Volta source build:

```bash
# If the ninja executable isn't on PATH, shim it first and prepend its dir
printf '#!/bin/bash\nexec ".venv/bin/python" -m ninja "$@"\n' > bin/ninja && chmod +x bin/ninja
export PATH="$(pwd)/bin:$PATH" CUDA_HOME=/usr
TORCH_CUDA_ARCH_LIST="7.0" MAX_JOBS=4 .venv/bin/python setup.py develop
```

After changing the torch version, delete
`build/temp.linux-x86_64-cpython-312` before rebuilding — otherwise a stale
`TORCH_INCLUDE_DIRS` can linger in the CMake cache and cause header
mismatches.

### 5.3 Automatic exclusion of sm_80+ sources (guarding unrelated kernels)

sm_80+-only kernels are already excluded from an sm_70-only build via two
existing mechanisms:

- **Heavy families** (machete, cutlass scaled_mm/moe, marlin bf16/fp8, fp4,
  w4a8): excluded automatically because `cuda_archs_loose_intersection`
  produces an empty arch intersection. Each `.cu` registers its own ops, so
  excluding them never causes a link error.
- **sm_80+ external projects** (flash-attn, deepgemm, fmha_sm100, flashmla,
  flashkda, qutlass, tml_fa4): excluded from sm_70 builds via the
  `_HAS_AMPERE_OR_NEWER` guard.

For the remaining case — an unconditionally-listed base source that still
fails to compile for sm_70 — there's an extensible exclusion hook:

```cmake
set(VLLM_SM70_EXCLUDED_SRCS "...")        # for _C_stable_libtorch
set(VLLM_SM70_MOE_EXCLUDED_SRCS "")       # for _moe_C_stable_libtorch
```

One file actually hit this pattern during the first
`TORCH_CUDA_ARCH_LIST="7.0"` build:

- `csrc/libtorch_stable/quantization/activation_kernels.cu` (the DeepGEMM
  `silu_mul_fp8_quant_deep_gemm_kernel` and friends): uses BF16/FP8 hardware
  intrinsics (`make_bfloat162`, etc.) unconditionally and extensively enough
  that patching each call site individually wasn't practical, so the whole
  file was added to `VLLM_SM70_EXCLUDED_SRCS`. The two ops it registered
  (`silu_and_mul_quant`, `persistent_masked_m_silu_mul_quant`) were gated
  `#ifndef VLLM_V100_ADAPTER` in `torch_bindings.cpp` to avoid an
  undefined-symbol link error. This is safe because both ops' only Python
  call site
  (`vllm/model_executor/layers/fusion/fused_act_quant.py`) already gates on
  `has_device_capability(90)`, making them unreachable on sm_70 regardless.

For single call sites that don't warrant excluding a whole file, a lighter
pattern was used instead: an `#if __CUDA_ARCH__ >= 800` guard with `__trap()`
in the `#else` branch (safe dead code on V100, since BF16 tensors never
reach device code there in the first place — see the BF16->FP16 fallback in
`vllm/platforms/cuda.py`). Applied to: the 3 BF16 packed-conversion functions
in `csrc/libtorch_stable/cuda_vec_utils.cuh`, and the BF16 vectorized-load
path in `csrc/libtorch_stable/moe/topk_softmax_kernels.cu` /
`topk_softplus_sqrt_kernels.cu`. This mirrors an existing upstream convention
already present in `csrc/libtorch_stable/type_convert.cuh`:
`#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800) ...`.

**Note**: if an excluded source's op is registered centrally in
`torch_bindings.cpp`, that `ops.impl(...)` must also be gated
`#ifndef VLLM_V100_ADAPTER` — otherwise it's an undefined-symbol link error
(same pattern as the adapter itself).

---

## 6. Verification (`test_v100/`)

A fork-local numerical verification suite lives in `test_v100/`, kept
separate from upstream `tests/` to avoid merge conflicts and to be runnable
independently.

- Selects a CC < 8.0 GPU (preferring a real V100); skips cleanly if none is
  present.
- Uses an already-built `vllm._C_stable_libtorch` directly if one exists
  (installed mode — this is the path used by the real build in §5). If not,
  the plan was to JIT-compile the real adapter `.cpp` with
  `torch.utils.cpp_extension` (`-DVLLM_V100_ADAPTER`) so the suite could be
  run without a full vLLM build — but since the adapter is now stable-ABI-only
  code (§2.2), this JIT path is currently unverified and needs revisiting.
  `test_v100/conftest.py` had a bug where it never actually
  `import vllm._C_stable_libtorch`'d before checking for the installed-mode
  ops, so it always fell through to the (now defunct) JIT path; fixed on
  2026-10-05.
- Calls each `torch.ops._C.<op>` and compares against an fp32 PyTorch
  reference (fp16 tolerance).

Run it:

```bash
.venv/bin/python -m pytest test_v100/ -v
```

### Hardware results (Tesla V100-SXM2-16GB, sm_70, 2026-10-05, real build)

All 8 cases PASS.

```
test_rms_norm[True] / [False]            PASSED
test_fused_add_rms_norm                  PASSED
test_silu_and_mul                        PASSED
test_gelu_and_mul                        PASSED
test_gelu_tanh_and_mul                   PASSED
test_silu_and_mul_with_clamp[defaults] / [alpha_beta]  PASSED
```

End-to-end real-model behavior was also verified:

```python
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="float16",
          gpu_memory_utilization=0.5, max_model_len=512, enforce_eager=True)
out = llm.generate(["The capital of France is", "2+2="],
                    SamplingParams(temperature=0.0, max_tokens=32))
```

- The `TRITON_ATTN` backend is selected (`FLASH_ATTN`/`FLASHINFER` are
  excluded for CC < 8.0, as designed).
- The model's `bfloat16` weights are cast to `float16` (the BF16->FP16
  fallback).
- `rms_norm` / `silu_and_mul` etc. are called through the adapter (`vllm_c`
  in the IR priority list), and generation is correct
  ("The capital of France is" -> " Paris. ..." etc.).

### Verification on other architectures (Qwen3.5 / Gemma4, 2026-10-05)

To confirm the adapter works generically and isn't limited to the Qwen2
family, two more models — representing newer architecture families already
implemented in this fork — were additionally verified (same
`dtype=float16`, `enforce_eager=True` recipe).

| Model | Architecture | What was verified |
| :-- | :-- | :-- |
| `Qwen/Qwen3.5-0.8B` | `Qwen3_5ForConditionalGeneration` (Mamba+attention hybrid, Gated DeltaNet linear attention) | Loads successfully. The fused GDN CUDA decode kernel correctly detects it requires CC >= 8.0 and falls back to the Triton/FLA implementation. FA2 is also explicitly rejected, selecting `TRITON_ATTN`. Generation is correct ("The capital of France is" -> " Paris." etc. — some repetition typical of a 0.8B model under greedy decoding, but numerically correct). |
| `google/gemma-4-e2b-it` | `Gemma4ForConditionalGeneration` (multimodal, heterogeneous attention head dims `{sliding_attention: 256, full_attention: 512}`) | Loads ~9.85 GiB of weights, fitting within V100 VRAM. FA4 is detected as unsupported, selecting `TRITON_ATTN`. **Via the chat template** (`llm.chat(...)`), output is fully correct: `"The capital of France is **Paris**."` / `"2 + 2 = **4**"`. Raw-text completion (`llm.generate()`) produced a repetition loop, but a side-by-side comparison at the same precision confirmed this is normal instruct-model behavior on an uncoerced raw prompt, not a V100/adapter bug. |

Both models exercise the V100 adapter path for `rms_norm` /
`fused_add_rms_norm` / `silu_and_mul`-family ops, giving additional evidence
that the adapter isn't tied to one specific model family.

---

## 7. Future work

1. ~~Run a full source build with `TORCH_CUDA_ARCH_LIST="7.0"` and register
   any source that fails the first time in `VLLM_SM70_EXCLUDED_SRCS`.~~ →
   **done** (§5.3).
2. Implement fallbacks for the remaining ops still marked "⏳ Planned" in the
   compatibility matrix: `rotary_embedding`, `get_cuda_view_from_cpu_tensor`,
   `topk_topp_sampler` (currently undiscovered unless a code path actually
   exercises them).
3. ~~After a successful build, verify end-to-end behavior and accuracy on
   real models (Qwen2.5 / Gemma / etc.).~~ → **done** (verified on
   Qwen2.5-0.5B-Instruct, Qwen3.5-0.8B, and Gemma4-E2B-it; see above).
   Further verification on larger models (7B+), longer generations, batched
   requests, and MoE architectures remains future work.
4. Verify/fix whether `test_v100/`'s JIT-compile path (the fallback for a
   non-installed-mode run) still works now that the adapter is stable-ABI-only
   code.
5. When upgrading torch again in the future, re-check both conditions in
   §5.0 (sm_70 presence / stable-ABI header presence) independently each
   time.
