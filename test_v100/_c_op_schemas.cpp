// Standalone `_C` op schema definitions used only by the test_v100 JIT harness.
//
// In a real sm_70 build these schemas come from the stable-ABI fragment in
// csrc/libtorch_stable/torch_bindings.cpp. When the test harness JIT-compiles
// csrc/v100_adapter/v100_fallback_ops.cpp in a process that has NOT imported a
// built vllm._C, there is no schema for the adapter's TORCH_LIBRARY_IMPL to bind
// to, so we declare them here. This file is compiled ONLY in that JIT path (when
// torch.ops._C is not already populated); it is never part of the vLLM build.
#include <torch/all.h>
#include <torch/library.h>

TORCH_LIBRARY(_C, m) {
  m.def("rms_norm(Tensor! result, Tensor input, Tensor? weight, float epsilon) -> ()");
  m.def("fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor? weight, float epsilon) -> ()");
  m.def("silu_and_mul(Tensor! result, Tensor input) -> ()");
  m.def("silu_and_mul_with_clamp(Tensor! result, Tensor input, float limit, float alpha=1.0, float beta=0.0) -> ()");
  m.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");
  m.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");
}
