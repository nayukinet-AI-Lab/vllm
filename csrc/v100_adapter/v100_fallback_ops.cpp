// V100 (Compute Capability 7.0 / sm_70) ATen-native fallbacks for the
// `torch.ops._C` custom ops that are missing or unsupported on Volta.
//
// Background and the full op status table live in
// docs/v100_fallback_archtecture/ (see the v100-csrc-adapter rule). The Python
// model layer keeps calling `torch.ops._C.<op>` unchanged; on an sm_70 build the
// dispatcher routes those calls here instead of the native sm_80+ kernels.
//
// These implementations and their registrations are compiled only when
// VLLM_V100_ADAPTER is defined. CMake defines it exclusively for CUDA builds
// whose target architectures are all pre-Ampere (CC < 8.0), and in that same
// build the matching native CUDA registrations in
// csrc/libtorch_stable/torch_bindings.cpp are #ifndef-gated out, so exactly one
// CUDA kernel is registered per op. Ampere/Hopper/Blackwell builds never define
// the macro and are therefore unaffected.
//
// This file is compiled as part of the `_C_stable_libtorch` target (built
// against the libtorch *stable* ABI: Py_LIMITED_API + TORCH_TARGET_VERSION), so
// it must use the torch::stable::Tensor surface (torch/csrc/stable/*) rather
// than legacy at::Tensor / torch/library.h — those headers #error out under
// TORCH_TARGET_VERSION. Ops with no pre-made torch::stable::ops.h wrapper are
// called directly through the dispatcher stack (torch_call_dispatcher), the
// same mechanism torch::stable::ops.h itself is built on (see e.g.
// csrc/libtorch_stable/cuda_view.cu for another direct use of this pattern in
// this codebase). Ops with Scalar-typed arguments (add/mul by a Python number,
// pow by a Python number) go through the codegen'd per-op C shim
// (aoti_torch_cuda_*) instead: Scalar is not yet generically boxable through
// the StableIValue stack (see the `fill_`/`full` comments in
// torch/csrc/stable/ops.h), but the shim takes the Scalar as a plain C double.

#include <torch/csrc/inductor/aoti_torch/generated/c_shim_cuda.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#include <array>
#include <optional>
#include <string>
#include <utility>

namespace v100_adapter {

namespace {

// Call a no-Scalar-argument op of the given arity through the stable ABI
// dispatcher stack (torch::stable::ops.h uses this same mechanism for ops
// like zero_/clone/sigmoid-shaped unary ops; it just doesn't happen to wrap
// these particular op names).
torch::stable::Tensor call1(const char* op_name,
                            const torch::stable::Tensor& self) {
  std::array<StableIValue, 1> stack{torch::stable::detail::from(self)};
  TORCH_ERROR_CODE_CHECK(
      torch_call_dispatcher(op_name, "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

torch::stable::Tensor call2(const char* op_name,
                            const torch::stable::Tensor& a,
                            const torch::stable::Tensor& b) {
  std::array<StableIValue, 2> stack{torch::stable::detail::from(a),
                                    torch::stable::detail::from(b)};
  TORCH_ERROR_CODE_CHECK(
      torch_call_dispatcher(op_name, "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

torch::stable::Tensor gelu(const torch::stable::Tensor& self,
                           const char* approximate) {
  std::array<StableIValue, 2> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(std::string(approximate))};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher("aten::gelu", "", stack.data(),
                                               TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

torch::stable::Tensor mul_tensor(const torch::stable::Tensor& a,
                                 const torch::stable::Tensor& b) {
  AtenTensorHandle ret0 = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_mul_Tensor(a.get(), b.get(), &ret0));
  return torch::stable::Tensor(ret0);
}

torch::stable::Tensor add_scalar(const torch::stable::Tensor& a, double other) {
  AtenTensorHandle ret0 = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_cuda_add_Scalar(a.get(), other, /*alpha=*/1.0, &ret0));
  return torch::stable::Tensor(ret0);
}

torch::stable::Tensor mul_scalar(const torch::stable::Tensor& a, double other) {
  AtenTensorHandle ret0 = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_mul_Scalar(a.get(), other, &ret0));
  return torch::stable::Tensor(ret0);
}

// A tensor the same shape/dtype/device as `ref`, filled with `value`.
// Used to clamp via minimum/maximum (Tensor, Tensor) instead of
// clamp/clamp_max (Tensor, Scalar), since Scalar args aren't yet generically
// boxable through the dispatcher stack (see the file header comment).
torch::stable::Tensor full_like_value(const torch::stable::Tensor& ref,
                                      double value) {
  return torch::stable::full(ref.sizes(), value, ref.scalar_type(),
                             ref.layout(), ref.device());
}

// Gated-activation inputs are laid out as [..., 2 * d]; split into the
// (gate, up) halves along the last dimension.
std::pair<torch::stable::Tensor, torch::stable::Tensor> split_gate_up(
    torch::stable::Tensor& input) {
  const int64_t last_dim = input.dim() - 1;
  const int64_t d = input.size(last_dim) / 2;
  auto gate = torch::stable::narrow(input, last_dim, 0, d);
  auto up = torch::stable::narrow(input, last_dim, d, d);
  return {gate, up};
}

}  // namespace

// rms_norm(Tensor! result, Tensor input, Tensor? weight, float epsilon) -> ()
// Delegates to aten::_fused_rms_norm (a plain composite CUDA op, not a
// tensor-core kernel, so it runs unmodified on sm_70); verified to match this
// adapter's previous hand-rolled fp32 formula bit-for-bit.
void rms_norm(torch::stable::Tensor& out, torch::stable::Tensor& input,
              std::optional<torch::stable::Tensor> weight, double epsilon) {
  const std::array<int64_t, 1> normalized_shape{input.size(input.dim() - 1)};
  AtenTensorHandle weight_handle = weight.has_value() ? weight->get() : nullptr;
  AtenTensorHandle* weight_ptr = weight.has_value() ? &weight_handle : nullptr;
  double eps = epsilon;
  AtenTensorHandle ret0 = nullptr;
  AtenTensorHandle ret1 = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_cuda__fused_rms_norm(
      input.get(), normalized_shape.data(), normalized_shape.size(),
      weight_ptr, &eps, &ret0, &ret1));
  torch::stable::Tensor result(ret0);
  torch::stable::Tensor rstd_unused(ret1);  // unused, but owns ret1's handle
  torch::stable::copy_(out, result);
}

// fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor? weight,
//                    float epsilon) -> ()
// residual := input + residual (the pre-norm sum carried to the next layer);
// input := rms_norm(residual) [* weight]. Both updates are in place.
void fused_add_rms_norm(torch::stable::Tensor& input,
                        torch::stable::Tensor& residual,
                        std::optional<torch::stable::Tensor> weight,
                        double epsilon) {
  AtenTensorHandle added_handle = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_add_Tensor(
      input.get(), residual.get(), /*alpha=*/1.0, &added_handle));
  torch::stable::Tensor added(added_handle);
  torch::stable::copy_(residual, added);
  rms_norm(input, added, weight, epsilon);
}

// silu_and_mul(Tensor! result, Tensor input) -> ()
// result = silu(gate) * up.
void silu_and_mul(torch::stable::Tensor& out, torch::stable::Tensor& input) {
  auto halves = split_gate_up(input);
  auto activated = call1("aten::silu", halves.first);
  auto result = mul_tensor(activated, halves.second);
  torch::stable::copy_(out, result);
}

// silu_and_mul_with_clamp(Tensor! result, Tensor input, float limit,
//                         float alpha=1.0, float beta=0.0) -> ()
// Matches silu_and_mul_clamp in activation_kernels.cu: gate is clamped to
// (-inf, limit], up is clamped to [-limit, limit], and
//   result = (gate * sigmoid(alpha * gate)) * (up + beta).
// alpha=1.0, beta=0.0 reduce this to silu(gate) * up.
void silu_and_mul_clamp(torch::stable::Tensor& out,
                        torch::stable::Tensor& input, double limit,
                        double alpha = 1.0, double beta = 0.0) {
  auto halves = split_gate_up(input);
  auto gate = torch::stable::to(halves.first, torch::headeronly::ScalarType::Float);
  auto up = torch::stable::to(halves.second, torch::headeronly::ScalarType::Float);

  auto gate_clamped = call2("aten::minimum", gate, full_like_value(gate, limit));
  auto up_clamped =
      call2("aten::maximum",
            call2("aten::minimum", up, full_like_value(up, limit)),
            full_like_value(up, -limit));

  auto sig = call1("aten::sigmoid", mul_scalar(gate_clamped, alpha));
  auto act = mul_tensor(gate_clamped, sig);
  auto result = mul_tensor(act, add_scalar(up_clamped, beta));
  torch::stable::copy_(out, result);
}

// gelu_and_mul(Tensor! out, Tensor input) -> ()  ('none' approximation)
void gelu_and_mul(torch::stable::Tensor& out, torch::stable::Tensor& input) {
  auto halves = split_gate_up(input);
  auto activated = gelu(halves.first, "none");
  auto result = mul_tensor(activated, halves.second);
  torch::stable::copy_(out, result);
}

// gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()  ('tanh' approximation)
void gelu_tanh_and_mul(torch::stable::Tensor& out,
                       torch::stable::Tensor& input) {
  auto halves = split_gate_up(input);
  auto activated = gelu(halves.first, "tanh");
  auto result = mul_tensor(activated, halves.second);
  torch::stable::copy_(out, result);
}

}  // namespace v100_adapter

#ifdef VLLM_V100_ADAPTER
// Scoped to the CUDA dispatch key only so the native sm_80+ kernels for these
// op names are never shadowed on Ampere/Hopper/Blackwell (those builds do not
// define VLLM_V100_ADAPTER). The ops are declared (ops.def) by the stable-ABI
// fragment in csrc/libtorch_stable/torch_bindings.cpp; here we only provide the
// sm_70 implementation.
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl("rms_norm", TORCH_BOX(&v100_adapter::rms_norm));
  ops.impl("fused_add_rms_norm", TORCH_BOX(&v100_adapter::fused_add_rms_norm));
  ops.impl("silu_and_mul", TORCH_BOX(&v100_adapter::silu_and_mul));
  ops.impl("silu_and_mul_with_clamp",
           TORCH_BOX(&v100_adapter::silu_and_mul_clamp));
  ops.impl("gelu_and_mul", TORCH_BOX(&v100_adapter::gelu_and_mul));
  ops.impl("gelu_tanh_and_mul", TORCH_BOX(&v100_adapter::gelu_tanh_and_mul));
}
#endif  // VLLM_V100_ADAPTER
