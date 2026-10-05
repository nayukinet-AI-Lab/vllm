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

#include <torch/all.h>
#include <torch/library.h>

#include <optional>
#include <utility>

namespace v100_adapter {

namespace {
// Gated-activation inputs are laid out as [..., 2 * d]; split into the
// (gate, up) halves along the last dimension.
std::pair<at::Tensor, at::Tensor> split_gate_up(const at::Tensor& input) {
  const auto d = input.size(-1) / 2;
  return {input.narrow(-1, 0, d), input.narrow(-1, d, d)};
}
}  // namespace

// rms_norm(Tensor! result, Tensor input, Tensor? weight, float epsilon) -> ()
// result = input * rsqrt(mean(input^2, -1) + epsilon) [* weight], computed in
// fp32 then cast back, matching csrc/libtorch_stable/layernorm_kernels.cu.
void rms_norm(at::Tensor& result, const at::Tensor& input,
              std::optional<at::Tensor> weight, double epsilon) {
  const auto x = input.to(at::kFloat);
  const auto variance = x.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
  auto normed = x * at::rsqrt(variance + epsilon);
  if (weight.has_value()) {
    normed = normed * weight->to(at::kFloat);
  }
  result.copy_(normed);
}

// fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor? weight,
//                    float epsilon) -> ()
// residual := input + residual (the pre-norm sum carried to the next layer);
// input := rms_norm(residual) [* weight]. Both updates are in place.
void fused_add_rms_norm(at::Tensor& input, at::Tensor& residual,
                        std::optional<at::Tensor> weight, double epsilon) {
  const auto added = input.to(at::kFloat) + residual.to(at::kFloat);
  residual.copy_(added);
  const auto variance = added.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
  auto normed = added * at::rsqrt(variance + epsilon);
  if (weight.has_value()) {
    normed = normed * weight->to(at::kFloat);
  }
  input.copy_(normed);
}

// silu_and_mul(Tensor! result, Tensor input) -> ()
// result = silu(gate) * up.
void silu_and_mul(at::Tensor& result, const at::Tensor& input) {
  const auto halves = split_gate_up(input);
  result.copy_(at::silu(halves.first) * halves.second);
}

// silu_and_mul_with_clamp(Tensor! result, Tensor input, float limit,
//                         float alpha=1.0, float beta=0.0) -> ()
// Matches silu_and_mul_clamp in activation_kernels.cu: gate is clamped to
// (-inf, limit], up is clamped to [-limit, limit], and
//   result = (gate * sigmoid(alpha * gate)) * (up + beta).
// alpha=1.0, beta=0.0 reduce this to silu(gate) * up.
void silu_and_mul_with_clamp(at::Tensor& result, const at::Tensor& input,
                             double limit, double alpha, double beta) {
  const auto halves = split_gate_up(input);
  const auto gate = at::clamp_max(halves.first.to(at::kFloat), limit);
  const auto up = at::clamp(halves.second.to(at::kFloat), -limit, limit);
  const auto act = gate * at::sigmoid(alpha * gate);
  result.copy_(act * (up + beta));
}

// gelu_and_mul(Tensor! out, Tensor input) -> ()  ('none' approximation)
void gelu_and_mul(at::Tensor& out, const at::Tensor& input) {
  const auto halves = split_gate_up(input);
  out.copy_(at::gelu(halves.first) * halves.second);
}

// gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()  ('tanh' approximation)
void gelu_tanh_and_mul(at::Tensor& out, const at::Tensor& input) {
  const auto halves = split_gate_up(input);
  out.copy_(at::gelu(halves.first, "tanh") * halves.second);
}

}  // namespace v100_adapter

#ifdef VLLM_V100_ADAPTER
// Scoped to the CUDA dispatch key only so the native sm_80+ kernels for these
// op names are never shadowed on Ampere/Hopper/Blackwell (those builds do not
// define VLLM_V100_ADAPTER). The ops are declared (ops.def) by the stable-ABI
// fragment in csrc/libtorch_stable/torch_bindings.cpp; here we only provide the
// sm_70 implementation.
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("rms_norm", &v100_adapter::rms_norm);
  m.impl("fused_add_rms_norm", &v100_adapter::fused_add_rms_norm);
  m.impl("silu_and_mul", &v100_adapter::silu_and_mul);
  m.impl("silu_and_mul_with_clamp", &v100_adapter::silu_and_mul_with_clamp);
  m.impl("gelu_and_mul", &v100_adapter::gelu_and_mul);
  m.impl("gelu_tanh_and_mul", &v100_adapter::gelu_tanh_and_mul);
}
#endif  // VLLM_V100_ADAPTER
