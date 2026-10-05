"""Numerical correctness of the V100 (sm_70) ATen fallbacks.

Each test exercises one ``torch.ops._C`` op provided by
``csrc/v100_adapter/v100_fallback_ops.cpp`` on a pre-Ampere GPU and compares the
result against an fp32 reference computed with PyTorch. Tolerances are sized for
fp16 round-off. The suite skips cleanly when no CC < 8.0 GPU is present (so it is
safe to run in any CI), and JIT-compiles the real adapter source so running it
after an upstream merge re-verifies the fallbacks against the current libtorch.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

# fp16 round-off: the adapter computes in fp32 then stores fp16, as the native
# kernels do, so differences are at the fp16 ULP level.
ATOL = 2e-2
RTOL = 2e-2
DTYPE = torch.float16


def _assert_close(got, ref, name):
    got_f = got.float()
    ref_f = ref.float()
    max_abs = (got_f - ref_f).abs().max().item()
    assert torch.allclose(got_f, ref_f, atol=ATOL, rtol=RTOL), (
        f"{name}: max_abs_err={max_abs:.3e} exceeds tol (atol={ATOL}, rtol={RTOL})"
    )


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.mark.parametrize("with_weight", [True, False])
def test_rms_norm(ops, device, with_weight):
    n, h, eps = 32, 512, 1e-6
    x = torch.randn(n, h, device=device, dtype=DTYPE)
    w = torch.randn(h, device=device, dtype=DTYPE) if with_weight else None
    out = torch.empty_like(x)

    ops.rms_norm(out, x, w, eps)

    xf = x.float()
    ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if w is not None:
        ref = ref * w.float()
    _assert_close(out, ref, "rms_norm")


def test_fused_add_rms_norm(ops, device):
    n, h, eps = 32, 512, 1e-6
    x = torch.randn(n, h, device=device, dtype=DTYPE)
    residual = torch.randn(n, h, device=device, dtype=DTYPE)
    w = torch.randn(h, device=device, dtype=DTYPE)

    added_ref = x.float() + residual.float()
    norm_ref = added_ref * torch.rsqrt(added_ref.pow(2).mean(-1, keepdim=True) + eps)
    norm_ref = norm_ref * w.float()

    x_io, res_io = x.clone(), residual.clone()
    ops.fused_add_rms_norm(x_io, res_io, w, eps)

    # residual carries the pre-norm sum; input carries the normalized output.
    _assert_close(res_io, added_ref, "fused_add_rms_norm[residual]")
    _assert_close(x_io, norm_ref, "fused_add_rms_norm[input]")


def test_silu_and_mul(ops, device):
    n, d = 32, 512
    x = torch.randn(n, 2 * d, device=device, dtype=DTYPE)
    out = torch.empty(n, d, device=device, dtype=DTYPE)

    ops.silu_and_mul(out, x)

    gate, up = x[..., :d].float(), x[..., d:].float()
    _assert_close(out, F.silu(gate) * up, "silu_and_mul")


def test_gelu_and_mul(ops, device):
    n, d = 32, 512
    x = torch.randn(n, 2 * d, device=device, dtype=DTYPE)
    out = torch.empty(n, d, device=device, dtype=DTYPE)

    ops.gelu_and_mul(out, x)

    gate, up = x[..., :d].float(), x[..., d:].float()
    _assert_close(out, F.gelu(gate) * up, "gelu_and_mul")


def test_gelu_tanh_and_mul(ops, device):
    n, d = 32, 512
    x = torch.randn(n, 2 * d, device=device, dtype=DTYPE)
    out = torch.empty(n, d, device=device, dtype=DTYPE)

    ops.gelu_tanh_and_mul(out, x)

    gate, up = x[..., :d].float(), x[..., d:].float()
    _assert_close(out, F.gelu(gate, approximate="tanh") * up, "gelu_tanh_and_mul")


@pytest.mark.parametrize(
    "limit,alpha,beta",
    [(3.0, 1.0, 0.0), (2.0, 1.702, 0.5)],
    ids=["defaults", "alpha_beta"],
)
def test_silu_and_mul_with_clamp(ops, device, limit, alpha, beta):
    n, d = 32, 512
    x = torch.randn(n, 2 * d, device=device, dtype=DTYPE)
    out = torch.empty(n, d, device=device, dtype=DTYPE)

    ops.silu_and_mul_with_clamp(out, x, limit, alpha, beta)

    gate = x[..., :d].float().clamp(max=limit)
    up = x[..., d:].float().clamp(min=-limit, max=limit)
    ref = (gate * torch.sigmoid(alpha * gate)) * (up + beta)
    _assert_close(out, ref, "silu_and_mul_with_clamp")
