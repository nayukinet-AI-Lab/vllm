"""Shared fixtures for the V100 (sm_70) C++ adapter test suite.

These tests validate the ATen-native fallbacks in
``csrc/v100_adapter/v100_fallback_ops.cpp`` on real pre-Ampere (CC < 8.0)
hardware without requiring a full vLLM build: the adapter is JIT-compiled with
``torch.utils.cpp_extension`` and its ops are exercised through
``torch.ops._C``. See ``test_v100/README.md`` for the rationale and how this
fits the fork's upstream-merge verification workflow.
"""

from __future__ import annotations

import os
import shutil
import stat
import sys
import tempfile

import pytest

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard dependency of vLLM
    torch = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTER_SRC = os.path.join(REPO_ROOT, "csrc", "v100_adapter", "v100_fallback_ops.cpp")
SCHEMAS_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_c_op_schemas.cpp")

# The ops the adapter provides fallbacks for.
ADAPTER_OPS = (
    "rms_norm",
    "fused_add_rms_norm",
    "silu_and_mul",
    "silu_and_mul_with_clamp",
    "gelu_and_mul",
    "gelu_tanh_and_mul",
)


def _pre_ampere_device():
    """Return the first CC < 8.0 CUDA device, preferring a real V100, or None."""
    if torch is None or not torch.cuda.is_available():
        return None
    preferred = None
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        if cap < (8, 0):
            if "V100" in torch.cuda.get_device_name(i):
                return torch.device(f"cuda:{i}")
            if preferred is None:
                preferred = torch.device(f"cuda:{i}")
    return preferred


def _ensure_ninja_on_path():
    """Make a `ninja` executable resolvable, shimming `python -m ninja` if needed."""
    if shutil.which("ninja"):
        return
    try:
        import ninja  # noqa: F401
    except ImportError:
        pytest.skip("ninja is required to JIT-compile the V100 adapter")
    shim_dir = os.path.join(tempfile.gettempdir(), "v100_adapter_ninja_shim")
    os.makedirs(shim_dir, exist_ok=True)
    shim = os.path.join(shim_dir, "ninja")
    if not os.path.exists(shim):
        with open(shim, "w") as f:
            f.write(f'#!/bin/sh\nexec "{sys.executable}" -m ninja "$@"\n')
        os.chmod(shim, os.stat(shim).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    os.environ["PATH"] = shim_dir + os.pathsep + os.environ.get("PATH", "")


def _c_ops_already_registered() -> bool:
    """True if a built vllm._C already exposes the adapter ops (installed mode)."""
    try:
        getattr(torch.ops._C, "rms_norm")
        return True
    except (AttributeError, RuntimeError):
        return False


@pytest.fixture(scope="session")
def device():
    """A pre-Ampere (CC < 8.0) CUDA device, or skip the suite."""
    dev = _pre_ampere_device()
    if dev is None:
        pytest.skip(
            "No CC < 8.0 (Volta/Turing) GPU available; the V100 fallbacks are "
            "only meaningful on sm_70/sm_75 hardware"
        )
    return dev


@pytest.fixture(scope="session")
def ops(device):
    """`torch.ops._C` with the adapter ops available.

    If a built ``vllm._C`` already provides them (installed mode) they are used
    directly; otherwise the real adapter source is JIT-compiled against the
    currently installed libtorch (which is what we want to re-verify after an
    upstream merge).
    """
    assert os.path.exists(ADAPTER_SRC), f"adapter source missing: {ADAPTER_SRC}"
    if not _c_ops_already_registered():
        _ensure_ninja_on_path()
        from torch.utils.cpp_extension import load

        load(
            name="v100_adapter_ci",
            sources=[SCHEMAS_SRC, ADAPTER_SRC],
            extra_cflags=["-DVLLM_V100_ADAPTER", "-std=c++17", "-O2"],
            is_python_module=False,
            verbose=bool(os.environ.get("V100_ADAPTER_VERBOSE")),
        )
    missing = [op for op in ADAPTER_OPS if not hasattr(torch.ops._C, op)]
    assert not missing, f"adapter failed to register ops: {missing}"
    return torch.ops._C
