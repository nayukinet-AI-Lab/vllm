# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.machinery
import os
import sys
import types

# In symlink mode (VLLM_FLASH_ATTN_SRC_DIR), cute/ is a symlink to the real
# source tree and its files use `flash_attn.cute.*` imports (not rewritten).
# Register a virtual `flash_attn` package so those imports resolve.
_cute_dir = os.path.join(os.path.dirname(__file__), "cute")
if os.path.islink(_cute_dir) and "flash_attn" not in sys.modules:
    _fa_mod = types.ModuleType("flash_attn")
    _fa_mod.__path__ = [os.path.dirname(os.path.realpath(_cute_dir))]
    _fa_mod.__package__ = "flash_attn"
    _fa_mod.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn", None, is_package=True
    )
    _fa_mod.__spec__.submodule_search_locations = _fa_mod.__path__
    sys.modules["flash_attn"] = _fa_mod

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

FLASH_ATTN_AVAILABLE = True
try:
    from vllm.vllm_flash_attn.flash_attn_interface import (  # noqa: E402
        FA2_AVAILABLE,
        FA3_AVAILABLE,
        compile_flash_attn_varlen_func_from_specs,
        fa_version_unsupported_reason,
        flash_attn_varlen_func,
        get_scheduler_metadata,
        is_fa_version_supported,
    )

    device_capability = current_platform.get_device_capability() if current_platform.is_cuda() else None
    if not (FA2_AVAILABLE or FA3_AVAILABLE) or (device_capability is not None and device_capability[0] < 8):
        FLASH_ATTN_AVAILABLE = False
        reason = "missing C++ extensions (_vllm_fa2_C or _vllm_fa3_C)" if not (FA2_AVAILABLE or FA3_AVAILABLE) else f"Compute Capability {device_capability} < 8.0 (V100)"
        logger.warning_once(
            "vllm_flash_attn is not available or not supported on this device (%s). "
            "FlashAttention will be disabled.",
            reason,
        )
except (ImportError, Exception) as e:
    FLASH_ATTN_AVAILABLE = False
    logger.warning_once(
        "Failed to import vllm_flash_attn extensions: %s. FlashAttention will be disabled.",
        e,
    )
    FA2_AVAILABLE = False
    FA3_AVAILABLE = False
    flash_attn_varlen_func = None  # type: ignore[assignment]
    compile_flash_attn_varlen_func_from_specs = None  # type: ignore[assignment]
    get_scheduler_metadata = None  # type: ignore[assignment]
    is_fa_version_supported = lambda *args, **kwargs: False  # type: ignore[assignment]
    fa_version_unsupported_reason = lambda *args, **kwargs: "vllm_flash_attn is not available"  # type: ignore[assignment]

if not FLASH_ATTN_AVAILABLE:
    flash_attn_varlen_func = None  # type: ignore[assignment]
    compile_flash_attn_varlen_func_from_specs = None  # type: ignore[assignment]
    get_scheduler_metadata = None  # type: ignore[assignment]
    if 'is_fa_version_supported' not in globals():
        is_fa_version_supported = lambda *args, **kwargs: False  # type: ignore[assignment]
    if 'fa_version_unsupported_reason' not in globals():
        fa_version_unsupported_reason = lambda *args, **kwargs: "vllm_flash_attn is not available"  # type: ignore[assignment]

__all__ = [
    "compile_flash_attn_varlen_func_from_specs",
    "fa_version_unsupported_reason",
    "flash_attn_varlen_func",
    "get_scheduler_metadata",
    "is_fa_version_supported",
    "FLASH_ATTN_AVAILABLE",
]
