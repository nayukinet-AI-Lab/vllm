# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

ToleranceSpec = dict[torch.dtype, dict[str, float]]

# Default tolerances for comparing IR op implementations against native.
# These are intentionally conservative (permissive) to avoid false failures
# across different hardware and kernel implementations. Ops that need tighter
# or looser bounds should use override_tolerance.
DEFAULT_TOLERANCES: ToleranceSpec = {}

_candidates = [
    (getattr(torch, "float64", None), {"atol": 1e-8, "rtol": 1e-8}),
    (getattr(torch, "float32", None), {"atol": 1e-5, "rtol": 1.3e-6}),
    (getattr(torch, "float16", None), {"atol": 1e-3, "rtol": 1e-3}),
    (getattr(torch, "bfloat16", None), {"atol": 1e-3, "rtol": 1.6e-2}),
    (getattr(torch, "float8_e4m3fn", None), {"atol": 1e-1, "rtol": 1e-1}),
    (getattr(torch, "float8_e5m2", None), {"atol": 2e-1, "rtol": 2e-1}),
    (getattr(torch, "int8", None), {"atol": 1, "rtol": 0}),
    (getattr(torch, "float4_e2m1fn_x2", None), {"atol": 3e-1, "rtol": 3e-1}),
]

for dt, tol in _candidates:
    if dt is not None:
        DEFAULT_TOLERANCES[dt] = tol
