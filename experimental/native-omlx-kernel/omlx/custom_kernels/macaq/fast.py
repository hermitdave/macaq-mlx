"""Python wrapper for the native MACaQ W4A16 kernel.

The extension is optional: when absent or ABI-incompatible, ``qmv`` raises and
the checkpoint shim falls back to the stock path.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from .._loader import load_extension, native_stream_kwargs

_ext, _IMPORT_ERROR = load_extension(__package__, Path(__file__).parent)


def has_native() -> bool:
    return _ext is not None


def is_native_available() -> bool:
    return has_native()


def import_error() -> Exception | None:
    return _IMPORT_ERROR


def _require():
    if _ext is None:
        raise RuntimeError(f"MACaQ native extension unavailable: {_IMPORT_ERROR}")
    return _ext


def qmv(x: mx.array, t: mx.array, w: mx.array, scales: mx.array, stream=None) -> mx.array:
    """Fused x*t + block-Hadamard + MXFP4 GEMV; x [M,K] -> [M,N]."""
    return _require().qmv(x, t, w, scales, int(w.shape[0]))
