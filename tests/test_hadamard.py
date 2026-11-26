"""Synthetic checks for the Hadamard MACaQ path (no model download needed)."""
import math
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from macaq_hadamard import channel_scale, quantize_rotated, rotate_block
from macaq_model import BLOCK, MACaQLinear


def test_rotate_block_orthonormal():
    x = mx.random.normal((4, 128))
    y = rotate_block(rotate_block(x))
    err = float(mx.max(mx.abs(y - x)))
    assert err < 1e-4, err


def test_rotated_linear_beats_plain_mxfp4():
    mx.random.seed(7)
    dim = 256
    W = mx.random.normal((dim, dim), dtype=mx.float32) * 0.02
    W = W * mx.random.uniform(0.2, 8.0, (dim, 1))
    X = mx.random.normal((64, dim), dtype=mx.bfloat16)
    H = X.astype(mx.float32).T @ X.astype(mx.float32)
    t = channel_scale(W, H, 64, 1.0)
    packed, scales, _ = quantize_rotated(W, H, t, 64, 1.0, (-1, 0, 1))

    layer = MACaQLinear(dim, dim, block=BLOCK)
    layer.weight = packed
    layer.scales = scales
    layer.t = t
    ref = X @ W.T.astype(mx.bfloat16)
    err_rot = float(mx.linalg.norm(layer(X) - ref) / mx.linalg.norm(ref))

    q, sc = mx.quantize(W, bits=4, group_size=32, mode="mxfp4")
    y_plain = mx.quantized_matmul(
        X, q, sc, transpose=True, group_size=32, bits=4, mode="mxfp4")
    err_plain = float(mx.linalg.norm(y_plain - ref) / mx.linalg.norm(ref))

    assert err_rot < err_plain * 0.8, (err_rot, err_plain)


if __name__ == "__main__":
    test_rotate_block_orthonormal()
    test_rotated_linear_beats_plain_mxfp4()
    print("hadamard tests passed")
