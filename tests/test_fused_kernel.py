"""Route A: fused scale + 32-block Hadamard kernel checks."""
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from macaq_model import MACaQLinear, fused_scale_hadamard  # noqa: E402


def test_fused_matches_hadamard():
    mx.random.seed(11)
    for rows, dim in ((1, 4096), (4, 12288), (16, 4096)):
        x = mx.random.normal((rows, dim), dtype=mx.bfloat16)
        t = mx.random.normal((dim,), dtype=mx.bfloat16) * 0.02 + 1.0
        got = fused_scale_hadamard(x, t)
        ref = mx.hadamard_transform(
            (x * t.astype(x.dtype)).reshape(rows, dim // 32, 32)
        ).reshape(rows, dim)
        mx.eval(got, ref)
        diff = mx.max(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)))
        # bf16 rounding differs by at most one ULP of the largest block value
        assert float(diff) <= 0.05, (rows, dim, float(diff))


def test_layer_fused_matches_stock():
    mx.random.seed(17)
    dim = 2048
    W = mx.random.normal((dim, dim), dtype=mx.float32) * 0.02
    q, scales = mx.quantize(W, bits=4, group_size=32, mode="mxfp4")
    t = mx.random.normal((dim,), dtype=mx.bfloat16) * 0 + 1.0

    def layer(fused):
        mod = MACaQLinear(dim, dim, fused=fused)
        mod.weight = q
        mod.scales = scales
        mod.t = t
        return mod

    x = mx.random.normal((8, dim), dtype=mx.bfloat16)
    a = layer(False)(x)
    b = layer(True)(x)
    mx.eval(a, b)
    rel = float(mx.linalg.norm(a.astype(mx.float32) - b.astype(mx.float32)) /
                mx.linalg.norm(a.astype(mx.float32)))
    assert rel < 0.01, rel


if __name__ == "__main__":
    test_fused_matches_hadamard()
    test_layer_fused_matches_stock()
    print("fused kernel tests passed")
