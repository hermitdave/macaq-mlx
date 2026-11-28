"""Route B GEMV correctness check (performance is deliberately not asserted)."""
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from macaq_model import fused_rotate_gemv  # noqa: E402


def test_gemv_matches_hadamard_then_matmul():
    mx.random.seed(23)
    for dim, out_dims, rows in ((64, 4, 1), (256, 64, 1), (256, 64, 4)):
        w = mx.random.normal((out_dims, dim), dtype=mx.float32) * 0.02
        q, scales = mx.quantize(w, bits=4, group_size=32, mode="mxfp4")
        t = mx.random.normal((dim,), dtype=mx.bfloat16) * 0 + 1.0
        x = mx.random.normal((rows, dim), dtype=mx.bfloat16)

        rotated = mx.hadamard_transform(
            (x * t.astype(x.dtype)).reshape(rows, dim // 32, 32)
        ).reshape(rows, dim)
        ref = mx.quantized_matmul(
            rotated, q, scales, transpose=True, group_size=32, bits=4,
            mode="mxfp4")
        got = fused_rotate_gemv(x, t, q, scales, out_dims)
        mx.eval(ref, got)
        rel = float(mx.linalg.norm(got.astype(mx.float32) - ref.astype(mx.float32)) /
                    mx.linalg.norm(ref.astype(mx.float32)))
        assert rel < 0.01, (dim, out_dims, rows, rel)


if __name__ == "__main__":
    test_gemv_matches_hadamard_then_matmul()
    print("GEMV kernel correctness passed (performance not asserted)")
