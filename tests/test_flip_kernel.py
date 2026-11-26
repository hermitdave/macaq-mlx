"""Fused flip kernel equivalence/speed smoke test."""
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from macaq_flip_kernel import flip_refine_metal
from macaq_hadamard import rotate_block, rotate_gram, hessian_objective
from macaq_refine import flip_refine, screen_exponents, unpack_mxfp4


def make_case(rows=256, cols=512, tokens=256, seed=23):
    mx.random.seed(seed)
    W = mx.random.normal((rows, cols), dtype=mx.float32) * 0.02
    t = mx.sqrt(mx.mean((W - mx.mean(W, axis=0)) ** 2, axis=0)).astype(mx.bfloat16)
    Wp = rotate_block(W / t.astype(mx.float32)[None, :])
    X = mx.random.normal((tokens, cols), dtype=mx.float32) / (tokens ** 0.5)
    H = X.T @ X
    Hp = rotate_gram(H, t)
    Hp = Hp + 1.0 * mx.mean(mx.diagonal(Hp)) * mx.eye(cols)
    q, sc = mx.quantize(Wp, bits=4, group_size=32, mode="mxfp4")
    sidx, log2s = unpack_mxfp4(q, sc)
    sidx, log2s = screen_exponents(Wp, sidx, log2s, mx.diagonal(Hp), offsets=(-1, 0, 1))
    mx.eval(Wp, sidx, log2s, Hp)
    return Wp, sidx, log2s, Hp


def test_fused_flip_matches_reference():
    Wp, sidx, log2s, Hp = make_case()
    pure, sp = flip_refine(sidx, log2s, Wp, Hp, 32)
    fast, sf = flip_refine_metal(sidx, log2s, Wp, Hp, 32)
    mx.eval(pure, fast)

    diff = int(mx.sum(mx.not_equal(pure, fast)).item())
    frac = diff / pure.size
    op = float(hessian_objective(Wp, pure, log2s, Hp))
    of = float(hessian_objective(Wp, fast, log2s, Hp))
    rel = abs(op - of) / max(abs(op), 1e-12)
    assert sp == sf, (sp, sf)
    assert frac < 1e-4, (diff, pure.size)
    assert rel < 1e-5, (op, of, rel)


if __name__ == "__main__":
    test_fused_flip_matches_reference()
    print("fused flip equivalence passed")
