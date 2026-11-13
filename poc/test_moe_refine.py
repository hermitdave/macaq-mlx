"""The batched MoE refiner must agree with the validated per-layer dense path."""
import numpy as np
import mlx.core as mx
import sys

sys.path.insert(0, "/Users/hermit.dave/Projects/ml/macaq-mlx")
from macaq_refine import flip_refine, flip_refine_b, screen_exponents, hessian_objective

mx.set_default_device(mx.gpu)
E, out_d, in_d = 4, 24, 128
rng = np.random.default_rng(0)

W = mx.array(rng.standard_normal((E, out_d, in_d)).astype(np.float32) * 0.1)
# distinct SPD Hessian per expert
H = mx.array(np.stack([
    (np.eye(in_d) * (1.0 + e) + 0.05 * rng.standard_normal((in_d, in_d))).astype(np.float32)
    for e in range(E)
]))
H = H @ H.transpose(0, 2, 1) + mx.eye(in_d) * 0.5

sidx = mx.array(rng.integers(0, 15, (E, out_d, in_d)).astype(np.int32))
log2s = mx.array(rng.integers(-2, 3, (E, out_d, in_d // 32)).astype(np.int32))

# ---- screen ----
s_b, l_b = screen_exponents(W, sidx, log2s, mx.diagonal(H, axis1=1, axis2=2))
ok = True
for e in range(E):
    s_e, l_e = screen_exponents(W[e], sidx[e], log2s[e], mx.diagonal(H[e]))
    d = float(mx.max(mx.abs(s_b[e] - s_e))) + float(mx.max(mx.abs(l_b[e] - l_e)))
    ok &= d == 0.0
print(f"screen matches per-expert: {ok}")

# ---- flip ----
sidx_b, sweeps_b = flip_refine_b(sidx, log2s, W, H, 60)
all_ok = True
for e in range(E):
    s_e, sw_e = flip_refine(sidx[e], log2s[e], W[e], H[e], 60)
    d = float(mx.max(mx.abs(sidx_b[e].astype(mx.int32) - s_e.astype(mx.int32))))
    obj_b = hessian_objective(W[e], sidx_b[e], log2s[e], H[e])
    obj_e = hessian_objective(W[e], s_e, log2s[e], H[e])
    all_ok &= d == 0.0
    print(f"  expert {e}: sidx diff={d}  obj batched={obj_b:.6e} per-expert={obj_e:.6e}")
print(f"\nflip matches per-expert: {all_ok}")
print(f"sweeps: batched={sweeps_b}")
print(f"objective: before={hessian_objective(W, sidx, log2s, H):.6e} "
      f"after={hessian_objective(W, sidx_b, log2s, H):.6e}")
