import sys, time, json
sys.path.insert(0, "/Users/hermit.dave/Projects/ml/macaq-mlx")
import mlx.core as mx
from macaq_hadamard import rotate_block, rotate_gram, hessian_objective
from macaq_refine import unpack_mxfp4, screen_exponents, flip_refine
from macaq_flip_kernel import flip_refine_metal

MODEL = "/Users/hermit.dave/Projects/ml/models/Spark-X2.6-1.7B-BF16"
INDEX = json.load(open(MODEL + "/model.safetensors.index.json"))["weight_map"]

CASES = [
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.1.mlp.up_proj.weight",
]
N_TOKENS = 512
ITERS = 32


def make_case(key):
    shard = INDEX[key]
    W = mx.load(f"{MODEL}/{shard}")[key].astype(mx.float32)
    out_d, in_d = W.shape
    # D1 channel scale, alpha=1.0, same as our W4A16 default.
    t = mx.sqrt(mx.mean((W - mx.mean(W, axis=0)) ** 2, axis=0)).astype(mx.bfloat16)
    Wp = rotate_block(W / t.astype(mx.float32)[None, :])

    mx.random.seed(23)
    X = mx.random.normal((N_TOKENS, in_d), dtype=mx.float32) / (N_TOKENS ** 0.5)
    H = X.T @ X
    Hp = rotate_gram(H, t)
    Hp = Hp + 1.0 * mx.mean(mx.diagonal(Hp)) * mx.eye(in_d)

    q, sc = mx.quantize(Wp, bits=4, group_size=32, mode="mxfp4")
    sidx, log2s = unpack_mxfp4(q, sc)
    sidx, log2s = screen_exponents(Wp, sidx, log2s, mx.diagonal(Hp), offsets=(-1, 0, 1))
    mx.eval(Wp, sidx, log2s, Hp)
    return Wp, sidx, log2s, Hp, (out_d, in_d)


def bench(fn, reps=3):
    for _ in range(1): mx.eval(fn())
    s = time.perf_counter()
    out = fn()
    mx.eval(out)
    return time.perf_counter() - s, out


for key in CASES:
    Wp, sidx, log2s, Hp, shape = make_case(key)
    base_obj = float(hessian_objective(Wp, sidx, log2s, Hp))

    t0, pure = bench(lambda: flip_refine(sidx, log2s, Wp, Hp, ITERS))
    pure_idx, pure_sweeps = pure
    t1, fast = bench(lambda: flip_refine_metal(sidx, log2s, Wp, Hp, ITERS))
    fast_idx, fast_sweeps = fast

    pure_obj = float(hessian_objective(Wp, pure_idx, log2s, Hp))
    fast_obj = float(hessian_objective(Wp, fast_idx, log2s, Hp))
    eq = bool(mx.array_equal(pure_idx, fast_idx).item())
    print(f"{key} shape={shape} iters={ITERS}")
    diff=int(mx.sum(mx.not_equal(pure_idx, fast_idx)).item())
    dq=float(mx.max(mx.abs(mx.take(mx.array([-6.,-4.,-3.,-2.,-1.5,-1.,-0.5,0.,0.5,1.,1.5,2.,3.,4.,6.],mx.float32), pure_idx)-mx.take(mx.array([-6.,-4.,-3.,-2.,-1.5,-1.,-0.5,0.,0.5,1.,1.5,2.,3.,4.,6.],mx.float32), fast_idx))))
    print(f"  sweeps pure={pure_sweeps} metal={fast_sweeps} codes_equal={eq} diff_codes={diff}/{sidx.size} max_grid_diff={dq}")
    print(f"  objective base={base_obj:.12e} pure={pure_obj:.12e} metal={fast_obj:.12e} delta={pure_obj-fast_obj:.3e}")
    print(f"  time pure={t0:.3f}s metal={t1:.3f}s speedup={t0/t1:.2f}x")
