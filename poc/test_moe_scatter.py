"""Can we scatter tokens into per-expert slots and batch the Gram matrices?"""
import mlx.core as mx
import numpy as np

mx.set_default_device(mx.gpu)
E, in_d, N = 8, 32, 256
n_max = 48

xf = mx.random.normal((N, in_d))
idx = mx.random.randint(0, E, (N,))

counts = mx.sum((idx[None, :] == mx.arange(E)[:, None]).astype(mx.int32), axis=1)
offsets = mx.cumsum(counts) - counts

# sort by expert so each expert's tokens are contiguous
order = mx.argsort(idx)
idx_s = idx[order]
xf_s = xf[order]

offsets_rep = mx.take(offsets, idx_s)
rank = mx.arange(N) - offsets_rep
print("counts:", [int(c) for c in counts], "max", int(mx.max(counts)))
print("rank min/max:", int(mx.min(rank)), int(mx.max(rank)))

Xp = mx.zeros((E, n_max, in_d))
Xp = Xp.at[idx_s, rank].add(xf_s)
H = Xp.transpose(0, 2, 1) @ Xp
print("H shape:", H.shape)

xf_n = np.array(xf_s)
idx_n = np.array(idx_s)
ok = True
for e in range(E):
    xe = xf_n[idx_n == e]
    ref = xe.T @ xe
    err = np.abs(np.array(H[e]) - ref).max()
    ok &= err < 1e-3
    print(f"  expert {e}: n={len(xe):3d} max|H_got - H_ref| = {err:.3e}")
print("\nALL CORRECT:", ok)

# timing at OLMoE scale
import time
E, in_d, T, k = 64, 2048, 2048, 8
N = T * k
xf = mx.random.normal((N, in_d))
idx = mx.random.randint(0, E, (N,))
counts = mx.sum((idx[None, :] == mx.arange(E)[:, None]).astype(mx.int32), axis=1)
offsets = mx.cumsum(counts) - counts
order = mx.argsort(idx)
idx_s = idx[order]
xf_s = xf[order]
rank = mx.arange(N) - mx.take(offsets, idx_s)
n_max = int(mx.max(counts))
mx.eval(n_max)
print(f"\nOLMoE-scale: E={E} in={in_d} N={N} n_max={n_max}")


def bench(fn, n=5):
    for _ in range(2):
        mx.eval(fn())
    mx.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t) / n * 1000


def pipeline():
    Xp = mx.zeros((E, n_max, in_d))
    Xp = Xp.at[idx_s, rank].add(xf_s)
    return Xp.transpose(0, 2, 1) @ Xp


print(f"scatter+Gram: {bench(pipeline):.1f} ms per batch")
print(f"  -> x100 batches = {bench(pipeline)*100/1000:.1f} s")
