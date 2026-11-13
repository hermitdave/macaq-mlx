import mlx.core as mx, math, time
mx.set_default_device(mx.gpu)

def hadamard_matrix(n, dtype=mx.float32):
    H = mx.ones((1, 1), dtype=dtype)
    while H.shape[0] < n:
        H = mx.concatenate([mx.concatenate([H, H], axis=1),
                            mx.concatenate([H, -H], axis=1)], axis=0)
    return H / math.sqrt(n)

H32 = hadamard_matrix(32)

def had_matmul(x, H):
    *lead, n = x.shape
    return (x.reshape(lead + [n // 32, 32]) @ H.T).reshape(lead + [n])

def had_butterfly(x):
    *lead, n = x.shape
    y = x.reshape(lead + [n // 32, 32]).astype(mx.float32)
    h = 1
    while h < 32:
        y = y.reshape(lead + [n // 32, 32 // (2 * h), 2, h])
        a = y[..., 0, :]
        b = y[..., 1, :]
        y = mx.concatenate([a + b, a - b], axis=-1).reshape(lead + [n // 32, 32])
        h *= 2
    return (y.reshape(lead + [n]) / math.sqrt(32.0)).astype(x.dtype)

xt = mx.random.normal((4, 128))
print("butterfly vs matmul-hadamard max err:", float(mx.max(mx.abs(had_butterfly(xt) - had_matmul(xt, H32)))))
# also verify matmul-hadamard == block-diagonal reference
Hfull = mx.zeros((128, 128))
for i in range(4):
    Hfull[i*32:(i+1)*32, i*32:(i+1)*32] = H32
print("matmul-hadamard vs block-diag max err:", float(mx.max(mx.abs(had_matmul(xt, H32) - xt @ Hfull))))


def bench(fn, n=100, warm=20):
    for _ in range(warm):
        mx.eval(fn())
    mx.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t) / n * 1000


print()
print("=== chain (32 layers, B=1, n=m=4096, Qwen3-8B-ish) ===")
N = 32
Ws = [mx.random.normal((4096, 4096)).astype(mx.bfloat16) for _ in range(N)]
Wsq = [mx.quantize(w, group_size=32, mode='mxfp4') for w in Ws]
tvec = [mx.random.uniform(0.5, 2.0, (4096,)) for _ in range(N)]
x0 = mx.random.normal((1, 4096)).astype(mx.bfloat16)

def chain_base(x):
    for wq, sc in Wsq:
        x = mx.qqmm(x, wq, scales=sc, group_size=32, bits=4, mode='mxfp4')
    return x

def chain_matmul(x):
    for (wq, sc), t in zip(Wsq, tvec):
        x = mx.qqmm(had_matmul(x * t, H32), wq, scales=sc, group_size=32, bits=4, mode='mxfp4')
    return x

def chain_butterfly(x):
    for (wq, sc), t in zip(Wsq, tvec):
        x = mx.qqmm(had_butterfly(x * t), wq, scales=sc, group_size=32, bits=4, mode='mxfp4')
    return x

b = bench(lambda: chain_base(x0), n=50, warm=10)
m1 = bench(lambda: chain_matmul(x0), n=50, warm=10)
m2 = bench(lambda: chain_butterfly(x0), n=50, warm=10)
print(f"mxfp4 only:             {b:.3f} ms")
print(f"+ scale+had(matmul):    {m1:.3f} ms  (transform marginal {100*(m1-b)/b:.0f}%)")
print(f"+ scale+had(butterfly): {m2:.3f} ms  (transform marginal {100*(m2-b)/b:.0f}%)")

print()
print("=== MLX mxfp4 grid conformance ===")
w = mx.random.normal((64, 4096)).astype(mx.float32) * mx.random.uniform(0.01, 100.0, (64, 1))
wq, sc = mx.quantize(w, group_size=32, mode='mxfp4')
wd = mx.dequantize(wq, sc, group_size=32, mode='mxfp4')
vals = wd.reshape(-1)
bad = 0
for v in vals:
    fv = float(v)
    if fv == 0.0:
        continue
    e = math.floor(math.log2(abs(fv)))
    m = abs(fv) / (2 ** e)
    if not any(abs(m - c) < 1e-4 for c in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]):
        bad += 1
print("values off E2M1 grid:", bad, "/", vals.size)
print("scale dtype/values sample:", sc.dtype, sorted(set(int(s) for s in sc.reshape(-1)))[:10])
