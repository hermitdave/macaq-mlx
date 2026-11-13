import mlx.core as mx, time, math
mx.set_default_device(mx.gpu)

def bench(fn, n=50, warm=15):
    for _ in range(warm):
        mx.eval(fn())
    mx.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t) / n * 1000

print("mlx version:", getattr(mx, "__version__", "?"))
print()
print("=== batch sweep: n=m=4096 (Qwen3-8B q/k/v/o dims) ===")
print(f"{'B':>5} {'bf16':>8} {'qmm-mxfp4':>11} {'qqmm-mxfp4':>12} {'affine4':>9}")
for B in [1, 4, 16, 64, 256, 512, 1024]:
    x = mx.random.normal((B, 4096)).astype(mx.bfloat16)
    W = mx.random.normal((4096, 4096)).astype(mx.bfloat16)
    wq, sc = mx.quantize(W, group_size=32, mode='mxfp4')
    wa, sa, ba = mx.quantize(W, group_size=64, bits=4)
    t_bf = bench(lambda: x @ W.T)
    t_q = bench(lambda: mx.quantized_matmul(x, wq, sc, transpose=True, group_size=32, mode='mxfp4'))
    t_qq = bench(lambda: mx.qqmm(x, wq, scales=sc, group_size=32, bits=4, mode='mxfp4'))
    t_af = bench(lambda: mx.quantized_matmul(x, wa, sa, ba, transpose=True, group_size=64, bits=4))
    print(f"{B:>5} {t_bf:>8.3f} {t_q:>8.3f} ({t_q/t_bf:.2f}x) {t_qq:>8.3f} ({t_qq/t_bf:.2f}x) {t_af:>8.3f} ({t_af/t_bf:.2f}x)")

print()
print("=== MLP shape n=151936, m=4096 (Qwen3-8B gate/up) ===")
for B in [1, 16, 256]:
    x = mx.random.normal((B, 151936)).astype(mx.bfloat16)
    W = mx.random.normal((4096, 151936)).astype(mx.bfloat16)
    wq, sc = mx.quantize(W, group_size=32, mode='mxfp4')
    t_bf = bench(lambda: x @ W.T, n=20, warm=5)
    t_q = bench(lambda: mx.quantized_matmul(x, wq, sc, transpose=True, group_size=32, mode='mxfp4'), n=20, warm=5)
    t_qq = bench(lambda: mx.qqmm(x, wq, scales=sc, group_size=32, bits=4, mode='mxfp4'), n=20, warm=5)
    print(f"B={B}: bf16={t_bf:.3f}  qmm-mxfp4={t_q:.3f} ({t_q/t_bf:.2f}x)  qqmm-mxfp4={t_qq:.3f} ({t_qq/t_bf:.2f}x)")
