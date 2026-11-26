import sys, time
sys.path.insert(0, "/tmp/omlxs/omlx-main")
import mlx.core as mx
from omlx.custom_kernels.macaq import fast, qmv

print("native:", fast.has_native(), "error:", fast.import_error())

def reference(x, t, w, sc):
    xf = mx.contiguous(x).reshape(-1, x.shape[-1])
    rot = mx.hadamard_transform(
        (xf * mx.contiguous(t.astype(xf.dtype))).reshape(xf.shape[0], xf.shape[1]//32, 32)
    ).reshape(xf.shape)
    return mx.quantized_matmul(rot, w, sc, transpose=True, group_size=32, bits=4, mode="mxfp4")

def bench(fn, reps=10):
    for _ in range(3): mx.eval(fn())
    s = time.perf_counter()
    for _ in range(reps): mx.eval(fn())
    return (time.perf_counter() - s) / reps

mx.random.seed(23)
for D, N, M in [(256,8,1),(256,64,1),(256,64,4),(512,128,1),(4096,64,1)]:
    W = mx.random.normal((N,D),dtype=mx.float32)*0.02
    q, sc = mx.quantize(W,bits=4,group_size=32,mode="mxfp4")
    t = mx.random.normal((D,),dtype=mx.bfloat16)*0+1.0
    x = mx.random.normal((M,D),dtype=mx.bfloat16)
    r = reference(x,t,q,sc)
    g = qmv(x,t,mx.contiguous(t.astype(x.dtype)) if False else q,sc) if False else qmv(x,t,q,sc)
    mx.eval(r,g)
    rel = float(mx.linalg.norm(g.astype(mx.float32)-r.astype(mx.float32))/mx.linalg.norm(r.astype(mx.float32)))
    print(f"correct {D}x{N}x{M}: rel={rel:.5f}")

D=N=4096
W=mx.random.normal((N,D),dtype=mx.float32)*0.02
q,sc=mx.quantize(W,bits=4,group_size=32,mode="mxfp4")
t=mx.random.normal((D,),dtype=mx.bfloat16)*0+1.0
print(f"{'M':>4} {'stock ms':>10} {'native ms':>10} {'speedup':>8}")
for M in [1,2,4,8,16,64]:
    x=mx.random.normal((M,D),dtype=mx.bfloat16)
    a=bench(lambda:reference(x,t,q,sc))
    b=bench(lambda:qmv(x,t,q,sc))
    print(f"{M:>4} {a*1000:>10.3f} {b*1000:>10.3f} {a/b:>8.2f}")
