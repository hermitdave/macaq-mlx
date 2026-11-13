import mlx.core as mx, math
mx.set_default_device(mx.gpu)

SG = mx.array([-6., -4., -3., -2., -1.5, -1., -0.5, 0.,
               0.5, 1., 1.5, 2., 3., 4., 6.], mx.float32)


def sidx_to_e2m1(sidx):
    """MACaQ signed-grid index (0..14) -> raw OCP E2M1 4-bit pattern."""
    return mx.where(sidx >= 7, sidx - 7, 15 - sidx).astype(mx.uint32)


def pack_mxfp4(sidx, log2_amax, block=32):
    """Pack into MLX's native mxfp4 layout (flat 4-bit codes + e8m0 scales)."""
    out, inp = sidx.shape
    nib = sidx_to_e2m1(sidx)
    shifts = mx.power(mx.array(2, mx.uint32), mx.arange(0, 32, 4, dtype=mx.uint32))
    packed = mx.sum(mx.unflatten(nib, -1, (-1, 8)) * shifts, axis=-1)
    scales = (log2_amax.squeeze(-1) + 127.0).astype(mx.uint8)
    return packed, scales


print("=== nibble mapping: MACaQ sidx -> MLX nibble -> value ===")
ok = True
for s in range(15):
    sc = mx.array([[127]], mx.uint8)
    nib = int(sidx_to_e2m1(mx.array([s]))[0])
    pk = mx.array([[nib, 0, 0, 0]], mx.uint32)
    v = float(mx.dequantize(pk, sc, group_size=32, mode='mxfp4')[0, 0])
    match = abs(v - float(SG[s])) < 1e-6
    ok &= match
    print(f"  sidx={s:2d} -> nibble={nib:2d} (0x{nib:X}) -> dequant={v:5.1f}  expected={float(SG[s]):5.1f}  {'OK' if match else 'MISMATCH'}")
print("all mappings OK:", ok)

print()
print("=== full round-trip: our MACaQ codes -> MLX packed -> MLX dequant ===")
mx.random.seed(3)
W = mx.random.normal((8, 256)) * mx.random.uniform(0.01, 5.0, (8, 1))
out, inp = W.shape
nblk = inp // 32
Mg = W.reshape(out, nblk, 32)
amax = mx.maximum(mx.max(mx.abs(Mg), axis=-1, keepdims=True), 1e-12)
log2_amax = mx.floor(mx.log2(amax)) - 2.0        # MACaQ RTN exponent rule
scale = mx.power(2.0, log2_amax)
abs_s = mx.abs(Mg) / scale
code = mx.where(abs_s > 5.0, 7,
       mx.where(abs_s > 3.5, 6,
       mx.where(abs_s > 2.5, 5,
       mx.where(abs_s > 1.75, 4,
       mx.where(abs_s > 1.25, 3,
       mx.where(abs_s > 0.75, 2,
       mx.where(abs_s > 0.25, 1, 0.0))))))).astype(mx.int32)
neg = Mg < 0
sidx = mx.where(code == 0, 7, mx.where(neg, 7 - code, 7 + code)).astype(mx.int32)
ours = (mx.take(SG, sidx) * scale).reshape(out, inp)

packed, scales = pack_mxfp4(sidx.reshape(out, inp), log2_amax)
back = mx.dequantize(packed, scales, group_size=32, mode='mxfp4')
print("packed shape", packed.shape, "scales shape", scales.shape)
print("max |ours - mlx_dequant| =", float(mx.max(mx.abs(ours - back))))

print()
print("=== MLX kernels consuming our custom-packed mxfp4 weights ===")
x = mx.random.normal((4, 256))
ref = x @ ours.T          # (4, 8)
y_qmm = mx.quantized_matmul(x, packed, scales, transpose=True, group_size=32, mode='mxfp4')
y_qq = mx.qqmm(x, packed, scales=scales, group_size=32, bits=4, mode='mxfp4')
print("quantized_matmul rel err vs our dequant:", float(mx.mean((y_qmm - ref) ** 2) / mx.mean(ref ** 2)))
print("qqmm (W4A4)           rel err vs our dequant:", float(mx.mean((y_qq - ref) ** 2) / mx.mean(ref ** 2)))
