# Native MACaQ kernel — working plan

Status: **native build blocked locally; design validated only via runtime Metal
kernel and measured slower. No PR.** The C++/Metal scaffold exists in
`/tmp/omlxs/omlx-main/omlx/custom_kernels/macaq/` but cannot be compiled here
(no `metal` compiler: `xcrun --find metal` fails, only Command Line Tools are
installed). Do not open an upstream PR on the current evidence.

## Decision (agreed)

- Build a **native C++/Metal custom kernel package** in oMLX:
  `omlx/custom_kernels/macaq/`.
- Target our existing **W4A16** checkpoint: MXFP4 E8M0 weights, BF16 activations
  after per-channel scale + 32-block Hadamard. Do **not** move to W4A4; that
  changes accuracy and needs a new published checkpoint.
- Use upstream Huawei CSL MACaQ Triton kernels as **design references only**:
  - `vllm/triton_mxfp4_moe_decode.py` — tiled K loop, multi-row activation
    reuse, fused epilogue, no token sort: the pattern Route B lacked.
  - `vllm/triton_mxfp4_dense.py::_rot_scale_quant_kernel` — fused pre-scale +
    Hadamard (+ MXFP4 activation quantization, which we do NOT need for W4A16).
  - `llamacpp/macaq_triton_kernels.py::_butterfly32` — same butterfly we already
    implemented.
- Respect Apache-2.0 attribution to Huawei CSL in source headers and PR.

## Kernel contract

Python wrapper `omlx/custom_kernels/macaq.fast`:

```python
qmv(x, t, w, scales, out_dims) -> [M, out_dims]  # bf16
```

- `x`: `[M,K]`, bf16, row-contiguous.
- `t`: `[K]`, bf16 per-channel pre-scale (the checkpoint's `t` tensor).
- `w`: `[N,K/8]`, uint32, MLX `mxfp4` nibble packing.
- `scales`: `[N,K/32]`, uint8 E8M0, exponent = byte - 127.
- Fuses: `x*t`, 32-block Hadamard (normalized by 1/sqrt(32)),
  E2M1 nibble decode with E8M0 block scale, fp32 dot product, bf16 store.
- Modes: decode path for M<=8; prefill keeps Route A + stock `quantized_matmul`.

Mandatory fallback: if `_ext is None`, `macaq_model.py` keeps using the current
Route A / stock path. No checkpoint may require the native kernel to load.

## What actually happened (this session)

1. Wrote this plan and scaffolded a native package in the local checkout:
   `omlx/custom_kernels/macaq/{__init__.py,fast.py,csrc/{macaq.h,macaq.cpp,
   macaq.metal,bindings.cpp}}` plus `setup_macaq.py`, using oQh's
   MetalExtension pattern and Huawei CSL's tiled decode design.
2. Build attempt with the bundled CPython 3.11 failed: CMake could not find
   Python development headers; overriding `Python_EXECUTABLE` to Homebrew
   Python 3.11 fixed that, then the build failed at `xcrun: unable to find
   utility "metal"` — no Metal compiler on this machine. The `.so`/`.metallib`
   were never produced, so **no native kernel was executed**.
3. Validated the same design through `mx.fast.metal_kernel` instead:
   - `simd_shuffle_xor` is a no-op in runtime kernels, so the simd version
     could not be exercised.
   - Portable threadgroup-memory butterfly version was numerically correct
     (rel 0.003–0.004) but slower: at D=N=4096, M=1 it took **0.67–1.42 ms vs
     0.40–0.43 ms stock (0.30–0.60×)**. R=4/R=8 multi-row variants were worse
     (0.15–0.27×).
4. Conclusion: no evidence of a win. The uncompiled native version could be
   faster because it can use real simd shuffles, but that is speculation until
   built on a machine with Xcode's Metal toolchain. Keep the scaffold as a
   starting point; do not PR.

## Re-open criteria

- Build on a machine with `metal` available (install Xcode or point `xcrun` at
  a toolchain that ships `metal`), then rerun gates 1–5 above.
- If the native simd version still loses at M=1, stop; the kernel is not worth
  an upstream PR.

## Validation gates (all before PR)

1. **Build** with the same interpreter that runs oMLX (bundled CPython 3.11),
   in-place in the checkout. Confirm `has_native() == True`.
2. **Correctness** on random and real checkpoint layers:
   - relative error < 0.01 vs `hadamard + quantized_matmul` for M=1,2,4,8,16,64
     and K/N shapes 256/64, 4096/4096, plus the real Qwen3.5 layer geometry.
   - full PPL/KL/top-1 unchanged within noise.
3. **Microbenchmark** at D=N=4096, M=1,2,4,8,16,64,256:
   - report native ms, stock ms, Route A ms, speedup curve and crossover M.
   - must beat stock by >=10% at M=1 to proceed to end-to-end.
4. **End-to-end oMLX A/B from the patched checkout**:
   - baseline (`fused="none"`), Route A (`"rotate"`), native (`"native"`).
   - same checkpoint, same prompts/settings, `mtp_enabled:false`, loaded model,
     warmup, unique prompt tags to defeat prefix cache, repeated trials.
   - report decode tok/s, prompt tok/s, TTFT, peak memory; re-run baseline after
     native to detect scheduler drift.
   - >=10% decode win and no quality/memory regression required.
5. **Integration**: log proof the native kernel is invoked; fallback works when
   the package is absent; no effect on models without the config flag.
6. **Only then** open upstream PR with the measurements attached.

## Environment / pitfalls

- oMLX checkout: `/tmp/omlxs/omlx-main`.
- Bundled interpreter: `/Applications/oMLX.app/Contents/Resources/Python/cpython-3.11/bin/python3`.
- Bundled site-packages: `/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages`.
- Use `PYTHONPATH=$SRC:$SITE` for any server run; source tree first.
- Stop the app's `omlx-server` before launching the source server; restore with
  `open -a oMLX` afterwards. Do not compare a patched source server against the
  installed 0.8.3 app — baseline must be the same unpatched checkout.
- MLX E8M0 exponent is `ceil(log2(amax/6))`; upstream MACaQ uses
  `floor(log2(amax))-2`. Keep the MLX convention for our weight scales.
- Keep working folders; do not clean scratch caches.
