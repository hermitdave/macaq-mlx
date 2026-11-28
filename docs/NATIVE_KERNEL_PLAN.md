# Native MACaQ kernel — working plan

Status: **two native kernel tilings built and executed locally; both numerically
correct, both slower than stock. Stopped — no PR.** The C++/Metal package lives
in `experimental/native-omlx-kernel/` and is NOT installed into the app bundle.
Do not open an upstream PR on the current evidence.

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

1. Scaffolded the native package in the local checkout:
   `omlx/custom_kernels/macaq/{__init__.py,fast.py,csrc/{macaq.h,macaq.cpp,
   macaq.metal,bindings.cpp}}` plus `setup_macaq.py`, following oQh's
   MetalExtension pattern and Huawei CSL's tiled decode design.
2. Build prerequisites resolved: Xcode was installed but not selected; using
   `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`, downloading the
   Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`), overriding
   `Python_EXECUTABLE` to Homebrew Python 3.11 and installing `nanobind==3.1.0`
   got a real `_ext.cpython-311-darwin.so` and `omlx_macaq_kernels.metallib`.
   Several C++ bugs were fixed (missing `mlx/utils.h`, `astype` as a free
   function, scale shape check `*4` → `/4`, kernel-name mismatch, and a
   threadgroup reduction race where 32 lanes wrote the same `red` slot).
3. Correctness gate passed with the app-bundled CPython 3.11: relative error
   **0.003–0.004** vs `hadamard + quantized_matmul` for M=1/4 and K/N shapes
   256/8, 256/64, 512/128, 4096/64, and for the D=N=4096 sweep up to M=64.
4. Microbenchmark gate **failed**. At D=N=4096:

   | M | Stock | Native (4 simdgroups) | Speedup |
   |---:|---:|---:|---:|
   | 1 | 0.423 ms | 0.610 ms | **0.69×** |
   | 2 | 0.247 ms | 1.052 ms | 0.23× |
   | 4 | 0.388 ms | 0.945 ms | 0.41× |
   | 8 | 0.316 ms | 1.173 ms | 0.27× |
   | 16 | 0.491 ms | 1.982 ms | 0.25× |
   | 64 | 0.554 ms | 6.827 ms | 0.08× |

   An 8-simdgroup variant was also tried and was worse at M=1 (0.691 ms).
   Root cause: one threadgroup per output row recomputes the Hadamard rotation
   for every output row, while the stock path computes it once and reuses it
   across all N rows. The K-group parallelism does not repay that lost reuse.
5. Conclusion: correct but not competitive. No end-to-end oMLX A/B was run
   because the microbenchmark gate failed. No PR.

## Second attempt: activation reuse (this session)

Implemented two further tilings in the native package:

- **v2** (`macaq_qmv_reuse`): 8 output rows per threadgroup, 2 simdgroups,
  serial K loop; rotate once per activation row and reuse across 8 rows.
  Correct (rel 0.003–0.004), but 0.43× at M=1 (0.551 ms vs 0.237 ms stock).
- **v3** (`macaq_qmv_reuse_k4`): 4 output rows per threadgroup, 4 simdgroups
  partition K (G/4 iterations), activation reused across 4 rows, one
  threadgroup reduction at the end. Fixed a `simd_sum` bug where only lane 0
  participated. Correct (rel 0.003–0.004), but still 0.77× at M=1
  (0.486 ms vs 0.372 ms stock) and 0.42–0.52× for M=2–8.

Both failed the ≥10% decode-win gate, so no end-to-end oMLX A/B was run and no
PR was opened. The stored experimental source is the v3 variant.

Conclusion: activation reuse closes the 4× redundant-rotation gap but the
kernel remains slower than MLX's tuned `quantized_matmul` at every tested
decode size. Further GEMM kernel work would require a format or precision
change (W4A4 + `mx.qqmm`) or an oQh-format rewrite, neither of which is in
scope for this checkpoint.

## Re-open criteria

- Activation reuse has now been tried (v2 and v3) and did not close the gap.
  Do not repeat it.
- Re-open only for a materially different contract: W4A4 with `mx.qqmm` and a
  new quality-validated checkpoint, or an oQh-format model that can use oMLX's
  native kernels. Neither is part of the current rotated W4A16 checkpoint.

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
