"""MACaQ for MLX: refine native mxfp4 codes against a BF16 teacher.

No runtime change, no format change, no online transforms. For every quantized
linear layer we:
  1. unpack MLX's E2M1 nibbles + E8M0 block scales,
  2. re-screen the block exponents with the activation Hessian,
  3. run Flip-Refinement (greedy coordinate descent on the E2M1 codes),
  4. repack and write back.

Usage:
  python macaq_refine.py --model LFM3-1.7B-MLX-mxfp4 \
      --bf16 LFM3-1.7B-MLX-BF16 --output LFM3-1.7B-MLX-mxfp4-refined
"""
import argparse
import json
import math
import re
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from datasets import load_dataset
from mlx.utils import tree_flatten
from mlx_lm.utils import load, load_config, save

BLOCK = 32
SG = mx.array([-6., -4., -3., -2., -1.5, -1., -0.5, 0.,
               0.5, 1., 1.5, 2., 3., 4., 6.], mx.float32)
SHIFTS = mx.power(mx.array(2, mx.uint32), mx.arange(0, 32, 4, dtype=mx.uint32))
INF = float("inf")


# ---------------------------------------------------------------- unpack/pack
# The packer multiplies nibbles by powers of two (1, 16, 256, ...). To unpack
# you must shift by the EXPONENTS (0, 4, 8, ...), not by those powers. Getting
# this wrong fails silently: a shift count >= 32 wraps modulo 32 on x86/ARM, so
# `x >> 256` evaluates as `x >> 0` and returns the unshifted value, not zero.
SHIFT_AMT = mx.arange(0, 32, 4, dtype=mx.uint32)


def unpack_mxfp4(packed, scales):
    """MLX packed uint32 + e8m0 scales -> (signed-grid index, per-block log2 scale)."""
    out_d, pw = packed.shape
    in_d = pw * 8
    nib = ((packed[:, :, None] >> SHIFT_AMT[None, None, :]) & 0xF).reshape(out_d, in_d)
    nib = nib.astype(mx.int32)
    sidx = mx.where(nib >= 8, 15 - nib, nib + 7)
    log2s = scales.astype(mx.int32) - 127
    return sidx, log2s


def pack_mxfp4(sidx, log2s):
    nib = mx.where(sidx >= 7, sidx - 7, 15 - sidx).astype(mx.uint32)
    packed = mx.sum(mx.unflatten(nib, -1, (-1, 8)) * SHIFTS, axis=-1)
    scales = (log2s + 127).astype(mx.uint8)
    return packed, scales


def scale_elem(log2s, in_d):
    out_d, nblk = log2s.shape
    return mx.broadcast_to(mx.power(2.0, log2s)[:, :, None], (out_d, nblk, BLOCK)).reshape(out_d, in_d)


def dequant(sidx, log2s, in_d):
    return mx.take(SG, sidx) * scale_elem(log2s, in_d)


# ------------------------------------------------------------------ refinement
def screen_exponents(W, sidx, log2s, Hd, offsets=(-1, 0, 1)):
    """Per-(row, block) exponent re-selection under the diagonal Hessian metric."""
    out_d, in_d = W.shape
    nblk = in_d // BLOCK
    Wg = W.reshape(out_d, nblk, BLOCK)
    Hdb = Hd.reshape(nblk, BLOCK)
    best_err = None
    best_log2s = None
    best_sidx = None
    for off in offsets:
        e = log2s + off
        sc = mx.power(2.0, e)
        abs_s = mx.abs(Wg) / sc[:, :, None]
        code = mx.where(abs_s > 5.0, 7,
               mx.where(abs_s > 3.5, 6,
               mx.where(abs_s > 2.5, 5,
               mx.where(abs_s > 1.75, 4,
               mx.where(abs_s > 1.25, 3,
               mx.where(abs_s > 0.75, 2,
               mx.where(abs_s > 0.25, 1, 0.0))))))).astype(mx.int32)
        neg = Wg < 0
        s = mx.where(code == 0, 7, mx.where(neg, 7 - code, 7 + code))
        deq = mx.take(SG, s) * sc[:, :, None]
        err = mx.sum(Hdb[None] * (Wg - deq) ** 2, axis=-1)
        if best_err is None:
            best_err, best_log2s, best_sidx = err, e, s
        else:
            better = err < best_err
            best_err = mx.where(better, err, best_err)
            best_log2s = mx.where(better, e, best_log2s)
            best_sidx = mx.where(better[:, :, None], s, best_sidx)
    return best_sidx.reshape(out_d, in_d), best_log2s


def flip_refine(sidx, log2s, W, Hf, n_iters):
    """Greedy coordinate descent on E2M1 codes; block exponents held fixed.

    Stops early once no row has an improving move. Returns (sidx, sweeps_used).
    """
    out_d, in_d = sidx.shape
    Hd = mx.diagonal(Hf)
    se = scale_elem(log2s, in_d)
    What = mx.take(SG, sidx) * se
    XTyfp = Hf @ W.T
    XTyq = Hf @ What.T
    rows = mx.arange(out_d)

    @mx.compile
    def sweep(sidx, XTyq):
        g = (XTyfp - XTyq).T
        up_ok = sidx < 14
        c_up = (mx.take(SG, mx.where(up_ok, sidx + 1, sidx)) - mx.take(SG, sidx)) * se
        dl_up = mx.where(up_ok, c_up * c_up * Hd - 2 * c_up * g, INF)
        dn_ok = sidx > 0
        c_dn = (mx.take(SG, mx.where(dn_ok, sidx - 1, sidx)) - mx.take(SG, sidx)) * se
        dl_dn = mx.where(dn_ok, c_dn * c_dn * Hd - 2 * c_dn * g, INF)
        up_better = dl_up <= dl_dn
        dl = mx.where(up_better, dl_up, dl_dn)
        best_dL = mx.min(dl, axis=1)
        best_j = mx.argmin(dl, axis=1).astype(mx.int32)
        improving = best_dL < -1e-12
        ub = mx.take_along_axis(up_better, best_j[:, None], axis=1)[:, 0]
        c_sel = mx.where(ub,
                         mx.take_along_axis(c_up, best_j[:, None], axis=1)[:, 0],
                         mx.take_along_axis(c_dn, best_j[:, None], axis=1)[:, 0])
        c_sel = mx.where(improving, c_sel, 0.0)
        delta = mx.where(ub, 1, -1).astype(mx.int32) * improving.astype(mx.int32)
        sidx = sidx.at[rows, best_j].add(delta)
        XTyq = XTyq + mx.take(Hf, best_j, axis=1) * c_sel[None, :]
        return sidx, XTyq, mx.sum(improving.astype(mx.int32))

    sweeps = 0
    for _ in range(n_iters):
        sidx, XTyq, n_imp = sweep(sidx, XTyq)
        sweeps += 1
        if int(n_imp) == 0:
            break
    return sidx, sweeps


def hessian_objective(W, sidx, log2s, H):
    R = W - dequant(sidx, log2s, W.shape[1])
    return float(mx.sum(R * (R @ H)))


# ------------------------------------------------------------------ hessians
class Catcher(nn.Module):
    """Wraps a linear layer and accumulates H = sum x^T x in float32."""

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.H = None

    def __call__(self, x, *args, **kwargs):
        xf = x.flatten(0, -2).astype(mx.float32)
        h = xf.T @ xf
        self.H = h if self.H is None else self.H + h
        return self.module(x, *args, **kwargs)


def _step(cur, p):
    if isinstance(cur, (list, tuple)):
        return cur[int(p)]
    return getattr(cur, p)


def get_module(root, path):
    cur = root
    for p in path.split("."):
        cur = _step(cur, p)
    return cur


def set_module(root, path, new):
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = _step(parent, p)
    last = parts[-1]
    if isinstance(parent, (list, tuple)):
        parent[int(last)] = new
    else:
        setattr(parent, last, new)


def layer_index(path):
    m = re.search(r"layers\.(\d+)\.", path)
    return int(m.group(1)) if m else -1


def calibration_batches(tokenizer, seq_len, batch_size, max_tokens):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    ids = []
    for t in ds["text"]:
        if t and t.strip():
            ids.extend(tokenizer.encode(t))
        if len(ids) >= max_tokens:
            break
    ids = ids[: (len(ids) // seq_len) * seq_len]
    data = mx.array(ids, mx.uint32).reshape(-1, seq_len)
    print(f"  calibration: {data.shape[0]} sequences x {seq_len} tokens")
    return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="MLX mxfp4 model to refine")
    ap.add_argument("--bf16", required=True, help="matching BF16 MLX model")
    ap.add_argument("--output", required=True)
    ap.add_argument("--calib-tokens", type=int, default=200_000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--flips", type=int, default=256)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--group-size", type=int, default=24)
    ap.add_argument("--hessian-cache", default="", help="dir to cache per-layer Hessians")
    ap.add_argument("--screen-offsets", default="-1,0,1",
                    help="comma-separated exponent offsets for the screen")
    ap.add_argument("--limit-layers", type=int, default=0)
    args = ap.parse_args()
    screen_offsets = tuple(int(x) for x in args.screen_offsets.split(","))

    mx.set_default_device(mx.gpu)
    t0 = time.time()
    model_q, tokenizer = load(args.model, trust_remote_code=True)
    model_bf, _ = load(args.bf16, trust_remote_code=True)
    print(f"loaded models in {time.time()-t0:.1f}s")

    targets = [
        (k, m) for k, m in tree_flatten(model_q.leaf_modules(), is_leaf=nn.Module.is_module)
        if isinstance(m, nn.QuantizedLinear)
    ]
    print(f"{len(targets)} quantized linear layers")
    if not targets:
        raise RuntimeError(
            "No nn.QuantizedLinear layers found in --model. MoE models are not "
            "supported: MLX-LM quantizes experts into QuantizedSwitchLinear "
            "(one stacked tensor), which this refiner does not handle. Use a "
            "dense or hybrid model."
        )
    if args.limit_layers:
        targets = targets[: args.limit_layers]

    # group by transformer layer
    groups = {}
    for k, m in targets:
        groups.setdefault(layer_index(k), []).append(k)
    order = sorted(groups)
    print(f"{len(order)} transformer layers, group size {args.group_size}")

    batches = calibration_batches(tokenizer, args.seq_len, args.batch_size, args.calib_tokens)

    # ---- Hessian cache ----
    cache_root = None
    if args.hessian_cache:
        cache_root = Path(args.hessian_cache) / f"tok{args.calib_tokens}_seq{args.seq_len}"
        meta = {"bf16": str(Path(args.bf16).resolve()),
                "calib_tokens": args.calib_tokens, "seq_len": args.seq_len}
        meta_path = cache_root / "meta.json"
        use_cache = meta_path.exists() and json.loads(meta_path.read_text()) == meta
        cache_root.mkdir(parents=True, exist_ok=True)
        if not use_cache:
            meta_path.write_text(json.dumps(meta, indent=2))
        print(f"hessian cache: {cache_root} ({'reuse' if use_cache else 'capture'})")

    def cpath(p):
        return cache_root / (p.replace(".", "__") + ".npy")

    total_before = total_after = 0.0
    n_done = 0
    sweep_stats = []
    t_run = time.time()
    for gi in range(0, len(order), args.group_size):
        idxs = order[gi : gi + args.group_size]
        paths = [p for i in idxs for p in groups[i]]
        print(f"\n[groups {idxs}] {len(paths)} layers")

        if cache_root is not None and all(cpath(p).exists() for p in paths):
            t1 = time.time()
            Hs = {p: mx.array(np.load(cpath(p))) for p in paths}
            print(f"  hessian load: {time.time()-t1:.1f}s")
        else:
            catchers = []
            for p in paths:
                mod = get_module(model_bf, p)
                c = Catcher(mod)
                set_module(model_bf, p, c)
                catchers.append((p, c, mod))
            t1 = time.time()
            for b in batches:
                model_bf(b)
                mx.eval([c.H for _, c, _ in catchers])
            print(f"  hessian capture: {time.time()-t1:.1f}s")
            Hs = {}
            for p, c, orig in catchers:
                Hs[p] = c.H
                if cache_root is not None:
                    np.save(cpath(p), np.array(c.H))
                set_module(model_bf, p, orig)

        for p in paths:
            H = Hs.pop(p)
            mx.eval(H)
            H = H + args.ridge * mx.mean(mx.diagonal(H)) * mx.eye(H.shape[0])
            W = get_module(model_bf, p).weight.astype(mx.float32)
            q = get_module(model_q, p)
            sidx, log2s = unpack_mxfp4(q.weight, q.scales)
            mx.eval(W, sidx, log2s)

            # one-time self-check: our unpack must reproduce mx.dequantize
            if n_done == 0:
                chk = float(mx.max(mx.abs(
                    dequant(sidx, log2s, W.shape[1])
                    - mx.dequantize(q.weight, q.scales, group_size=32, mode="mxfp4").astype(mx.float32))))
                print(f"  [self-check] unpack vs mx.dequantize max diff = {chk:.3e}")
                if chk > 1e-6:
                    raise RuntimeError(f"unpack self-check failed for {p}: {chk}")

            before = hessian_objective(W, sidx, log2s, H)
            sidx, log2s = screen_exponents(W, sidx, log2s, mx.diagonal(H), offsets=screen_offsets)
            after_screen = hessian_objective(W, sidx, log2s, H)
            sidx, sweeps = flip_refine(sidx, log2s, W, H, args.flips)
            after_flip = hessian_objective(W, sidx, log2s, H)
            sweep_stats.append(sweeps)

            packed, scales = pack_mxfp4(sidx, log2s)
            q.weight = packed
            q.scales = scales
            mx.eval(q.weight, q.scales)

            total_before += before
            total_after += after_flip
            n_done += 1
            print(f"  {p:52s} {before:.4e} -> {after_flip:.4e} "
                  f"({100*(1-after_flip/before):5.1f}%)  [screen {100*(1-after_screen/before):4.1f}%]"
                  f"  sweeps {sweeps}")
            del H

    print(f"\ntotal Hessian objective: {total_before:.6e} -> {total_after:.6e} "
          f"({100*(1-total_after/total_before):.1f}% reduction) over {n_done} layers")
    print(f"flip sweeps: mean {np.mean(sweep_stats):.1f}  max {max(sweep_stats)}  "
          f"(budget {args.flips})")
    print(f"refine wall time: {time.time()-t_run:.1f}s")

    print(f"\nsaving to {args.output}")
    save(args.output, args.model, model_q, tokenizer, load_config(Path(args.model)))
    print("done")


if __name__ == "__main__":
    main()
