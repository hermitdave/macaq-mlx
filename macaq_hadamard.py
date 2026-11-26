"""MACaQ W4A16 with a 32-block Hadamard rotation, natively in MLX.

Unlike macaq_refine.py this does NOT refine existing mxfp4 codes: it starts
from the BF16 weight, applies the D1 channel scale and the orthonormal
32-block Walsh-Hadamard transform to both weight and activation, then screens
exponents and runs Flip-Refinement in the rotated basis.

  y = ((x * t) B) ((W / t) B)^T

Only W' = (W / t) B is quantized; t is stored per input channel in bf16 and B
is applied online by macaq_model.py. The resulting checkpoint needs
trust_remote_code=True (and cannot be loaded as plain mxfp4).

Usage:
  python macaq_hadamard.py --bf16 Qwen3.5-9B-BF16-text --output OUT \
      --calib-tokens 200000 --flips 256
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.utils import load, load_config, save

from macaq_refine import (
    Catcher,
    calibration_batches,
    flip_refine,
    get_module,
    hessian_objective,
    layer_index,
    pack_mxfp4,
    screen_exponents,
    set_module,
    unpack_mxfp4,
)
from macaq_model import BLOCK, FORMAT, FORMAT_VERSION, MACaQLinear


def rotate_block(x):
    """Orthonormal Walsh-Hadamard on each trailing BLOCK-wide group."""
    *lead, n = x.shape
    if n % BLOCK:
        raise ValueError(f"width {n} not divisible by {BLOCK}")
    return mx.hadamard_transform(x.reshape(*lead, n // BLOCK, BLOCK)).reshape(x.shape)


def rotate_gram(H, t):
    """H' = B D H D B for D = diag(t), B = block Hadamard."""
    Hd = H * t[:, None] * t[None, :]
    Hc = rotate_block(Hd)
    Hr = rotate_block(mx.swapaxes(Hc, -1, -2))
    return mx.swapaxes(Hr, -1, -2)


def channel_scale(W, Hd, n_tokens, alpha, lo=1e-4, hi=1e4):
    """t_j = sigma_j(W)^alpha / rms_j(x)^(1-alpha), clamped, returned as bf16."""
    Wf = W.astype(mx.float32)
    wstd = mx.sqrt(mx.mean((Wf - mx.mean(Wf, axis=0)) ** 2, axis=0))
    if alpha == 1.0:
        t = wstd
    else:
        rms = mx.sqrt(mx.maximum(mx.diagonal(Hd) / max(n_tokens, 1), 1e-20))
        t = mx.power(wstd, alpha) / mx.power(rms, 1 - alpha)
    return mx.clip(t, lo, hi).astype(mx.bfloat16)


def quantize_rotated(W, H, t, flips, ridge, offsets):
    """Return packed MXFP4 codes of (W/t)B and refinement stats."""
    tf = t.astype(mx.float32)
    Wp = rotate_block(W / tf[None, :])
    Hp = rotate_gram(H, tf)
    Hp = Hp + ridge * mx.mean(mx.diagonal(Hp)) * mx.eye(Hp.shape[0])

    q, sc = mx.quantize(Wp, bits=4, group_size=BLOCK, mode="mxfp4")
    sidx, log2s = unpack_mxfp4(q, sc)
    before = hessian_objective(Wp, sidx, log2s, Hp)
    sidx, log2s = screen_exponents(Wp, sidx, log2s, mx.diagonal(Hp), offsets=offsets)
    after_screen = hessian_objective(Wp, sidx, log2s, Hp)
    sidx, sweeps = flip_refine(sidx, log2s, Wp, Hp, flips)
    after = hessian_objective(Wp, sidx, log2s, Hp)
    packed, scales = pack_mxfp4(sidx, log2s)
    return packed, scales, (before, after_screen, after, sweeps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True, help="BF16 MLX model (text-only preferred)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--calib-tokens", type=int, default=200_000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--flips", type=int, default=256)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="D1 exponent; 1.0 is best for W4A16")
    ap.add_argument("--group-size", type=int, default=2,
                    help="transformer layers whose Hessians are resident at once")
    ap.add_argument("--hessian-cache", default="")
    ap.add_argument("--screen-offsets", default="-1,0,1")
    ap.add_argument("--limit-layers", type=int, default=0)
    args = ap.parse_args()
    offsets = tuple(int(x) for x in args.screen_offsets.split(","))

    mx.set_default_device(mx.gpu)
    t0 = time.time()
    model, tokenizer = load(args.bf16, trust_remote_code=True)
    print(f"loaded BF16 model in {time.time()-t0:.1f}s", flush=True)

    targets = [
        (k, m) for k, m in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
        if isinstance(m, nn.Linear)
        and m.weight.shape[1] % BLOCK == 0
        and m.weight.shape[1] >= 128
        and not any(s in k.lower() for s in ("lm_head", "embed", "norm"))
    ]
    if args.limit_layers:
        targets = [(k, m) for k, m in targets
                   if 0 <= layer_index(k) < args.limit_layers]
    groups = {}
    for k, _ in targets:
        groups.setdefault(layer_index(k), []).append(k)
    order = sorted(groups)
    print(f"{len(targets)} linear targets in {len(order)} layers", flush=True)
    if not targets:
        raise RuntimeError("no target linears found")

    batches = calibration_batches(tokenizer, args.seq_len, args.batch_size,
                                  args.calib_tokens)
    n_tokens = args.calib_tokens

    cache_root = None
    if args.hessian_cache:
        cache_root = Path(args.hessian_cache) / f"tok{args.calib_tokens}_seq{args.seq_len}"
        cache_root.mkdir(parents=True, exist_ok=True)
        (cache_root / "meta.json").write_text(json.dumps({
            "bf16": str(Path(args.bf16).resolve()),
            "calib_tokens": args.calib_tokens,
            "seq_len": args.seq_len,
        }, indent=2))

    def cpath(p):
        return cache_root / (p.replace(".", "__") + ".npy")

    specs = {}
    totals = [0.0, 0.0]
    t_run = time.time()
    done = 0
    for gi in range(0, len(order), args.group_size):
        idxs = order[gi:gi + args.group_size]
        paths = [p for i in idxs for p in groups[i]]
        print(f"\n[layers {idxs}] {len(paths)} targets", flush=True)

        import numpy as np
        if cache_root and all(cpath(p).exists() for p in paths):
            Hs = {p: mx.array(np.load(cpath(p))) for p in paths}
            print("  hessians from cache", flush=True)
        else:
            catchers = []
            for p in paths:
                mod = get_module(model, p)
                c = Catcher(mod)
                set_module(model, p, c)
                catchers.append((p, c, mod))
            t1 = time.time()
            for b in batches:
                model(b)
                mx.eval([c.H for _, c, _ in catchers])
            print(f"  hessian capture {time.time()-t1:.1f}s", flush=True)
            Hs = {}
            for p, c, orig in catchers:
                Hs[p] = c.H
                if cache_root:
                    np.save(cpath(p), np.array(c.H))
                set_module(model, p, orig)

        for p in paths:
            H = Hs.pop(p)
            mx.eval(H)
            mod = get_module(model, p)
            W = mod.weight.astype(mx.float32)
            t = channel_scale(W, H, n_tokens, args.alpha)
            packed, scales, (before, after_screen, after, sweeps) = quantize_rotated(
                W, H, t, args.flips, args.ridge, offsets)
            del W, H
            mx.clear_cache()

            new = MACaQLinear(mod.weight.shape[1], mod.weight.shape[0],
                              block=BLOCK, has_t=True, rotate=True)
            new.weight = packed
            new.scales = scales
            new.t = t
            set_module(model, p, new)
            mx.eval(new.weight, new.scales, new.t)
            specs[p] = {"block": BLOCK, "has_t": True, "rotate": True}

            totals[0] += before
            totals[1] += after
            done += 1
            print(f"  {p:56s} {before:.3e} -> {after:.3e} "
                  f"({100*(1-after/before):4.1f}%) screen {100*(1-after_screen/before):4.1f}% "
                  f"sweeps {sweeps}", flush=True)

    print(f"\nrotated Hessian objective: {totals[0]:.5e} -> {totals[1]:.5e} "
          f"({100*(1-totals[1]/totals[0]):.1f}%) over {done} layers")
    print(f"refine wall time {time.time()-t_run:.1f}s", flush=True)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(Path(args.bf16))
    cfg["model_file"] = "macaq_model.py"
    cfg["macaq"] = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "block": BLOCK,
        "alpha": args.alpha,
        "flips": args.flips,
        "modules": specs,
    }
    print(f"saving to {out}", flush=True)
    save(str(out), args.bf16, model, tokenizer, cfg)
    shutil.copy2(Path(__file__).with_name("macaq_model.py"), out / "macaq_model.py")
    src = Path(args.bf16)
    for name in ("chat_template.jinja",):
        if (src / name).exists() and not (out / name).exists():
            shutil.copy2(src / name, out / name)
    (out / "macaq_provenance.json").write_text(json.dumps({
        "source": str(Path(args.bf16).resolve()),
        "calib_tokens": args.calib_tokens,
        "seq_len": args.seq_len,
        "flips": args.flips,
        "ridge": args.ridge,
        "alpha": args.alpha,
        "layers": len(specs),
        "wall_time_s": round(time.time() - t_run, 1),
    }, indent=2) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
