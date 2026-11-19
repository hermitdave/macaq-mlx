#!/usr/bin/env python3
"""MACaQ streamed Pass 1: per-layer Hessian capture with ONLY the BF16 teacher.

Why this exists: ``macaq_refine.py`` loads the BF16 teacher and the mxfp4
student into memory together. That is fine up to ~9B, but at 24B (48 GB teacher
+ 13 GB student) it cannot fit in 64 GB unified memory. This script captures
the Hessians with the teacher alone (peak ~= teacher + one layer group of
Hessians), writes them to disk, and exits. Pass 2 (``macaq_refine_stream.py``)
then refines against the cached Hessians with only the mxfp4 student resident.

The capture is numerically identical to ``macaq_refine.py``: same Catcher /
SwitchCatcher hooks, same WikiText-2 calibration batches, same accumulation.
Only the memory residency changes.

Usage:
  python macaq_capture.py --bf16 <MLX-BF16-dir> --targets targets.json \
      --cache hessian_cache/tok200000_seq512 --group-size 1 \
      --calib-tokens 200000 --seq-len 512 --batch-size 4

  # smoke test one layer first:
  python macaq_capture.py ... --only-layer 7
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from macaq_refine import (  # noqa: E402
    Catcher,
    SwitchCatcher,
    calibration_batches,
    get_module,
    layer_index,
    set_module,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True, help="MLX BF16 teacher dir")
    ap.add_argument("--targets", required=True, help="targets.json from dump_targets.py")
    ap.add_argument("--cache", required=True, help="output Hessian cache dir")
    ap.add_argument("--group-size", type=int, default=1,
                    help="layers captured per forward-pass group (memory knob)")
    ap.add_argument("--calib-tokens", type=int, default=200_000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--only-layer", type=int, default=None,
                    help="capture just this transformer layer (smoke test)")
    ap.add_argument("--limit-layers", type=int, default=0,
                    help="stop after N layers (0 = all)")
    args = ap.parse_args()

    mx.set_default_device(mx.gpu)

    tgt = json.loads(Path(args.targets).read_text())
    entries = [(t["path"], "dense") for t in tgt["dense"]]
    entries += [(t["path"], "moe") for t in tgt["moe"]]

    groups: dict[int, list[tuple[str, str]]] = {}
    for p, kind in entries:
        groups.setdefault(layer_index(p), []).append((p, kind))
    order = sorted(groups)
    if args.only_layer is not None:
        order = [i for i in order if i == args.only_layer]
    if args.limit_layers:
        order = order[: args.limit_layers]
    print(f"{len(order)} layers to capture, group size {args.group_size}", flush=True)

    cache_root = Path(args.cache)
    cache_root.mkdir(parents=True, exist_ok=True)

    from mlx_lm.utils import load
    t0 = time.time()
    model_bf, tokenizer = load(args.bf16, trust_remote_code=True)
    model_bf.eval()
    print(f"teacher loaded in {time.time()-t0:.1f}s", flush=True)

    batches = calibration_batches(tokenizer, args.seq_len, args.batch_size,
                                  args.calib_tokens)
    print(f"{len(batches)} calibration batches x {args.batch_size} x {args.seq_len}",
          flush=True)

    def cpath(p: str) -> Path:
        return cache_root / (p.replace(".", "__") + ".npy")

    n_done = 0
    t_run = time.time()
    for gi in range(0, len(order), args.group_size):
        idxs = order[gi : gi + args.group_size]
        paths = [p for i in idxs for p in groups[i]]
        n_moe = sum(1 for _, k in paths if k == "moe")
        print(f"\n[layers {idxs}] {len(paths)} modules ({n_moe} MoE)", flush=True)

        catchers = []
        for p, kind in paths:
            mod = get_module(model_bf, p)
            c = Catcher(mod) if kind == "dense" else SwitchCatcher(mod)
            set_module(model_bf, p, c)
            catchers.append((p, c, mod))

        t1 = time.time()
        for bi, b in enumerate(batches):
            model_bf(b)
            mx.eval([c.H for _, c, _ in catchers])
            if (bi + 1) % 25 == 0:
                print(f"  batch {bi+1}/{len(batches)}  ({time.time()-t1:.0f}s)",
                      flush=True)
        t_cap = time.time() - t1

        for p, c, orig in catchers:
            np.save(cpath(p), np.array(c.H))
            set_module(model_bf, p, orig)
        mx.clear_cache()
        n_done += len(idxs)
        print(f"  captured in {t_cap:.1f}s -> {cache_root}", flush=True)

    meta = {
        "bf16": str(Path(args.bf16).resolve()),
        "calib_tokens": args.calib_tokens,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "layers": len(order),
        "targets": len(entries),
        "wall_time_s": round(time.time() - t_run, 1),
    }
    (cache_root / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\ncapture done: {n_done} layers, {time.time()-t_run:.1f}s", flush=True)


if __name__ == "__main__":
    main()
