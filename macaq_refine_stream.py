#!/usr/bin/env python3
"""MACaQ streamed Pass 2: refine an MLX mxfp4 student against cached Hessians.

Companion to ``macaq_capture.py``. Only the mxfp4 student is resident; the
BF16 teacher is never loaded. Per-layer BF16 weights are read on demand from
the HF source checkpoint (key mapping mirrors mlx_lm's ``sanitize``: experts
are stacked back into ``(E, out, in)``, and ``w1``/``w2``/``w3`` are mapped to
``gate_proj``/``down_proj``/``up_proj``), so peak memory is
student + one layer's weights + one layer's Hessians (~16 GB for LFM2.5-24B-A2B).

The refinement math is imported unchanged from ``macaq_refine.py``.

Usage:
  python macaq_refine_stream.py --model <MLX-mxfp4> --hf-source <HF-BF16-dir> \
      --cache hessian_cache/tok200000_seq512 --output <refined-dir> \
      --group-size 4 --flips 256 --ridge 1.0 --expert-chunk 8
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from mlx.utils import tree_flatten
from mlx_lm.models.switch_layers import QuantizedSwitchLinear
from mlx_lm.utils import load, load_config, save
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from macaq_refine import (  # noqa: E402
    flip_refine,
    hessian_objective,
    layer_index,
    pack_mxfp4,
    refine_switch_linear,
    screen_exponents,
    unpack_mxfp4,
)

_RENAME = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}


class HFWeightStream:
    """Read individual tensors from an HF safetensors dir without loading it."""

    def __init__(self, hf_dir: Path):
        self.dir = Path(hf_dir)
        idx = json.loads((self.dir / "model.safetensors.index.json").read_text())
        self.key2shard = idx["weight_map"]
        self.handles = {}

    def _handle(self, shard: str):
        if shard not in self.handles:
            self.handles[shard] = safe_open(
                str(self.dir / shard), framework="pt", device="cpu")
        return self.handles[shard]

    def get(self, key: str) -> mx.array:
        shard = self.key2shard[key]
        t = self._handle(shard).get_tensor(key)
        return mx.array(t.to(torch.float32).numpy())

    def close(self):
        for h in list(self.handles.values()):
            try:
                h.__exit__(None, None, None)
            except Exception:
                pass
        self.handles.clear()


def hf_keys_for(path: str, num_experts: int):
    """Map an MLX module path to its HF source keys (+ whether they stack)."""
    if ".feed_forward.switch_mlp." in path:
        base, proj = path.split(".feed_forward.switch_mlp.")
        w = _RENAME[proj]
        return [f"{base}.feed_forward.experts.{e}.{w}.weight"
                for e in range(num_experts)], True
    p = path
    for mlx_name, hf_name in _RENAME.items():
        if p.endswith("." + mlx_name):
            p = p[: -len(mlx_name)] + hf_name
            break
    return [p + ".weight"], False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="MLX mxfp4 model to refine")
    ap.add_argument("--hf-source", required=True, help="HF BF16 source dir")
    ap.add_argument("--cache", required=True, help="Hessian cache from macaq_capture.py")
    ap.add_argument("--output", required=True)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--flips", type=int, default=256)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--expert-chunk", type=int, default=8)
    ap.add_argument("--screen-offsets", default="-1,0,1")
    ap.add_argument("--limit-layers", type=int, default=0)
    ap.add_argument("--only-layer", type=int, default=None)
    args = ap.parse_args()

    screen_offsets = tuple(int(x) for x in args.screen_offsets.split(","))
    mx.set_default_device(mx.gpu)

    cfg = json.loads((Path(args.hf_source) / "config.json").read_text())
    num_experts = int(cfg["num_experts"])
    stream = HFWeightStream(Path(args.hf_source))

    t0 = time.time()
    model_q, tokenizer = load(args.model, trust_remote_code=True)
    print(f"student loaded in {time.time()-t0:.1f}s", flush=True)

    leaves = tree_flatten(model_q.leaf_modules(), is_leaf=nn.Module.is_module)
    dense_targets = [(k, m) for k, m in leaves
                     if isinstance(m, nn.QuantizedLinear)
                     and getattr(m, "mode", "") == "mxfp4"
                     and getattr(m, "group_size", 32) == 32]
    moe_targets = [(k, m) for k, m in leaves
                   if isinstance(m, QuantizedSwitchLinear)]
    print(f"{len(dense_targets)} dense targets, {len(moe_targets)} MoE switch targets",
          flush=True)
    if not dense_targets and not moe_targets:
        raise RuntimeError("no quantized layers found in --model")

    groups: dict[int, list[tuple[str, str]]] = {}
    for k, _ in dense_targets:
        groups.setdefault(layer_index(k), []).append((k, "dense"))
    for k, _ in moe_targets:
        groups.setdefault(layer_index(k), []).append((k, "moe"))
    order = sorted(groups)
    if args.only_layer is not None:
        order = [i for i in order if i == args.only_layer]
    if args.limit_layers:
        order = order[: args.limit_layers]
    print(f"{len(order)} transformer layers, group size {args.group_size}", flush=True)

    cache_root = Path(args.cache)

    def cpath(p: str) -> Path:
        return cache_root / (p.replace(".", "__") + ".npy")

    total_before = total_after = 0.0
    n_done = 0
    sweep_stats = []
    t_run = time.time()
    try:
        for gi in range(0, len(order), args.group_size):
            idxs = order[gi : gi + args.group_size]
            paths = [p for i in idxs for p in groups[i]]
            n_moe = sum(1 for _, k in paths if k == "moe")
            print(f"\n[layers {idxs}] {len(paths)} modules ({n_moe} MoE)", flush=True)

            for p, kind in paths:
                H = mx.array(np.load(cpath(p)))
                mx.eval(H)
                keys, stacked = hf_keys_for(p, num_experts)
                t_w = time.time()
                arrs = [stream.get(k) for k in keys]
                W = mx.stack(arrs, axis=0) if stacked else arrs[0]
                mx.eval(W)
                print(f"  {p}: W {W.shape} loaded in {time.time()-t_w:.1f}s", flush=True)

                # fetch the live module (path lookup is cheap; reuse walk)
                cur = model_q
                for part in p.split("."):
                    cur = cur[int(part)] if isinstance(cur, (list, tuple)) else getattr(cur, part)
                q = cur

                if kind == "dense":
                    sidx, log2s = unpack_mxfp4(q.weight, q.scales)
                    Hr = H + args.ridge * mx.mean(mx.diagonal(H)) * mx.eye(H.shape[0])
                    before = hessian_objective(W, sidx, log2s, Hr)
                    sidx, log2s = screen_exponents(W, sidx, log2s,
                                                   mx.diagonal(Hr), offsets=screen_offsets)
                    sidx, sweeps = flip_refine(sidx, log2s, W, Hr, args.flips)
                    after = hessian_objective(W, sidx, log2s, Hr)
                    packed, scales = pack_mxfp4(sidx, log2s)
                    q.weight = packed
                    q.scales = scales
                    mx.eval(q.weight, q.scales)
                else:
                    before, after, sweeps = refine_switch_linear(
                        q, W, H, args.ridge, args.flips, screen_offsets,
                        args.expert_chunk)
                sweep_stats.append(sweeps)

                total_before += before
                total_after += after
                n_done += 1
                print(f"  {p:52s} {before:.4e} -> {after:.4e} "
                      f"({100*(1-after/before):5.1f}%)  sweeps {sweeps}", flush=True)
                del H, W, arrs
                mx.clear_cache()
    finally:
        stream.close()

    print(f"\ntotal Hessian objective: {total_before:.6e} -> {total_after:.6e} "
          f"({100*(1-total_after/total_before):.1f}% reduction) over {n_done} layers",
          flush=True)
    print(f"flip sweeps: mean {np.mean(sweep_stats):.1f}  max {max(sweep_stats)}  "
          f"(budget {args.flips})", flush=True)
    print(f"refine wall time: {time.time()-t_run:.1f}s", flush=True)

    print(f"\nsaving to {args.output}", flush=True)
    save(args.output, args.model, model_q, tokenizer, load_config(Path(args.model)))
    print("done", flush=True)


if __name__ == "__main__":
    main()
