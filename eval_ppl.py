"""Perplexity evaluation for MLX models on WikiText-2.

Usage: python eval_ppl.py --model <path> [--seq-len 512] [--samples 256]
"""
import argparse
import math
import time

import mlx.core as mx
import mlx.nn as nn
from datasets import load_dataset
from mlx_lm.utils import load

DATASET = "Salesforce/wikitext"
CONFIG = "wikitext-2-raw-v1"


def get_tokens(tokenizer, split="test", max_tokens=None):
    ds = load_dataset(DATASET, CONFIG, split=split)
    texts = [t for t in ds["text"] if t and t.strip()]
    # tokenizers differ on special tokens; encode plain text
    ids = []
    for t in texts:
        ids.extend(tokenizer.encode(t))
        if max_tokens and len(ids) >= max_tokens:
            break
    return ids


def build_batch(tokens, seq_len, n_samples):
    n = min(n_samples, len(tokens) // seq_len)
    ids = tokens[: n * seq_len]
    return mx.array(ids, mx.uint32).reshape(n, seq_len)


def eval_ppl(model, data, batch_size=8):
    losses = []
    for s in range(0, len(data), batch_size):
        batch = data[s : s + batch_size]
        logits = model(batch[:, :-1]).astype(mx.float32)
        loss = nn.losses.cross_entropy(logits, batch[:, 1:], reduction="none")
        mx.eval(loss)
        losses.append(loss.flatten())
    all_losses = mx.concatenate(losses)
    mean = float(all_losses.mean())
    var = float(all_losses.var(ddof=1))
    se = math.sqrt(var / all_losses.size)
    return math.exp(mean), math.exp(mean) * se, all_losses.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-tokens", type=int, default=0)
    args = ap.parse_args()

    mx.set_default_device(mx.gpu)
    model, tokenizer = load(args.model)
    t0 = time.time()
    tokens = get_tokens(tokenizer, args.split, args.max_tokens or None)
    print(f"tokenized {len(tokens)} tokens in {time.time()-t0:.1f}s")
    data = build_batch(tokens, args.seq_len, args.samples)
    print(f"eval set: {data.shape[0]} sequences x {args.seq_len}")

    t0 = time.time()
    ppl, se, ntok = eval_ppl(model, data, args.batch_size)
    dt = time.time() - t0
    print(f"\nmodel: {args.model}")
    print(f"perplexity: {ppl:.4f} +/- {se:.4f}  ({ntok} tokens, {dt:.1f}s, {ntok/dt:.0f} tok/s)")
    print(f"peak memory: {mx.get_peak_memory()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
