"""Tokens-per-second benchmark: refined mxfp4 vs BF16 (local mlx-lm).

Uses natural WikiText-2 prose (repeated text causes degenerate short outputs)
and reports prompt-processing and generation throughput separately.
"""
import argparse
import re
import subprocess
import sys

from datasets import load_dataset


def natural_text(tokens_target):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    parts = [t.strip() for t in ds["text"] if t and len(t.strip()) > 200]
    return "\n\n".join(parts)


def run(model, prompt, max_tokens, warmup=True):
    if warmup:
        subprocess.run(
            ["mlx_lm.generate", "--model", model, "--prompt", "Hello, how are you?",
             "--max-tokens", "8"],
            capture_output=True, text=True)
    proc = subprocess.run(
        ["mlx_lm.generate", "--model", model, "--prompt", "-", "--max-tokens", str(max_tokens)],
        input=prompt, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    m = re.search(r"Prompt:\s*(\d+) tokens,\s*([\d.]+) tokens-per-sec", out)
    g = re.search(r"Generation:\s*(\d+) tokens,\s*([\d.]+) tokens-per-sec", out)
    mem = re.search(r"Peak memory:\s*([\d.]+) GB", out)
    return (int(m.group(1)), float(m.group(2))) if m else (None, None), \
           (int(g.group(1)), float(g.group(2))) if g else (None, None), \
           float(mem.group(1)) if mem else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--contexts", nargs="+", type=int, default=[1024, 8192])
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    text = natural_text(0)
    # rough token estimate: ~1 token per 4 chars for English
    print(f"corpus chars: {len(text):,}")

    for ctx in args.contexts:
        approx_chars = ctx * 4
        prompt = text[:approx_chars]
        print(f"\n=== context ~{ctx} tokens (prompt chars {len(prompt):,}) ===")
        print(f"{'model':40s} {'prompt tok':>10s} {'prompt tps':>11s} {'gen tok':>8s} {'gen tps':>9s} {'peak GB':>8s}")
        for model in args.models:
            (pt, ptps), (gt, gtps), mem = run(model, prompt, args.max_tokens)
            name = model.split("/")[-1]
            print(f"{name:40s} {str(pt):>10s} {ptps if ptps else 0:>11.1f} "
                  f"{str(gt):>8s} {gtps if gtps else 0:>9.1f} {mem if mem else 0:>8.2f}")


if __name__ == "__main__":
    main()
