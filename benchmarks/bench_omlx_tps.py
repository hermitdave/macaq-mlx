"""Served-throughput benchmark for an oMLX model.

Sends natural WikiText-2 prompts at several context lengths and reads the
authoritative tokens-per-second from the oMLX server log (client-side streaming
counts are unreliable).
"""
import argparse
import json
import os
import re
import time
import urllib.request

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--contexts", nargs="+", type=int, default=[1024, 8192])
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--log", default=os.path.expanduser("~/.omlx/logs/server.log"))
    args = ap.parse_args()

    key = json.load(open(os.path.expanduser("~/.omlx/settings.json")))["auth"]["api_key"]
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t.strip() for t in ds["text"] if t and len(t.strip()) > 200)

    def ask(prompt, max_tokens):
        body = json.dumps({
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        req = urllib.request.Request(
            "http://localhost:8000/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)

    def served_tps_since(pos):
        with open(args.log) as f:
            f.seek(pos)
            tail = f.read()
        out = []
        for m in re.finditer(
                r"Chat completion: model=([^,]+), (\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\), prompt: (\d+)",
                tail):
            out.append((m.group(1), int(m.group(2)), float(m.group(4)), int(m.group(5))))
        return out, len(tail)

    log_size = os.path.getsize(args.log)

    # warmup
    print("warming up...")
    ask(text[:200], 16)

    for ctx in args.contexts:
        prompt = text[: ctx * 4]
        print(f"\n=== context ~{ctx} tokens (prompt chars {len(prompt):,}) ===")
        for i in range(args.repeats):
            before = os.path.getsize(args.log)
            d = ask(prompt, args.max_tokens)
            time.sleep(0.5)
            rows, _ = served_tps_since(before)
            row = [r for r in rows if r[0] == args.model]
            if row:
                _, ntok, tps, ptok = row[-1]
                print(f"  run {i+1}: prompt_tokens={ptok:6d} generated={ntok:4d}  served {tps:7.1f} tok/s")
            else:
                print(f"  run {i+1}: no log line found (rows={rows[-1] if rows else None})")


if __name__ == "__main__":
    main()
