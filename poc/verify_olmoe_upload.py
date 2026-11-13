"""Spot-check that the remote OLMoE safetensors bytes match local."""
import hashlib
import urllib.request

REPO = "hermitdave/OLMoE-1B-7B-MLX-MACaQ-4.25bpw"
PATH = "/Users/hermit.dave/Projects/ml/models/OLMoE-1B-7B-MLX-MACaQ-4.25bpw/model.safetensors"
URL = f"https://huggingface.co/{REPO}/resolve/main/model.safetensors"
SIZE = 3676037759

# sample windows: header, several tensor bodies, tail
windows = [(0, 65536), (1 << 20, (1 << 20) + 65536), (900_000_000, 900_000_000 + 65536),
           (1_800_000_000, 1_800_000_000 + 65536), (SIZE - 65536, SIZE)]

with open(PATH, "rb") as f:
    for start, end in windows:
        f.seek(start)
        local = hashlib.sha256(f.read(end - start)).hexdigest()
        req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end-1}"})
        with urllib.request.urlopen(req) as r:
            remote = hashlib.sha256(r.read()).hexdigest()
        status = "MATCH" if local == remote else "MISMATCH"
        print(f"  [{start:>12,} .. {end:>12,}]  {status}")
