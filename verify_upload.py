"""Verify the uploaded safetensors matches local, via HTTP range requests."""
import json
import struct
import urllib.request

REPO = "hermitdave/LFM3-1.7B-MLX-MACaQ-R-4.25bpw"
LOCAL = "/Users/hermit.dave/Projects/ml/models/LFM3-1.7B-MLX-mxfp4-f256/model.safetensors"
URL = f"https://huggingface.co/{REPO}/resolve/main/model.safetensors"


def remote_header(url):
    req = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
    with urllib.request.urlopen(req, timeout=120) as r:
        n = struct.unpack("<Q", r.read(8))[0]
        total = r.headers.get("Content-Range", "")
    req = urllib.request.Request(url, headers={"Range": f"bytes=8-{8 + n - 1}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read(n)), total


def local_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


rh, cr = remote_header(URL)
lh = local_header(LOCAL)

rt = {k: v for k, v in rh.items() if k != "__metadata__"}
lt = {k: v for k, v in lh.items() if k != "__metadata__"}

print("remote tensors:", len(rt), " local tensors:", len(lt))
print("same tensor set:", set(rt) == set(lt))

mismatch = []
for k in sorted(lt):
    a, b = lt[k], rt.get(k)
    if b is None or tuple(a["shape"]) != tuple(b["shape"]) or a["dtype"] != b["dtype"]:
        mismatch.append((k, a.get("shape"), b and b.get("shape")))
print("shape/dtype mismatches:", len(mismatch))
for m in mismatch[:5]:
    print("  ", m)

# total payload size implied by the headers
def payload_size(d):
    return max(v["data_offsets"][1] for v in d.values())

print(f"remote header payload: {payload_size(rt):,} bytes")
print(f"local  header payload: {payload_size(lt):,} bytes")
print(f"Content-Range: {cr}")

import os
print(f"local file size:      {os.path.getsize(LOCAL):,} bytes")
