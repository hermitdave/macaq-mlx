"""Loader shim for MACaQ W4A16 checkpoints with 32-block Hadamard rotation.

Shipped next to config.json. mlx-lm / oMLX load it via
  "model_file": "macaq_model.py"  +  trust_remote_code=True

Weights are MXFP4 codes of the transformed matrix W' = ((W / t) B), where B is
the orthonormal 32-block Walsh-Hadamard transform. The layer computes
  y = mxfp4_matmul(((x * t) B), W'^T)
with bf16 activations (W4A16). Without this loader a plain mxfp4 load would
silently produce garbage, so the config marker is mandatory.
"""
import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from mlx_lm.utils import _get_classes

FORMAT = "macaq-hadamard-mxfp4-w4a16"
FORMAT_VERSION = 1
BLOCK = 32

# One threadgroup per activation row, one lane per element of a 32-block.
# Butterfly steps s = 1,2,4,8,16 compute the unnormalised Walsh-Hadamard;
# the final 1/sqrt(32) makes it orthonormal, matching mx.hadamard_transform.
_ROTATE_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    uint gid = threadgroup_position_in_grid.y;
    int groups = K_DIM / 32;
    uint row = gid / uint(groups);
    int g = int(gid % uint(groups));
    threadgroup float buf[32];
    {
        int idx = int(row) * K_DIM + g * 32 + int(lane);
        float a = float(x[idx]) * float(t[g * 32 + int(lane)]);
        for (int s = 1; s < 32; s *= 2) {
            buf[lane] = a;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            int k = (int(lane) / s) % 2;
            int base = int(lane) - k * s;
            float a0 = buf[base];
            float a1 = buf[base + s];
            a = (k == 0) ? (a0 + a1) : (a0 - a1);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        out[idx] = T(a * 0.1767766952966369f);
    }
"""

_kernel_cache = {}
_kernel_failed = False


def _rotate_kernel():
    if "kernel" not in _kernel_cache:
        _kernel_cache["kernel"] = mx.fast.metal_kernel(
            name="macaq_scale_hadamard32",
            input_names=["x", "t"],
            output_names=["out"],
            source=_ROTATE_SOURCE,
        )
    return _kernel_cache["kernel"]


def fused_scale_hadamard(x, t):
    """Fused x*t + orthonormal 32-block Hadamard, bf16 in/out."""
    shape = x.shape
    m = math.prod(shape[:-1])
    k = shape[-1]
    xf = mx.contiguous(x).reshape(m, k)
    tf = mx.contiguous(t.astype(xf.dtype))
    out = _rotate_kernel()(
        inputs=[xf, tf],
        template=[("T", xf.dtype), ("K_DIM", k)],
        grid=(32, m * (k // BLOCK), 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(m, k)],
        output_dtypes=[xf.dtype],
    )[0]
    return out.reshape(shape)


def _try_fused(x, t):
    global _kernel_failed
    if _kernel_failed:
        return None
    try:
        return fused_scale_hadamard(x, t)
    except Exception as exc:  # keep the stock path available on compile failure
        _kernel_failed = True
        print(f"[MACaQ] fused kernel unavailable, using stock path: {exc}", flush=True)
        return None


class MACaQLinear(nn.Module):
    """Drop-in Linear holding MXFP4 codes of a Hadamard-rotated weight."""

    def __init__(self, input_dims, output_dims, block=BLOCK, has_t=True,
                 rotate=True, fused=False):
        super().__init__()
        if input_dims % block:
            raise ValueError(f"input width {input_dims} not divisible by {block}")
        self.block = int(block)
        self.rotate = bool(rotate)
        self.fused = bool(fused)
        self.weight = mx.zeros((output_dims, input_dims // 8), dtype=mx.uint32)
        self.scales = mx.zeros((output_dims, input_dims // block), dtype=mx.uint8)
        if has_t:
            self.t = mx.ones((input_dims,), dtype=mx.bfloat16)
        self.freeze()

    def __call__(self, x):
        rotated = None
        if "t" in self:
            if self.fused:
                rotated = _try_fused(x, self.t)
            if rotated is None:
                x = x * self.t.astype(x.dtype)
        if rotated is not None:
            x = rotated
        elif self.rotate:
            shape = x.shape
            x = mx.hadamard_transform(
                x.reshape(*shape[:-1], shape[-1] // self.block, self.block)
            ).reshape(shape)
        return mx.quantized_matmul(
            x, self.weight, self.scales, transpose=True,
            group_size=self.block, bits=4, mode="mxfp4"
        )


class ModelArgs:
    @classmethod
    def from_dict(cls, params):
        return params


class Model:
    """Factory: stock mlx-lm model for config['model_type'] with rotated linears."""

    def __new__(cls, config):
        meta = config.get("macaq")
        if not meta or meta.get("format") != FORMAT:
            raise ValueError(f"config has no '{FORMAT}' macaq section")
        if int(meta.get("format_version", 0)) > FORMAT_VERSION:
            raise ValueError("checkpoint format newer than this loader")
        base_cls, base_args = _get_classes(config)
        model = base_cls(base_args.from_dict(config))
        mods = dict(model.named_modules())
        fused = str(meta.get("fused", "none")) == "rotate"
        new = []
        for path, spec in meta["modules"].items():
            mod = mods.get(path)
            if not isinstance(mod, nn.Linear):
                raise ValueError(f"MACaQ module {path} is not a Linear in base model")
            inp, out = mod.weight.shape[1], mod.weight.shape[0]
            new.append((path, MACaQLinear(
                inp, out,
                block=int(spec.get("block", BLOCK)),
                has_t=bool(spec.get("has_t", True)),
                rotate=bool(spec.get("rotate", True)),
                fused=fused,
            )))
        model.update_modules(tree_unflatten(new))
        return model
