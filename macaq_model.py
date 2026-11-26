"""Loader shim for MACaQ W4A16 checkpoints with 32-block Hadamard rotation.

Shipped next to config.json. mlx-lm / oMLX load it via
  "model_file": "macaq_model.py"  +  trust_remote_code=True

Weights are MXFP4 codes of the transformed matrix W' = ((W / t) B), where B is
the orthonormal 32-block Walsh-Hadamard transform. The layer computes
  y = mxfp4_matmul(((x * t) B), W'^T)
with bf16 activations (W4A16). Without this loader a plain mxfp4 load would
silently produce garbage, so the config marker is mandatory.
"""
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from mlx_lm.utils import _get_classes

FORMAT = "macaq-hadamard-mxfp4-w4a16"
FORMAT_VERSION = 1
BLOCK = 32


class MACaQLinear(nn.Module):
    """Drop-in Linear holding MXFP4 codes of a Hadamard-rotated weight."""

    def __init__(self, input_dims, output_dims, block=BLOCK, has_t=True, rotate=True):
        super().__init__()
        if input_dims % block:
            raise ValueError(f"input width {input_dims} not divisible by {block}")
        self.block = int(block)
        self.rotate = bool(rotate)
        self.weight = mx.zeros((output_dims, input_dims // 8), dtype=mx.uint32)
        self.scales = mx.zeros((output_dims, input_dims // block), dtype=mx.uint8)
        if has_t:
            self.t = mx.ones((input_dims,), dtype=mx.bfloat16)
        self.freeze()

    def __call__(self, x):
        if "t" in self:
            x = x * self.t.astype(x.dtype)
        if self.rotate:
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
            )))
        model.update_modules(tree_unflatten(new))
        return model
