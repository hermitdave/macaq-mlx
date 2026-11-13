import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.switch_layers import QuantizedSwitchLinear
from mlx_lm.utils import load

mx.set_default_device(mx.gpu)
model, tok = load("OLMoE-1B-7B-MLX-mxfp4")

seen = []
orig = QuantizedSwitchLinear.__call__


def probe(self, x, indices, sorted_indices=False):
    if len(seen) < 8:
        seen.append((tuple(x.shape), tuple(indices.shape), bool(sorted_indices)))
        print(f"  x={x.shape}  indices={indices.shape}  sorted={sorted_indices}  E={self.num_experts}")
    return orig(self, x, indices, sorted_indices=sorted_indices)


QuantizedSwitchLinear.__call__ = probe

print("=== short prompt (T*k < 64) ===")
model(mx.array([[1, 2, 3, 4, 5]], mx.uint32))
print("\n=== long prompt (T*k >= 64) ===")
model(mx.array([list(range(200))], mx.uint32))

print("\ndistinct signatures:", sorted(set(seen)))

# BF16-side weight shapes: check the unquantized model's SwitchLinear
print("\n=== BF16 SwitchLinear weight layout ===")
import json
from mlx_lm.models.switch_layers import SwitchLinear
print("SwitchLinear attrs:", [a for a in dir(SwitchLinear) if not a.startswith('_')][:20])
