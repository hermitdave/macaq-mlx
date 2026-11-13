import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.switch_layers import QuantizedSwitchLinear
from mlx_lm.utils import load

mx.set_default_device(mx.gpu)
model, tok = load("OLMoE-1B-7B-MLX-mxfp4")

counts = {}
switch = []
for k, m in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
    counts[type(m).__name__] = counts.get(type(m).__name__, 0) + 1
    if isinstance(m, QuantizedSwitchLinear):
        switch.append((k, m))
print("module type counts:")
for t, c in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {t:28s} {c}")

print(f"\nQuantizedSwitchLinear modules: {len(switch)}")
for k, m in switch[:6]:
    print(f"  {k:50s} weight {m.weight.shape} scales {m.scales.shape}")

print(f"\nQuantizedLinear modules: {counts.get('QuantizedLinear', 0)}")

out = model(mx.array([[1, 2, 3, 4, 5]], mx.uint32))
print("\nforward OK, logits shape:", out.shape)
