# Experimental native MACaQ kernel for oMLX

This is the local research scaffold built during the Route B/native-kernel investigation.
It is **not installed into oMLX**, not part of the published checkpoint, and **not ready
for an upstream PR**.

Status: built and executed successfully on Apple Silicon, numerically correct, but
slower than stock `mx.hadamard_transform + mx.quantized_matmul`. See
`docs/NATIVE_KERNEL_PLAN.md` for the measured numbers and re-open criteria.

## Contents

- `setup_macaq.py` — builds only this MetalExtension in place; run with the oMLX
  bundled CPython 3.11 and its site-packages on `PYTHONPATH`, with
  `DEVELOPER_DIR` pointing at Xcode and an overridden `Python_EXECUTABLE` for
  Python development headers.
- `omlx/custom_kernels/macaq/` — Python wrapper, nanobind bindings, C++ primitive,
  and Metal source.
- `validate_native_qmv.py` — correctness and microbenchmark harness; run it with
  the same bundled interpreter that loaded the extension.

Licensing: Apache-2.0; kernel design references Huawei CSL MACaQ (also Apache-2.0).
