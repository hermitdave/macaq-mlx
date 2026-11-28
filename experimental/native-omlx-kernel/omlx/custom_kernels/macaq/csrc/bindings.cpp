// Copyright © 2026 oMLX contributors
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>

#include "macaq.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "Native MACaQ W4A16 kernels for oMLX";

  // ABI canary checked by custom_kernels/_loader.py.
  m.def(
      "abi_probe",
      [](const mlx::core::array& a) { return static_cast<int64_t>(a.size()); },
      "a"_a);
  m.def(
      "qmv",
      [](const mlx::core::array& x, const mlx::core::array& t,
         const mlx::core::array& w, const mlx::core::array& scales,
         int64_t out_dims) {
        return omlx::macaq::qmv(x, t, w, scales, static_cast<int>(out_dims));
      },
      "x"_a, "t"_a, "w"_a, "scales"_a, "out_dims"_a);
}
