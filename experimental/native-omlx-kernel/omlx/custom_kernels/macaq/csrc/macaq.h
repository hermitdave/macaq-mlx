// Copyright © 2026 oMLX contributors
// SPDX-License-Identifier: Apache-2.0
// Based on the MACaQ algorithm by Huawei CSL (Apache-2.0).

#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace omlx::macaq {

using mlx::core::array;
using mlx::core::StreamOrDevice;

// x: [M, K] float16/bfloat16, t: [K] pre-scale, w: [N, K/8] uint32
// (OCP mxfp4 nibble packing), scales: [N, K/32] uint8 E8M0.
// Returns [M, N]. Fuses x*t, 32-block Hadamard and MXFP4 W4A16 GEMV.
array qmv(
    const array& x,
    const array& t,
    const array& w,
    const array& scales,
    int out_dims,
    StreamOrDevice s = {});

} // namespace omlx::macaq
