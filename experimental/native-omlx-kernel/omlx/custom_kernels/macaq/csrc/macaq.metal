// Copyright © 2026 oMLX contributors
// SPDX-License-Identifier: Apache-2.0
//
// Native W4A16 kernel for MACaQ Hadamard checkpoints.
//
// Fuses per-channel pre-scale, normalized 32-wide Hadamard (butterfly via
// simd_shuffle_xor, no threadgroup barriers inside the K loop), OCP MXFP4
// E2M1/E8M0 weight decode and a batched GEMV. Design follows the tiled decode
// pattern in Huawei CSL MACaQ's vllm/triton_mxfp4_moe_decode.py, adapted to
// weight-only W4A16 (activations stay bf16/fp16).

#include "mlx/backend/metal/kernels/utils.h"

constant float MACAQ_SG[15] = {
    -6.0f, -4.0f, -3.0f, -2.0f, -1.5f, -1.0f, -0.5f, 0.0f,
     0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f};

template <typename T, int R>
[[kernel]] void macaq_qmv(
    const device T* x [[buffer(0)]],
    const device T* t [[buffer(1)]],
    const device uint32_t* w [[buffer(2)]],
    const device uint8_t* sc [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup float laneacc[4][32];
  threadgroup float red[4];

  const int out_row = int(tg.x);
  const int xrow0 = int(tg.y) * R;
  const int groups = K / 32;

  thread float acc[R];
  for (int r = 0; r < R; ++r) acc[r] = 0.0f;

  // Four simdgroups cover different K groups, so the K loop is 4x shorter
  // than a single-simdgroup GEMV and has no per-group barriers.
  for (int g = int(simd_gid); g < groups; g += 4) {
    const int kbase = g * 32 + int(simd_lid);
    const float pre = float(t[kbase]);

    thread float rot[R];
    for (int r = 0; r < R; ++r) {
      const int xrow = xrow0 + r;
      float a = xrow < M ? float(x[size_t(xrow) * K + kbase]) * pre : 0.0f;
      for (int s = 1; s < 32; s <<= 1) {
        const float other = simd_shuffle_xor(a, s);
        const bool up = (simd_lid & uint(s)) != 0u;
        a = up ? (other - a) : (a + other);
      }
      rot[r] = a * 0.1767766952966369f;
    }

    const uint32_t word =
        w[size_t(out_row) * (K / 8) + size_t(g) * 4u + simd_lid / 8u];
    const int nib = int((word >> (4u * (simd_lid % 8u))) & 0xFu);
    const int sidx = (nib >= 8) ? (15 - nib) : (nib + 7);
    const float scale = exp2(float(sc[size_t(out_row) * (K / 32) + g]) - 127.0f);
    const float wv = MACAQ_SG[sidx] * scale;

    for (int r = 0; r < R; ++r) acc[r] += rot[r] * wv;
  }

  for (int r = 0; r < R; ++r) {
    laneacc[simd_gid][simd_lid] = acc[r];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lid == 0u) {
      float partial = 0.0f;
      for (int j = 0; j < 32; ++j) partial += laneacc[simd_gid][j];
      red[simd_gid] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0u && simd_lid == 0u) {
      const float total = red[0] + red[1] + red[2] + red[3];
      const int xrow = xrow0 + r;
      if (xrow < M) y[size_t(xrow) * N + out_row] = static_cast<T>(total);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

#define macaq_instantiate_rows(type, sfx)                                       \
  instantiate_kernel("macaq_qmv_" #type sfx "_r1", macaq_qmv, type, 1)           \
  instantiate_kernel("macaq_qmv_" #type sfx "_r4", macaq_qmv, type, 4)           \
  instantiate_kernel("macaq_qmv_" #type sfx "_r8", macaq_qmv, type, 8)

macaq_instantiate_rows(float16_t, "_f16")
macaq_instantiate_rows(bfloat16_t, "_bf16")
