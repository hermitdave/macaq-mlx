// Copyright © 2026 oMLX contributors
// SPDX-License-Identifier: Apache-2.0
// Native MACaQ W4A16 GEMV dispatch. Metal source: macaq.metal.

#include "macaq.h"

#include <sstream>
#include <string>
#include <vector>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "omlx/binary_dir.h"

namespace omlx::macaq {

namespace {

using namespace mlx::core;

constexpr const char* kMetallibName = "omlx_macaq_kernels";

MTL::ComputePipelineState* get_kernel(metal::Device& d, const std::string& name) {
  auto* lib = d.get_library(kMetallibName, binary_dir());
  return d.get_kernel(name, lib);
}

std::string type_str(Dtype dt) {
  if (dt == float16) return "float16_t_f16";
  if (dt == bfloat16) return "bfloat16_t_bf16";
  throw std::invalid_argument("macaq: x must be float16 or bfloat16");
}

class MacaqQmv : public Primitive {
 public:
  MacaqQmv(Stream s, int rows_per_group)
      : Primitive(s), r_(rows_per_group) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("MacaqQmv has no CPU path.");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));

    const auto& x = inputs[0];
    const auto& t = inputs[1];
    const auto& w = inputs[2];
    const auto& sc = inputs[3];

    const int K = static_cast<int>(w.shape(1)) * 8;
    const int N = static_cast<int>(w.shape(0));
    const int M = static_cast<int>(x.size() / K);
    const std::string name =
        "macaq_qmv_reuse_k4_" + type_str(x.dtype()) + "_r" + std::to_string(r_);

    auto& enc = metal::get_command_encoder(s);
    enc.set_compute_pipeline_state(get_kernel(d, name));
    int c = 0;
    enc.set_input_array(x, c++);
    enc.set_input_array(t, c++);
    enc.set_input_array(w, c++);
    enc.set_input_array(sc, c++);
    enc.set_output_array(out, c++);
    enc.set_bytes(K, c++);
    enc.set_bytes(N, c++);
    enc.set_bytes(M, c++);
    enc.dispatch_threadgroups(
        MTL::Size((M + r_ - 1) / r_, N / 4, 1), MTL::Size(32, 4, 1));
  }

  bool is_equivalent(const Primitive& other) const override {
    return r_ == static_cast<const MacaqQmv&>(other).r_;
  }

  DEFINE_NAME(MacaqQmv)

 private:
  int r_;
};

} // namespace

array qmv(
    const array& x,
    const array& t,
    const array& w,
    const array& scales,
    int out_dims,
    StreamOrDevice s_) {
  auto s = to_stream(s_);
  if (x.ndim() != 2 || w.ndim() != 2 || t.ndim() != 1 || scales.ndim() != 2) {
    throw std::invalid_argument("macaq: expected x [M,K], t [K], w [N,K/8], scales [N,K/32]");
  }
  const int K = static_cast<int>(w.shape(1)) * 8;
  const int N = static_cast<int>(w.shape(0));
  if (x.shape(1) != K || t.size() != static_cast<size_t>(K) ||
      scales.shape(0) != w.shape(0) || scales.shape(1) != w.shape(1) / 4 ||
      out_dims != N || K % 32 != 0 || N % 4 != 0) {
    throw std::invalid_argument("macaq: bad shapes; K%32 and N%4 required");
  }
  if (x.dtype() != float16 && x.dtype() != bfloat16) {
    throw std::invalid_argument("macaq: x must be float16 or bfloat16");
  }
  if (w.dtype() != uint32 || scales.dtype() != uint8) {
    throw std::invalid_argument("macaq: w must be uint32 and scales uint8");
  }

  const int rows = static_cast<int>(x.size() / K);
  const int r = rows <= 1 ? 1 : (rows <= 4 ? 4 : 8);
  std::vector<array> inputs = {
      contiguous(x, false, s),
      contiguous(astype(t, x.dtype(), s), false, s),
      contiguous(w, false, s),
      contiguous(scales, false, s)};
  return array(
      {rows, N}, x.dtype(), std::make_shared<MacaqQmv>(s, r), std::move(inputs));
}

} // namespace omlx::macaq
