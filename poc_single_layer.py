"""Single-linear-layer MACaQ PoC in MLX.

Implements, in mlx.core:
  1. per-channel activation-aware balancing  (p = w_std^a / act_rms^(1-a))
  2. block-Hadamard rotation (block=32)
  3. MXFP4 RTN + Hessian-weighted block exponent screen
  4. Flip-Refinement (greedy coordinate descent on E2M1 codes)

Compares Hessian-weighted reconstruction error of RTN vs full MACaQ.
"""
import math
import mlx.core as mx

mx.set_default_device(mx.gpu)

BLOCK = 32
# signed E2M1 grid, index 0..14 (index 7 == 0.0)
SG = mx.array([-6., -4., -3., -2., -1.5, -1., -0.5, 0.,
               0.5, 1., 1.5, 2., 3., 4., 6.], mx.float32)
INF = float("inf")


def hadamard_matrix(n, dtype=mx.float32):
    H = mx.ones((1, 1), dtype=dtype)
    while H.shape[0] < n:
        H = mx.concatenate([mx.concatenate([H, H], axis=1),
                            mx.concatenate([H, -H], axis=1)], axis=0)
    return H / math.sqrt(n)


def rotate_blocks(M, H32):
    """Apply block-diagonal Hadamard along the last axis."""
    *lead, n = M.shape
    return (M.reshape(lead + [n // 32, 32]) @ H32.T).reshape(lead + [n])


def balance_pre_scale(W, sum_x2, n_tokens, alpha=0.75):
    w_std = mx.std(W, axis=0) + 1e-8
    act_rms = mx.sqrt(mx.maximum(sum_x2 / max(float(n_tokens), 1.0), 1e-12))
    return mx.clip(mx.power(w_std, alpha) / mx.power(act_rms, 1.0 - alpha), 1e-4, 1e4)


def rtn_codes(M, block=BLOCK):
    """MXFP4 round-to-nearest -> (sidx, log2_amax per (row,block), scale per element)."""
    out, inp = M.shape
    nblk = inp // block
    Mg = M.reshape(out, nblk, block)
    amax = mx.maximum(mx.max(mx.abs(Mg), axis=-1, keepdims=True), 1e-12)
    log2_amax = mx.floor(mx.log2(amax)) - 2.0
    scale = mx.power(2.0, log2_amax)
    abs_s = mx.abs(Mg) / scale
    code = mx.where(abs_s > 5.0, 7,
           mx.where(abs_s > 3.5, 6,
           mx.where(abs_s > 2.5, 5,
           mx.where(abs_s > 1.75, 4,
           mx.where(abs_s > 1.25, 3,
           mx.where(abs_s > 0.75, 2,
           mx.where(abs_s > 0.25, 1, 0.0))))))).astype(mx.int32)
    neg = Mg < 0
    sidx = mx.where(code == 0, 7, mx.where(neg, 7 - code, 7 + code)).astype(mx.int32)
    scale_elem = mx.broadcast_to(scale, (out, nblk, block)).reshape(out, inp)
    return sidx.reshape(out, inp), log2_amax.squeeze(-1), scale_elem


def screen_exponents(M, Hd, block=BLOCK, offsets=(1, 0, -1)):
    """Hessian-diag-weighted per-(row,block) exponent selection."""
    out, inp = M.shape
    nblk = inp // block
    Mg = M.reshape(out, nblk, block)
    amax = mx.maximum(mx.max(mx.abs(Mg), axis=-1, keepdims=True), 1e-12)
    base = mx.floor(mx.log2(amax)) - 2.0
    Hdb = Hd.reshape(nblk, block)
    best_e = None
    best_err = None
    for off in offsets:
        e = base + off
        scale = mx.power(2.0, e)
        abs_s = mx.abs(Mg) / scale
        code = mx.where(abs_s > 5.0, 7,
               mx.where(abs_s > 3.5, 6,
               mx.where(abs_s > 2.5, 5,
               mx.where(abs_s > 1.75, 4,
               mx.where(abs_s > 1.25, 3,
               mx.where(abs_s > 0.75, 2,
               mx.where(abs_s > 0.25, 1, 0.0))))))).astype(mx.int32)
        neg = Mg < 0
        sidx = mx.where(code == 0, 7, mx.where(neg, 7 - code, 7 + code))
        deq = mx.take(SG, sidx) * scale
        err = mx.sum(Hdb * (Mg - deq) ** 2, axis=-1)   # (out, nblk)
        if best_err is None:
            best_err, best_e = err, mx.zeros_like(err)
            best_sidx, best_scale = sidx, scale
        else:
            better = err < best_err
            best_err = mx.where(better, err, best_err)
            best_e = mx.where(better, e.squeeze(-1), best_e)
            best_sidx = mx.where(better[..., None], sidx, best_sidx)
            best_scale = mx.where(better[..., None], scale, best_scale)
    se = mx.broadcast_to(best_scale, (out, nblk, block)).reshape(out, inp)
    return best_sidx.reshape(out, inp), best_e, se


def dequant(sidx, scale_elem):
    return mx.take(SG, sidx) * scale_elem


def flip_refine(sidx, scale_elem, M_target, Hf, n_iters, ridge_used=True):
    """Dense greedy coordinate descent on E2M1 codes (exponents frozen)."""
    out, inp = sidx.shape
    Hd = mx.diagonal(Hf)
    sg = SG
    What = dequant(sidx, scale_elem)
    XTyfp = Hf @ M_target.T          # (in, out)
    XTyq = Hf @ What.T               # (in, out)
    rows = mx.arange(out)

    @mx.compile
    def sweep(sidx, scale_elem, XTyq):
        g = (XTyfp - XTyq).T                       # (out, in)
        up_ok = sidx < 14
        s_up = mx.where(up_ok, sidx + 1, sidx)
        c_up = (mx.take(sg, s_up) - mx.take(sg, sidx)) * scale_elem
        dl_up = mx.where(up_ok, c_up * c_up * Hd - 2 * c_up * g, INF)
        dn_ok = sidx > 0
        s_dn = mx.where(dn_ok, sidx - 1, sidx)
        c_dn = (mx.take(sg, s_dn) - mx.take(sg, sidx)) * scale_elem
        dl_dn = mx.where(dn_ok, c_dn * c_dn * Hd - 2 * c_dn * g, INF)
        up_better = dl_up <= dl_dn
        dl = mx.where(up_better, dl_up, dl_dn)
        best_dL = mx.min(dl, axis=1)
        best_j = mx.argmin(dl, axis=1).astype(mx.int32)
        improving = best_dL < -1e-12
        ub = mx.take_along_axis(up_better, best_j[:, None], axis=1)[:, 0]
        c_sel = mx.where(ub, mx.take_along_axis(c_up, best_j[:, None], axis=1)[:, 0],
                         mx.take_along_axis(c_dn, best_j[:, None], axis=1)[:, 0])
        c_sel = mx.where(improving, c_sel, 0.0)
        delta = mx.where(ub, 1, -1).astype(mx.int32) * improving.astype(mx.int32)
        sidx = sidx.at[rows, best_j].add(delta)
        XTyq = XTyq + mx.take(Hf, best_j, axis=1) * c_sel[None, :]
        return sidx, XTyq

    for _ in range(n_iters):
        sidx, XTyq = sweep(sidx, scale_elem, XTyq)
    return sidx


def hessian_err(M, What, H):
    R = M - What
    return float(mx.sum(R * (R @ H)))


def main():
    out, inp, N = 4096, 4096, 4096
    alpha, ridge, n_flips = 0.75, 1.0, 64
    mx.random.seed(0)
    # synthetic layer + calibration activations with an outlier channel
    W = mx.random.normal((out, inp)) * 0.02
    X = mx.random.normal((N, inp))
    X = X.at[:, 100].multiply(20.0)          # outlier channel
    H = X.T @ X
    sum_x2 = mx.sum(X * X, axis=0)

    # ---------- baseline: RTN ----------
    sidx0, _, se0 = rtn_codes(W)
    err_rtn = hessian_err(W, dequant(sidx0, se0), H)

    # ---------- MACaQ ----------
    p = balance_pre_scale(W, sum_x2, N, alpha)
    Ws = W / p[None, :]
    Hs = H * p[:, None] * p[None, :]
    H32 = hadamard_matrix(32)
    Wr = rotate_blocks(Ws, H32)
    Hr = rotate_blocks(rotate_blocks(Hs, H32).T, H32).T
    Hf = Hr + ridge * mx.mean(mx.diagonal(Hr)) * mx.eye(inp)

    sidx1, _, se1 = screen_exponents(Wr, mx.diagonal(Hf))
    err_screen = hessian_err(Wr, dequant(sidx1, se1), Hr)

    sidx2 = flip_refine(sidx1, se1, Wr, Hf, n_flips)
    err_flip = hessian_err(Wr, dequant(sidx2, se1), Hr)

    print(f"layer {out}x{inp}, {N} calibration tokens, alpha={alpha}, ridge={ridge}")
    print(f"RTN                       Hessian err = {err_rtn:.6e}")
    print(f"MACaQ screen (no flip)    Hessian err = {err_screen:.6e}  "
          f"({100*(1-err_screen/ (hessian_err(Wr, dequant(rtn_codes(Wr)[0], rtn_codes(Wr)[2]), Hr))):.1f}% vs RTN-in-rotated-frame)")
    print(f"MACaQ screen+flip         Hessian err = {err_flip:.6e}  "
          f"({100*(1-err_flip/err_screen):.1f}% vs screen)")
    # RTN in rotated frame for a fair denominator
    s_rtn, _, se_rtn = rtn_codes(Wr)
    err_rtn_rot = hessian_err(Wr, dequant(s_rtn, se_rtn), Hr)
    print(f"RTN in rotated frame      Hessian err = {err_rtn_rot:.6e}")
    print(f"MACaQ total improvement over RTN(rotated): "
          f"{100*(1-err_flip/err_rtn_rot):.1f}%")

    # ---------- sanity: MLX native mxfp4 agrees with our E2M1 grid ----------
    wq, sc = mx.quantize(W[:8, :64].astype(mx.float32), group_size=32, mode='mxfp4')
    wd = mx.dequantize(wq, sc, group_size=32, mode='mxfp4')
    si, _, sei = rtn_codes(W[:8, :64])
    ours = dequant(si, sei)
    print(f"\nMLX native mxfp4 vs our RTN codes: max abs diff = "
          f"{float(mx.max(mx.abs(wd - ours))):.3e}")


if __name__ == "__main__":
    main()
