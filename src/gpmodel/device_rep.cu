#include "device_common.cuh"
#include "gpmodel/device_rep.h"

void DRCovar::left_mm(const mmf& p, mmf& Kp, bool transpose) const {
  auto Km = transpose ? K_.n() : K_.m(), Kn = transpose ? K_.m() : K_.n();
  CHECK(Km == Kp.m() && Kn == p.m() && p.n() == Kp.n());
  CHECK(p.data() != Kp.data());

  auto op = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
  const float one = 1.0, zero = 0.0;
  safe_cublas(cublasSgemm, -1, CUBLAS_OP_N, op, p.n(), Km, Kn, &one, p.data(),
              p.n(), K_.data(), K_.n(), &zero, Kp.data(), Kp.n());
}

void DRCovarODLR::left_mm(const mmf& p, mmf& Kp) const {
  CHECK(p.m() == b_ * n_blocks_ && p.m() == Kp.m() && p.n() == Kp.n());
  CHECK(p.data() != Kp.data());

  const float one = 1.0, zero = 0.0;
  mmf psum(n_blocks_, p.n()), scaled_psum(n_blocks_, p.n()),
      Kp_odlr(Kp.m(), Kp.n());

  // clang-format off
  // precompute sum of each column in each block of p
  // p[i] read as n * block_sz (read in column major), LD(p) = n
  // psum[i] read as n * 1 (read in column major), LD(psum) = n
  safe_cublas(cublasSgemvStridedBatched, -1, CUBLAS_OP_N,
              p.n(), b_,
              &one, p.data(), p.n(), b_ * p.n(),
              data_.ones.data(), 1, 0,
              &zero, psum.data(), 1, p.n(),
              n_blocks_);

  // Kp (diagonal part) = K_diag * p (compute batched), m * n
  safe_cublas(cublasSgemmStridedBatched, -1, CUBLAS_OP_N, CUBLAS_OP_N,
              p.n(), b_, b_,
              &one, p.data(), p.n(), b_ * p.n(),
              K_diag_.data(), b_, b_ * b_,
              &zero, Kp.data(), p.n(), b_ * p.n(),
              n_blocks_);

  // scaled_psum (not extended yet) = K_odlr * psum, n_blocks * n
  safe_cublas(cublasSgemm, -1, CUBLAS_OP_N, CUBLAS_OP_N,
              p.n(), n_blocks_, n_blocks_,
              &one, psum.data(), p.n(),
              K_odlr_.data(), n_blocks_,
              &zero, scaled_psum.data(), p.n());

  // extend scaled_psum into Kp_odlr (representation part)
  safe_cublas(cublasSgemmStridedBatched, -1, CUBLAS_OP_N, CUBLAS_OP_N,
              p.n(), b_, 1,
              &one, scaled_psum.data(), p.n(), p.n(),
              data_.ones.data(), 1, 0,
              &zero, Kp_odlr.data(), p.n(), b_ * p.n(),
              n_blocks_);
  // clang-format on

  // Kp += Kp_odlr
  safe_cublas(cublasSaxpy, -1, Kp_odlr.size(), &one, Kp_odlr.data(), 1,
              Kp.data(), 1);
}

void DRPrecondSmush::left_mm(const mmf& p, mmf& Kp) const {
  // Delegating most sanity checks to the precond and DR, except for p != Kp
  CHECK(p.data() != Kp.data());

  mmf tmp(p.m(), p.n()), tmp2(p.m(), p.n());

  // tmp = R^-1 * p or Dq * p
  bp_.precond(p, tmp, false);
  // tmp = K * Kp
  dr_.left_mm(tmp, tmp2);
  // Kp = R^-T * tmp or Dq^T * tmp
  bp_.precond(tmp2, Kp, true);
}

void DRHornerSmush::left_mm(const mmf& p, mmf& Kp) const {
  // Delegating most sanity checks to the DR, except for p != Kp
  CHECK(p.data() != Kp.data());

  mmf tmp(p.m(), p.n());

  // Kp = 0
  dmemset(Kp.data(), 0.0f, Kp.size());
  // Kp += coeffs[0] * p
  auto c = coeffs_[0];
  safe_cublas(cublasSaxpy, -1, p.size(), &c, p.data(), 1, Kp.data(), 1);

  for (int i = 1; i < coeffs_.size(); ++i) {
    // tmp = dr * Kp
    dr_.left_mm(Kp, tmp);
    // tmp += coeffs[i] * p
    if (c = coeffs_[i]; c != 0.0)
      safe_cublas(cublasSaxpy, -1, p.size(), &c, p.data(), 1, tmp.data(), 1);
    // Kp = tmp
    safe_cublas(cublasScopy, -1, tmp.size(), tmp.data(), 1, Kp.data(), 1);
  }
}
