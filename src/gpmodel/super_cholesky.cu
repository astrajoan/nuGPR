#include <chrono>

#include "device_common.cuh"
#include "gpmodel/super_cholesky.h"

float SuperCholeskyCPUSolver::logdet(float output_scale, float gamma,
                                     float delta) {
  if (delta == 0.0) return n_ * std::log(gamma);

  vf tmp1(n_ * n_), tmp2(n_ * n_);
  square_gemm(L_odlr_, WtW_, tmp1, true);
  square_gemm(tmp1, L_odlr_, tmp2, false);

  for (int i = 0; i < n_; ++i) {
    for (int j = 0; j < n_; ++j) {
      tmp2[i * n_ + j] *= output_scale * delta;
      if (i == j) tmp2[i * n_ + j] += gamma;
    }
  }

  CHECK(try_chol(tmp2, tmp1));
  float res = 0.0;
  for (int i = 0; i < n_; ++i) res += std::log(tmp1[i * n_ + i]);
  return 2 * res;
}

float SuperCholeskyCPUSolver::binary_search_chol() {
  float l = 0.0, r = 1.0, mid;
  while (l < r - kBisectionEps) {
    mid = l * 0.5 + r * 0.5;
    for (int i = 0; i < n_; ++i) K_odlr_[i * n_ + i] = mid;
    if (try_chol(K_odlr_, L_odlr_))
      r = mid;
    else
      l = mid;
  }
  return l;
}

void SuperCholeskyCPUSolver::square_gemm(const vf& A, const vf& B, vf& C,
                                         bool trans_A) {
  for (int i = 0; i < n_; ++i) {
    for (int j = 0; j < n_; ++j) {
      float res = 0.0;
      for (int k = 0; k < n_; ++k)
        res += A[trans_A ? k * n_ + i : i * n_ + k] * B[k * n_ + j];
      C[i * n_ + j] = res;
    }
  }
}

bool SuperCholeskyCPUSolver::try_chol(const vf& A, vf& L) {
  // iterate over each row in A
  for (int i = 0; i < n_; ++i) {
    // iterate over each column until the ith in A
    for (int j = 0; j <= i; ++j) {
      float sum = 0.0;
      // dot product of ith and jth row until column j
      for (int k = 0; k < j; ++k) sum += L[i * n_ + k] * L[j * n_ + k];
      float val = A[i * n_ + j] - sum;

      if (i == j && val < 0.0) return false;
      L[i * n_ + j] = i == j ? std::sqrt(val) : val / L[j * n_ + j];
    }
  }
  return true;
}

void SuperCholeskyPreconditioner::decompose() {
  auto start_ts = std::chrono::high_resolution_clock::now();

  CholeskyPreconditioner::decompose();

  mmf WtW(n_blocks_, n_blocks_);
  const float one = 1.0, zero = 0.0;
  // clang-format off
  // precompute each block V_i = R_i^{-T} * ones
  safe_cublas(cublasSgemvStridedBatched, -1, CUBLAS_OP_N,
              b_, b_,
              &one, R_inv_.data(), b_, b_ * b_,
              data_.ones.data(), 1, 0,
              &zero, V_.data(), 1, b_,
              n_blocks_);
  // clang-format on

  auto dur_us = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::high_resolution_clock::now() - start_ts);
  LOG(INFO) << "Cholesky took " << dur_us.count() / 1e3 << " ms";
}

void SuperCholeskySmush::left_mm(const mmf& p, mmf& Kp) const {
  // given preconditioner is not super cholesky, or given covar is not ODLR,
  // or we cannot write K' = gamma * K + delta * I, use regular left_mm
  auto bp_super = dynamic_cast<const SuperCholeskyPreconditioner*>(&bp_);
  auto dr_odlr = dynamic_cast<const DRCovarODLR*>(&dr_);
  if (!bp_super || !dr_odlr || (gamma_ == 0.0 && delta_ == 0.0)) {
    DRPrecondSmush smush(data_, dr_, bp_);
    smush.left_mm(p, Kp);
    return;
  }

  CHECK(p.m() == b_ * n_blocks_ && p.m() == Kp.m() && p.n() == Kp.n());
  CHECK(p.data() != Kp.data());

  mmf tmp(p.m(), p.n()), Vp(n_blocks_, p.n()), scaled_Vp(n_blocks_, p.n());

  // diagonal part of Kp
  if (delta_ != 0.0) {
    bp_super->precond(p, tmp, false);
    bp_super->precond(tmp, Kp, true);
  }
  safe_cublas(cublasSscal, -1, Kp.size(), &delta_, Kp.data(), 1);
  safe_cublas(cublasSaxpy, -1, Kp.size(), &gamma_, p.data(), 1, Kp.data(), 1);

  const float one = 1.0, zero = 0.0;
  // clang-format off
  // precompute block product of V_i in preconditioner against p_i
  safe_cublas(cublasSgemvStridedBatched, -1, CUBLAS_OP_N,
              p.n(), b_,
              &one, p.data(), p.n(), b_ * p.n(),
              bp_super->V().data(), 1, b_,
              &zero, Vp.data(), 1, p.n(),
              n_blocks_);
  
  // scaled_Vp (not extended yet) = K_odlr * Vp, size = n_blocks * n
  safe_cublas(cublasSgemm, -1, CUBLAS_OP_N, CUBLAS_OP_N,
              p.n(), n_blocks_, n_blocks_,
              &one, Vp.data(), p.n(),
              dr_odlr->K_odlr().data(), n_blocks_,
              &zero, scaled_Vp.data(), p.n());
  
  // extend scaled_psum into tmp (off-diagonal part of Kp)
  safe_cublas(cublasSgemmStridedBatched, -1, CUBLAS_OP_N, CUBLAS_OP_N,
              p.n(), b_, 1,
              &one, scaled_Vp.data(), p.n(), p.n(),
              bp_super->V().data(), 1, b_,
              &zero, tmp.data(), p.n(), b_ * p.n(),
              n_blocks_);
  // clang-format on

  // add off-diagonal part with diagonal part
  safe_cublas(cublasSaxpy, -1, Kp.size(), &one, tmp.data(), 1, Kp.data(), 1);
}

SuperCholeskyHornerSmush::SuperCholeskyHornerSmush(const GPData& data,
                                                   const SuperCholeskySmush& dr,
                                                   size_t n_unit, size_t n_rand,
                                                   vf coeffs)
    : DeviceRep(data), dr_(dr), coeffs_(coeffs.size(), n_unit + n_rand) {
  // prepare coeffs data
  vf expand_coeffs(coeffs_.size());
  for (int i = 0; i < coeffs.size(); ++i)
    for (int j = 0; j < coeffs_.n(); ++j)
      expand_coeffs[i * coeffs_.n() + j] = j >= n_rand ? 0.0 : coeffs[i];

  for (int j = n_rand; j < coeffs_.n(); ++j)
    expand_coeffs[(coeffs.size() - 2) * coeffs_.n() + j] = 1.0;

  dmemcpy<TransferType::kHostToDevice>(coeffs_.data(), expand_coeffs.data(),
                                       expand_coeffs.size());
}

void SuperCholeskyHornerSmush::left_mm(const mmf& p, mmf& Kp) const {
  CHECK(p.m() == Kp.m() && p.n() == Kp.n() && p.n() == coeffs_.n());
  CHECK(p.data() != Kp.data());

  mmf tmp(p.m(), p.n());

  // Kp = coeffs[0] * p
  safe_cublas(cublasSdgmm, -1, CUBLAS_SIDE_LEFT, p.n(), p.m(), p.data(), p.n(),
              coeffs_.data(), 1, Kp.data(), Kp.n());

  const float one = 1.0;
  for (int i = 1; i < coeffs_.m(); ++i) {
    // tmp = dr * Kp
    dr_.left_mm(Kp, tmp);
    // Kp = tmp
    safe_cublas(cublasScopy, -1, Kp.size(), tmp.data(), 1, Kp.data(), 1);
    // tmp = coeffs[i] * p
    safe_cublas(cublasSdgmm, -1, CUBLAS_SIDE_LEFT, p.n(), p.m(), p.data(),
                p.n(), coeffs_.data() + i * p.n(), 1, tmp.data(), tmp.n());
    // Kp += tmp
    safe_cublas(cublasSaxpy, -1, Kp.size(), &one, tmp.data(), 1, Kp.data(), 1);
  }
}
