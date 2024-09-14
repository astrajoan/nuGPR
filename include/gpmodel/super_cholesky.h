#pragma once

#include "device_rep.h"
#include "precond.h"

/**
 * @brief a CPU-based Cholesky solver for extremely small matrices
 */
class SuperCholeskyCPUSolver {
 public:
  explicit SuperCholeskyCPUSolver(size_t n_blocks)
      : n_(n_blocks),
        K_odlr_(n_blocks * n_blocks),
        L_odlr_(n_blocks * n_blocks, 0.0),
        WtW_(n_blocks * n_blocks) {}

  float* K_odlr_data() { return K_odlr_.data(); }
  float* WtW_data() { return WtW_.data(); }

  // compute the logdet according to det(I + W^T * W) = det(I + W * W^T)
  float logdet(float output_scale, float gamma, float delta);

  // binary search the smallest possible diagonal values to fill K_odlr
  float binary_search_chol();

 private:
  static constexpr float kBisectionEps = 1e-6;

  void square_gemm(const vf& A, const vf& B, vf& C, bool trans_A);
  bool try_chol(const vf& A, vf& L);

  size_t n_;
  vf K_odlr_, L_odlr_, WtW_;
};

/**
 * @brief binder struct to compute outputscale gradient from noise
 */
struct SuperNoiseOutputscaleBinder {
  float e0, grad1, grad2;
};

/**
 * @brief advanced version of Cholesky preconditioner for more optimizations
 */
class SuperCholeskyPreconditioner : public CholeskyPreconditioner {
 public:
  SuperCholeskyPreconditioner(const GPData& data, const mmf& A, size_t n,
                              size_t n_blocks, SuperCholeskyCPUSolver& chol_cpu)
      : CholeskyPreconditioner(data, A, n, n_blocks),
        V_(n),
        chol_cpu_(chol_cpu) {}

  void decompose() override;

  const mmf& V() const { return V_; }

 private:
  mmf V_;
  SuperCholeskyCPUSolver& chol_cpu_;
};

/**
 * @brief a smush to compute R^{-T} * K' * R^{-1}, K' = gamma * K + delta * I
 */
class SuperCholeskySmush : public DeviceRep {
 public:
  SuperCholeskySmush(const GPData& data, const DeviceRep& dr,
                     const BatchPreconditioner& bp, size_t n, size_t n_blocks,
                     float gamma, float delta)
      : DeviceRep(data),
        dr_(dr),
        bp_(bp),
        n_blocks_(n_blocks),
        b_(n / n_blocks),
        gamma_(gamma),
        delta_(delta) {}

  void left_mm(const mmf& p, mmf& Kp) const override;

 private:
  const DeviceRep& dr_;
  const BatchPreconditioner& bp_;
  const size_t n_blocks_, b_;
  const float gamma_, delta_;
};

/**
 * @brief experimental API
 */
class SuperCholeskyHornerSmush : public DeviceRep {
 public:
  SuperCholeskyHornerSmush(const GPData& data, const SuperCholeskySmush& dr,
                           size_t n_unit, size_t n_rand, vf coeffs);

  void left_mm(const mmf& p, mmf& Kp) const override;

 private:
  const SuperCholeskySmush& dr_;
  mmf coeffs_;
};
