#pragma once

#include <optional>

#include "gpdata/gp.h"

/**
 * @brief abstract base class for batched preconditioner for solving linear
 * systems using iterative algorithms
 */
class BatchPreconditioner {
 public:
  /**
   * @param data reference to GP data
   * @param A diagonal blocks of covar
   * @param n total matrix size including all clusters
   * @param n_blocks number of blocks (clusters)
   * @param m max number of columns on the RHS
   */
  BatchPreconditioner(const GPData& data, const mmf& A, size_t n,
                      size_t n_blocks)
      : data_(data), A_(A), n_blocks_(n_blocks), b_(n / n_blocks) {
    // Sanity check the dimensions of A during constructor, or fail early
    CHECK(A.m() == b_ * b_ && A.n() == n_blocks_);
  }

  virtual ~BatchPreconditioner() = default;

  virtual void decompose() = 0;

  /**
   * @brief apply the preconditioner to x using the decomposition result
   *  - cholesky: solves R * z = x or R^T * z = x
   *  - syevd: computes z = D * Q * x or z = (D * Q)^T * x
   *
   * @param x input matrix (or vector) to be modified in place
   * @param m number of columns in x
   * @param transpose if true, transposes the preconditioner
   */
  void precond(const mmf& x, mmf& z, bool transpose) {
    if (!logdet_.has_value()) decompose();
    precond_impl(x, z, transpose);
  }

  void precond(const mmf& x, mmf& z, bool transpose) const {
    CHECK(logdet_.has_value());
    precond_impl(x, z, transpose);
  }

  /**
   * @brief solves the system A * x = b using the decomposition result
   *
   * @param x input matrix (or vector) to be modified in place
   * @param m number of columns in x
   * @param trans if true, solve A^T * x = b instead
   */
  void solve(const mmf& x, mmf& z) {
    if (!logdet_.has_value()) decompose();
    solve_impl(x, z);
  }

  void solve(const mmf& x, mmf& z) const {
    CHECK(logdet_.has_value());
    solve_impl(x, z);
  }

  /**
   * @brief returns logdet of the preconditioner for scaling the logdet of the
   * original matrix
   *
   * @return logdet of the preconditioner
   */
  float logdet() {
    if (!logdet_.has_value()) decompose();
    return logdet_.value();
  }

  float logdet() const {
    CHECK(logdet_.has_value());
    return logdet_.value();
  }

  /**
   * @brief resets the decomposition result so it will be recomputed
   */
  void reset() { logdet_.reset(); }

 protected:
  /**
   * @param data_ reference to GP data
   * @param A_ diagonal blocks of covar
   * @param n_blocks_ number of blocks (clusters)
   * @param b_ block (cluster) size
   * @param logdet_ cached logdet value (see logdet() for more)
   */
  const GPData& data_;
  const mmf& A_;
  const size_t n_blocks_, b_;
  std::optional<float> logdet_;

  virtual void precond_impl(const mmf& x, mmf& z, bool transpose) const = 0;
  virtual void solve_impl(const mmf& x, mmf& z) const;

 private:
  BatchPreconditioner() = delete;
  BatchPreconditioner(const BatchPreconditioner&) = delete;
  BatchPreconditioner& operator=(const BatchPreconditioner&) = delete;
  BatchPreconditioner(BatchPreconditioner&&) = delete;
  BatchPreconditioner& operator=(BatchPreconditioner&&) = delete;
};

/**
 * @brief implementation of Cholesky-based preconditioner
 */
class CholeskyPreconditioner : public BatchPreconditioner {
 public:
  CholeskyPreconditioner(const GPData& data, const mmf& A, size_t n,
                         size_t n_blocks)
      : BatchPreconditioner(data, A, n, n_blocks), R_inv_(A.m(), A.n()) {}

  void decompose() override;

 protected:
  void precond_impl(const mmf& x, mmf& z, bool transpose) const override;
  void solve_impl(const mmf& x, mmf& z) const override;

  mmf R_, R_inv_;
};

/**
 * @brief implementation of gesvdr-based preconditioner
 */
class GesvdrPreconditioner : public BatchPreconditioner {
 public:
  GesvdrPreconditioner(const GPData& data, const mmf& A, size_t n,
                       size_t n_blocks, size_t rs)
      : BatchPreconditioner(data, A, n, n_blocks),
        S_(n_blocks * rs),
        U_(n / n_blocks * rs, n_blocks),
        V_(n / n_blocks * rs, n_blocks),
        rs_(rs),
        P_(n_blocks * rs) {}

  void decompose() override;

  // additional API to solve against a polynomial matrix denoted by `coeffs`
  void set_coeffs(vf coeffs);

 protected:
  mmf S_, P_, U_, V_;
  size_t rs_;

 private:
  void precond_impl(const mmf& x, mmf& z, bool transpose) const override {
    LOG(FATAL) << "Syevd preconditioner does not support precond() operation";
  }

  void solve_impl(const mmf& x, mmf& z) const override;
};
