#pragma once

#include <limits>

#include "gpmodel/device_rep.h"

/**
 * @brief simple Conjugate Gradient (CG) solver for linear systems
 */
class BatchCGSolver {
 public:
  explicit BatchCGSolver(const GPData& data) : data_(data) {}
  BatchCGSolver(const GPData& data, size_t max_iter, float tol)
      : data_(data), max_iter_(max_iter), tol_(tol) {}

  ~BatchCGSolver() = default;

  /**
   * @brief compute column wise dot product of m1 and m2
   *
   * @details m1 and m2 has shape (m, n), res has shape (n, 1) (1D size n)
   *
   * @param m1 first matrix
   * @param m2 second matrix
   * @param res result
   */
  void batch_dot(const mmf& m1, const mmf& m2, mmf& res) const;

  /**
   * @brief batched CG algorithm, A * x = b
   *
   * @details
   *  - A is some representation of K, may be original or precond of K
   *  - in the first case, we solve K * x = b
   *  - in the second case, we solve transform(K) * x = b, used in logdet
   *
   * @param A system to solve
   * @param x result
   * @param b vector(s) to solve against, can be one or more vectors
   */
  void bcg(const DeviceRep& A, mmf& x, const mmf& b) const;

  /**
   * @brief preconditioned batched CG algorithm, A * x = b
   *
   * @details a solve M * z = r (M given by `bp`) is done every iteration
   *
   * @param bp the preconditioner to use
   * @param precond_done whether the preconditioner is already applied to A
   */
  void zbcg(const DeviceRep& A, const BatchPreconditioner& bp, mmf& x,
            const mmf& b) const;

  /**
   * @brief preconditioned batched CG algorithm, A * x = b
   *
   * @param bp the preconditioner to use
   * @param precond_done whether the preconditioner is already applied to A
   */
  void pbcg(const DeviceRep& A, const BatchPreconditioner& bp, mmf& x,
            const mmf& b, bool precond_done = false) const;

 private:
  BatchCGSolver() = delete;
  BatchCGSolver(const BatchCGSolver&) = delete;
  BatchCGSolver& operator=(const BatchCGSolver&) = delete;
  BatchCGSolver(BatchCGSolver&&) = delete;
  BatchCGSolver& operator=(BatchCGSolver&&) = delete;

  const GPData& data_;
  /**
   * @param max_iter maximum number of iterations for the CG algorithm
   * @param tol stopping criteria, norm2(r) < tol * tol
   */
  const size_t max_iter_ = 2000;
  const float tol_ = std::numeric_limits<float>::epsilon();
};
