#pragma once

#include "gpmodel/precond.h"

/**
 * @brief abstract class wrapper for device representation matrix or operator
 *
 * @details
 *  - n rows * m columns, row major unless otherwise specified
 *  - fully runs on GPU, doesn't own resources
 *  - abstract base class for matrix/operator that reside/run on GPU
 *  - includes abstract interface for left matrix multiplication
 *  - supports efficient Horner's method for polynomial evaluation
 */
class DeviceRep {
 public:
  explicit DeviceRep(const GPData& data) : data_(data) {}

  virtual ~DeviceRep() = default;

  /**
   * @brief let current representation = K, computes Kp = K * p
   *
   * @param p input matrix
   * @param Kp output matrix
   */
  virtual void left_mm(const mmf& p, mmf& Kp) const = 0;

 protected:
  const GPData& data_;

 private:
  DeviceRep() = delete;
  DeviceRep(const DeviceRep&) = delete;
  DeviceRep& operator=(const DeviceRep&) = delete;
  DeviceRep(DeviceRep&&) = delete;
  DeviceRep& operator=(DeviceRep&&) = delete;
};

/******************************************************************************
 * matrix representations
 ******************************************************************************/

/**
 * @brief original covariance matrix without any compression, m * n
 */
class DRCovar : public DeviceRep {
 public:
  DRCovar(const GPData& data, size_t m, size_t n) : DeviceRep(data), K_(m, n) {}
  DRCovar(const GPData& data, size_t n) : DRCovar(data, n, n) {}

  const mmf& K() const { return K_; }
  float* K_data() { return K_.data(); }

  void left_mm(const mmf& p, mmf& Kp) const override { left_mm(p, Kp, false); }
  /**
   * @brief Kp = op(K_) * p, op = `transpose`
   */
  void left_mm(const mmf& p, mmf& Kp, bool transpose) const;

 private:
  mmf K_;
};

/**
 * @brief covariance matrix with off-diagonal blocks approximated by low-rank
 *        representation (OD-LR), has to be square matrix n * n
 */
class DRCovarODLR : public DeviceRep {
 public:
  DRCovarODLR(const GPData& data, size_t n, size_t n_blocks)
      : DeviceRep(data),
        n_blocks_(n_blocks),
        b_(n / n_blocks),
        K_diag_((n / n_blocks) * (n / n_blocks), n_blocks),
        K_odlr_(n_blocks, n_blocks) {}

  const mmf& K_diag() const { return K_diag_; }
  float* K_diag_data() { return K_diag_.data(); }

  const mmf& K_odlr() const { return K_odlr_; }
  float* K_odlr_data() { return K_odlr_.data(); }

  void left_mm(const mmf& p, mmf& Kp) const override;

 private:
  const size_t n_blocks_, b_;

  /**
   * @param K_diag_ diagonal blocks of covar
   * @param K_odlr_ low-rank representation of all off diagonal blocks,
   *                generated from input data clusters with diagonals = 0
   */
  mmf K_diag_, K_odlr_;
};

/******************************************************************************
 * operator representations
 ******************************************************************************/

/**
 * @brief operator that represents the smush of covar and preconditioner
 *
 * @details
 *  - holds references to covar and its preconditioner, doesn't own resources
 *  - cholesky: represents R^-T * K * R^-1
 *  - syevd: represents D^(-1/2) * Q^T * K * Q * D^(-1/2) = Dq^T * K * Dq
 *  - supports efficient left matrix multiplication with covar and indirect
 *    preconditioning
 */
class DRPrecondSmush : public DeviceRep {
 public:
  DRPrecondSmush(const GPData& data, const DeviceRep& dr,
                 const BatchPreconditioner& bp)
      : DeviceRep(data), dr_(dr), bp_(bp) {}

  /**
   * @brief Kp = precond(dr_) * p, dr_ is some representation of K
   */
  void left_mm(const mmf& p, mmf& Kp) const override;

 private:
  const DeviceRep& dr_;
  const BatchPreconditioner& bp_;
};

/**
 * @brief operator that represents the smush of K and its polynomial
 *
 * @details
 *  - holds a constant reference to K and the coefficients in the polynomial,
 *    doesn't own K
 *  - represents P(K) for the representation K given by dr
 */
class DRHornerSmush : public DeviceRep {
 public:
  DRHornerSmush(const GPData& data, const DeviceRep& dr, const vf& coeffs)
      : DeviceRep(data), dr_(dr), coeffs_(coeffs) {}

  /**
   * @brief Kp = polynomial(dr_) * p, dr_ is some representation of K, may be
   *        preconditioned
   */
  void left_mm(const mmf& p, mmf& Kp) const override;

 private:
  const DeviceRep& dr_;
  const vf coeffs_;
};
