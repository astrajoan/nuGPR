#include <gtest/gtest.h>

#include "device_test_helper.h"
#include "gpmodel/solver.h"
#include "test_helper.h"

namespace {

class CholeskyPreconditionerTest : public CholeskyPreconditioner {
 public:
  using CholeskyPreconditioner::CholeskyPreconditioner;

  const mmf& R() const { return R_; }
  float* R_data() { return R_.data(); }

  void solve_baseline(const mmf& x, mmf& z) const {
    // Call two-precond-pass solve explicitly for comparison purposes
    BatchPreconditioner::solve_impl(x, z);
  }
};

float get_exact_mock_spd_matrix(vf& A_diag, vf& R_cpu, size_t n, size_t b,
                                size_t n_blocks) {
  float res = 0.0;
  for (int i = 0; i < n_blocks; ++i) {
    vf L = generate_random_vector(b, b, -1.0, 1.0);
    for (int j = 0; j < b; ++j) {
      for (int k = 0; k < b; ++k) {
        L[j * b + k] = j >= k ? (L[j * b + k] > 0.0 ? 1.0 : 0.5) : 0.0;
        if (j == k) res += 2 * log(L[j * b + k]);
      }
    }
    vf A = matmul<false, true>(L, L, b, b, b);
    for (int j = 0; j < b; ++j) {
      for (int k = 0; k < b; ++k) {
        A_diag[i * b * b + j * b + k] = A[j * b + k];
        R_cpu[i * b * b + j * b + k] = L[k * b + j];  // column major
      }
    }
  }
  return res;
}

}  // namespace

TEST(PrecondSolverTests, miniCholeskyDecomposeTest) {
  size_t n = 500, n_blocks = 5;
  auto b = n / n_blocks;

  initialize_cuda_env(n_blocks);

  vf A_diag(n * b), R_cpu(n * b);
  auto logdet_cpu = get_exact_mock_spd_matrix(A_diag, R_cpu, n, b, n_blocks);

  GPData data;
  init_mock_data(data, n);

  mmf d_A_diag(b * b, n_blocks);
  CholeskyPreconditionerTest chol(data, d_A_diag, n, n_blocks);
  dmemcpy_wrapper(A_diag.data(), d_A_diag.data(), A_diag.size(), true);

  chol.decompose();

  vf R_gpu(n * b);
  dmemcpy_wrapper(R_gpu.data(), chol.R_data(), R_gpu.size(), false);
  auto logdet_gpu = chol.logdet();

  finalize_cuda_env();

  // Zero out the unused triangular part of R_gpu due to cuSOLVER behavior
  for (int i = 0; i < n * b; ++i) {
    auto bidx = i % (b * b), j = bidx / b, k = bidx % b;
    if (j > k) R_gpu[i] = 0.0;
  }

  compare_vector(R_cpu, R_gpu, 1e-3);
  ASSERT_NEAR(logdet_cpu, logdet_gpu, 1e-3);
}

TEST(PrecondSolverTests, cholCGCrossValidationTest) {
  size_t n = 500, n_blocks = 5, n_cols = 10;
  auto b = n / n_blocks;

  initialize_cuda_env(n_blocks);

  vf x1 = generate_random_vector(n, 1, -1.0, 1.0);
  vf x2 = generate_random_vector(n, 1, -1.0, 1.0);
  vf A_diag(n * b);
  generate_covar_cpu(A_diag, x1, x2, 1, n_blocks, 0.5, 0.1, 1.0);
  vf rhs = generate_random_vector(n, n_cols, -1.0, 1.0);

  GPData data;
  init_mock_data(data, n);

  DRCovarODLR dr_A(data, n, n_blocks);
  mmf d_rhs(n, n_cols);
  CholeskyPreconditionerTest chol(data, dr_A.K_diag(), n, n_blocks);
  BatchCGSolver cgsolver(data);

  dmemcpy_wrapper(A_diag.data(), dr_A.K_diag_data(), A_diag.size(), true);
  dmemcpy_wrapper(rhs.data(), d_rhs.data(), rhs.size(), true);
  vf zeros(n_blocks * n_blocks, 0.0);
  dmemcpy_wrapper(zeros.data(), dr_A.K_odlr_data(), zeros.size(), true);

  mmf d_z_truth(n, n_cols), d_z1(n, n_cols), d_z2(n, n_cols);
  chol.decompose();
  chol.solve_baseline(d_rhs, d_z_truth);
  chol.solve(d_rhs, d_z1);
  cgsolver.bcg(dr_A, d_z2, d_rhs);

  vf z_truth(n * n_cols), z1(n * n_cols), z2(n * n_cols);
  dmemcpy_wrapper(z1.data(), d_z1.data(), z1.size(), false);
  dmemcpy_wrapper(z2.data(), d_z2.data(), z2.size(), false);
  dmemcpy_wrapper(z_truth.data(), d_z_truth.data(), z_truth.size(), false);

  finalize_cuda_env();

  compare_vector(z_truth, z1, 1e-3);
  compare_vector(z_truth, z2, 1e-3);
}

TEST(PrecondSolverTests, syevdTest) {
  initialize_cuda_env(1);
  syevd_test_impl();
  finalize_cuda_env();
}
