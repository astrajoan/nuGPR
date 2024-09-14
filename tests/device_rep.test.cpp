#include "gpmodel/device_rep.h"

#include <gtest/gtest.h>

#include "device_test_helper.h"
#include "test_helper.h"

namespace {

vf odlr_left_mm_cpu(const vf& K_diag, const vf& K_rep, const vf& p, size_t n,
                    size_t m, size_t b, size_t n_blocks) {
  vf K(n * n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      auto bi = i / b, bj = j / b, ii = i % b, jj = j % b;
      K[i * n + j] = bi == bj ? K_diag[bi * b * b + ii * b + jj]
                              : K_rep[bi * n_blocks + bj];
    }
  }
  return matmul(K, p, n, n, m);
}

}  // namespace

TEST(DeviceRepTests, regularCovarTest) {
  size_t n1 = 1000, n2 = 1100, m = 10;

  // We only need one stream since regular covar only calls one gemm kernel
  initialize_cuda_env(1);

  auto K = generate_random_vector(n1, n2, -1.0, 1.0);
  auto p1 = generate_random_vector(n2, m, -1.0, 1.0);
  auto p2 = generate_random_vector(n1, m, -1.0, 1.0);

  auto Kp_cpu = matmul(K, p1, n1, n2, m);
  auto Kpt_cpu = matmul<true, false>(K, p2, n2, n1, m);

  GPData data;
  init_mock_data(data, std::max(std::max(n1, n2), m));
  DRCovar covar(data, n1, n2);
  mmf d_p1(n2, m), d_p2(n1, m), d_Kp(n1, m), d_Kpt(n2, m);

  dmemcpy_wrapper(K.data(), covar.K_data(), K.size(), true);
  dmemcpy_wrapper(p1.data(), d_p1.data(), p1.size(), true);
  dmemcpy_wrapper(p2.data(), d_p2.data(), p2.size(), true);

  vf Kp_gpu(n1 * m), Kpt_gpu(n2 * m);
  covar.left_mm(d_p1, d_Kp);
  dmemcpy_wrapper(Kp_gpu.data(), d_Kp.data(), Kp_gpu.size(), false);
  covar.left_mm(d_p2, d_Kpt, true);
  dmemcpy_wrapper(Kpt_gpu.data(), d_Kpt.data(), Kpt_gpu.size(), false);

  finalize_cuda_env();

  compare_vector(Kp_cpu, Kp_gpu, 1e-3);
  compare_vector(Kpt_cpu, Kpt_gpu, 1e-3);
}

TEST(DeviceRepTests, ODLRCovarTest) {
  size_t n = 1600, m = 10, n_blocks = 10;
  auto b = n / n_blocks;

  initialize_cuda_env(n_blocks);

  vf K_diag = generate_random_vector(b * b, n_blocks, -1.0, 1.0);
  vf K_rep = generate_random_vector(n_blocks, n_blocks, -1.0, 1.0);
  vf p = generate_random_vector(n, m, -1.0, 1.0);
  for (int i = 0; i < n_blocks; ++i) K_rep[i * n_blocks + i] = 0.0;

  auto Kp_cpu = odlr_left_mm_cpu(K_diag, K_rep, p, n, m, b, n_blocks);

  GPData data;
  init_mock_data(data, n);
  DRCovarODLR odlr(data, n, n_blocks);
  mmf d_p(n, m), d_Kp(n, m);

  dmemcpy_wrapper(K_diag.data(), odlr.K_diag_data(), K_diag.size(), true);
  dmemcpy_wrapper(K_rep.data(), odlr.K_odlr_data(), K_rep.size(), true);
  dmemcpy_wrapper(p.data(), d_p.data(), p.size(), true);

  vf Kp_gpu(n * m);
  odlr.left_mm(d_p, d_Kp);
  dmemcpy_wrapper(Kp_gpu.data(), d_Kp.data(), Kp_gpu.size(), false);

  finalize_cuda_env();

  compare_vector(Kp_cpu, Kp_gpu, 1e-3);
}

TEST(DeviceRepTests, hornerSmushTest) {
  size_t n = 400, m = 10;

  // We only need one stream since Horner of regular matrix only calls gemm
  initialize_cuda_env(1);

  vf K = generate_random_vector(n, n, -1.0, 1.0);
  vf p = generate_random_vector(n, m, -1.0, 1.0);
  vf coeffs{0.25, 1.0, 0.5};

  vf K2 = matmul(K, K, n, n, n);
  vf P_K(n * n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      P_K[i * n + j] = coeffs[0] * K2[i * n + j] + coeffs[1] * K[i * n + j] +
                       coeffs[2] * (i == j);
    }
  }
  vf Kp_cpu = matmul(P_K, p, n, n, m);

  GPData data;
  init_mock_data(data, n);
  DRCovar covar(data, n);
  DRHornerSmush horner(data, covar, coeffs);
  mmf d_p(n, m), d_Kp(n, m);

  dmemcpy_wrapper(K.data(), covar.K_data(), K.size(), true);
  dmemcpy_wrapper(p.data(), d_p.data(), p.size(), true);

  vf Kp_gpu(n * m);
  horner.left_mm(d_p, d_Kp);
  dmemcpy_wrapper(Kp_gpu.data(), d_Kp.data(), Kp_gpu.size(), false);

  finalize_cuda_env();

  compare_vector(Kp_cpu, Kp_gpu, 1e-3);
}
