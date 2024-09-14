#include "gpmodel/model.h"

#include <gtest/gtest.h>
#include <math.h>

#include "device_test_helper.h"
#include "test_helper.h"

namespace {

void gen_well_clustered_data(vf& train_x, vf& train_y, vf& test_x, vf& rep_x,
                             vf& rand_z, size_t n_train, size_t n_test,
                             size_t n_rand, size_t n_blocks, size_t d) {
  auto b = n_train / n_blocks;
  float val_min = -10.0, val_max = 10.0, dist_min = 0.8, dist_max = 3.2;

  // Find a few rep points within a reasonable distance from each other
  rep_x.resize(n_blocks * d);
  for (int i = 0; i < n_blocks; ++i) {
    while (true) {
      vf pt = generate_random_vector(1, d, val_min, val_max);
      auto cnt = 0;
      for (int j = 0; j < i; ++j) {
        auto dist = 0.0;
        for (int k = 0; k < d; ++k)
          dist += (pt[k] - rep_x[j * d + k]) * (pt[k] - rep_x[j * d + k]);
        cnt += (dist > dist_min * d && dist < dist_max * d) ? 1 : 0;
      }
      if (cnt == i) {
        std::copy(pt.begin(), pt.end(), rep_x.begin() + i * d);
        break;
      }
    }
  }

  // Generate training data centered around each rep point for each block
  train_x.resize(n_train * d);
  for (int i = 0; i < n_blocks; ++i) {
    for (int k = 0; k < d; ++k) {
      float coord_min = rep_x[i * d + k] - dist_max;
      float coord_max = rep_x[i * d + k] + dist_max;
      vf coords = generate_random_vector(b, 1, coord_min, coord_max);
      for (int j = 0; j < b; ++j) train_x[(i * b + j) * d + k] = coords[j];
    }
  }

  // The rest of data are not constrained by the choice of rep points
  train_y = generate_random_vector(n_train, 1, val_min, val_max);
  test_x = generate_random_vector(n_test, d, val_min, val_max);
  rand_z = generate_random_vector(n_train, n_rand, -1.0, 1.0);
  for (auto& val : rand_z) val = val > 0.0 ? 1.0 : -1.0;  // Hutchinson
}

float train_forward_cpu(const GPData& data, const vf& train_x,
                        const vf& train_y, const vf& rep_x, size_t d,
                        size_t n_blocks, float lengthscale, float noise,
                        float output_scale) {
  // Make sure CUDA context is already initialized when calling this method
  EXPECT_GT(cublas_handles().size(), 0);

  auto n_train = train_y.size(), b = n_train / n_blocks;

  // Compute the two parts of covariance matrix
  vf A_diag(n_train * b), A_odlr(n_blocks * n_blocks);
  generate_covar_cpu(A_diag, train_x, train_x, d, n_blocks, lengthscale, noise,
                     output_scale);
  generate_covar_cpu(A_odlr, rep_x, rep_x, d, 1, lengthscale, noise,
                     output_scale);

  // Recreate the dense version of covariance matrix => A_dense
  vf A_dense(n_train * n_train);
  for (int i = 0; i < n_train; ++i) {
    for (int j = 0; j < n_train; ++j) {
      auto bi = i / b, bj = j / b, ri = i % b, rj = j % b;
      if (bi == bj)
        A_dense[i * n_train + j] = A_diag[bi * b * b + ri * b + rj];
      else
        A_dense[i * n_train + j] = A_odlr[bi * n_blocks + bj];
    }
  }

  // Some data needs to be copied to GPU to utilize Cholesky solve
  mmf d_A_dense(n_train * n_train), d_train_y(n_train), d_solve(n_train);
  CholeskyPreconditioner chol(data, d_A_dense, n_train, 1);
  dmemcpy_wrapper(A_dense.data(), d_A_dense.data(), A_dense.size(), true);
  dmemcpy_wrapper(train_y.data(), d_train_y.data(), train_y.size(), true);

  chol.decompose();
  chol.solve(d_train_y, d_solve);

  vf solve(n_train);
  dmemcpy_wrapper(solve.data(), d_solve.data(), solve.size(), false);

  // Now everything is ready for log_prob; no need to destroy CUDA context
  auto log_prob =
      -0.5 * (train_y * solve + chol.logdet() + n_train * log(2 * M_PI));
  return -log_prob / n_train;
}

void posterior_forward_cpu(vf& mean, vf& upper, vf& lower, const GPData& data,
                           const vf& train_x, const vf& train_y,
                           const vf& test_x, size_t d, float lengthscale,
                           float noise, float output_scale) {
  // Make sure CUDA context is already initialized when calling this method
  EXPECT_GT(cublas_handles().size(), 0);

  auto n_train = train_y.size(), n_test = test_x.size() / d;

  vf K_y(n_train * n_train), K_s(n_train * n_test), K_ss(n_test * n_test);
  generate_covar_cpu(K_y, train_x, train_x, d, 1, lengthscale, noise,
                     output_scale);
  generate_covar_cpu(K_s, train_x, test_x, d, 1, lengthscale, 0.0,
                     output_scale);
  generate_covar_cpu(K_ss, test_x, test_x, d, 1, lengthscale, 0.0,
                     output_scale);

  // Some data needs to be copied to GPU to utilize Cholesky solve
  mmf d_K_y(n_train * n_train), d_K_s(n_train, n_test), d_train_y(n_train);
  mmf d_K_solve(n_train, n_test), d_y_solve(n_train);
  CholeskyPreconditioner chol(data, d_K_y, n_train, 1);
  dmemcpy_wrapper(K_y.data(), d_K_y.data(), K_y.size(), true);
  dmemcpy_wrapper(K_s.data(), d_K_s.data(), K_s.size(), true);
  dmemcpy_wrapper(train_y.data(), d_train_y.data(), train_y.size(), true);

  chol.decompose();
  // K_solve = K_y^{-1} * K_s
  chol.solve(d_K_s, d_K_solve);
  // y_solve = K_y^{-1} * y
  chol.solve(d_train_y, d_y_solve);

  vf K_solve(n_train * n_test), y_solve(n_train);
  dmemcpy_wrapper(K_solve.data(), d_K_solve.data(), K_solve.size(), false);
  dmemcpy_wrapper(y_solve.data(), d_y_solve.data(), y_solve.size(), false);

  // mu = K_s^T * y_solve
  mean = matmul<true, false>(K_s, y_solve, n_test, n_train, 1);
  // K_tmp = K_s^T * K_solve
  auto K_tmp = matmul<true, false>(K_s, K_solve, n_test, n_train, n_test);
  // K = K_ss - K_tmp
  auto K = K_ss - K_tmp;

  upper.resize(n_test);
  lower.resize(n_test);
  generate_confidence_region_cpu(upper, lower, mean, K, noise);
}

}  // namespace

TEST(ModelTests, miniTrainForwardTest) {
  size_t n_train = 1000, n_rand = 10, d = 3, n_blocks = 5;

  initialize_cuda_env(n_blocks);

  vf train_x, train_y, test_x, rep_x, rand_z;
  gen_well_clustered_data(train_x, train_y, test_x, rep_x, rand_z, n_train, 0,
                          n_rand, n_blocks, d);

  GPData data;
  init_mock_data(data, train_x, train_y, test_x, rep_x, rand_z, n_train, 0,
                 n_rand, n_blocks, d);

  float lengthscale = 0.5, noise = 0.1, output_scale = 1.0;

  // train forward on CPU
  auto l1 = train_forward_cpu(data, train_x, train_y, rep_x, d, n_blocks,
                              lengthscale, noise, output_scale);

  // train forward on GPU
  GPResult result(data);
  GPMain model(data, result);
  model.set_params(lengthscale, noise, output_scale);
  auto l2 = model.train_forward();

  finalize_cuda_env();

  LOG(INFO) << "Ground truth loss = " << l1 << ", approximate loss = " << l2;
  EXPECT_NEAR(l1, l2, 0.1 * fabs(l1));
}

TEST(ModelTests, miniPosteriorForwardTest) {
  size_t n_train = 1000, n_test = 400, n_rand = 10, d = 3, n_blocks = 1;

  initialize_cuda_env(n_blocks);

  vf train_x, train_y, test_x, rep_x, rand_z;
  gen_well_clustered_data(train_x, train_y, test_x, rep_x, rand_z, n_train,
                          n_test, n_rand, n_blocks, d);

  GPData data;
  init_mock_data(data, train_x, train_y, test_x, rep_x, rand_z, n_train, n_test,
                 n_rand, n_blocks, d);

  float lengthscale = 0.5, noise = 0.1, output_scale = 1.0;

  // posterior forward on CPU
  vf mean_cpu, upper_cpu, lower_cpu;
  posterior_forward_cpu(mean_cpu, upper_cpu, lower_cpu, data, train_x, train_y,
                        test_x, d, lengthscale, noise, output_scale);

  // posterior forward on GPU
  GPResult result(data);
  GPMain model(data, result);
  model.set_params(lengthscale, noise, output_scale);
  model.train_forward();
  model.posterior_forward();

  vf mean_gpu(n_test), upper_gpu(n_test), lower_gpu(n_test);
  dmemcpy_wrapper(mean_gpu.data(), result.mean.data(), n_test, false);
  dmemcpy_wrapper(upper_gpu.data(), result.upper.data(), n_test, false);
  dmemcpy_wrapper(lower_gpu.data(), result.lower.data(), n_test, false);

  finalize_cuda_env();

  compare_vector(mean_cpu, mean_gpu, 1e-3);
  compare_vector(upper_cpu, upper_gpu, 1e-3);
  compare_vector(lower_cpu, lower_gpu, 1e-3);
}
