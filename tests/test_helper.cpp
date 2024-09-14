#include "test_helper.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cmath>
#include <random>

namespace {

float rbf(const vf& x1, const vf& x2, size_t xi1, size_t xi2, size_t d,
          float lengthscale) {
  float norm = 0.0;
  for (int i = 0; i < d; ++i)
    norm += (x1[xi1 + i] - x2[xi2 + i]) * (x1[xi1 + i] - x2[xi2 + i]);
  auto res = -norm / (2 * lengthscale * lengthscale);
  return std::exp(res);
}

}  // namespace

void compare_vector(const vf& v1, const vf& v2, float eps) {
  ASSERT_EQ(v1.size(), v2.size()) << "Vectors are of unequal length";

  size_t cnt = 0;
  for (int i = 0; i < v1.size(); ++i) {
    if (std::fabs(v1[i] - v2[i]) > eps) {
      ++cnt;
      ADD_FAILURE() << "i = " << i << ": " << v1[i] << " != " << v2[i];
    }
  }
  ASSERT_EQ(cnt, 0) << "Wrong generated data, number of mismatch = " << cnt;
}

vf generate_random_vector(size_t n, size_t d, float val_min, float val_max) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(val_min, val_max);

  vf res(n * d);
  for (int i = 0; i < n * d; ++i) res[i] = dis(gen);
  return res;
}

vf operator-(const vf& v1, const vf& v2) {
  if (v1.size() != v2.size())
    LOG(FATAL) << "Inputs are of unequal length during matrix subtraction!";

  vf res(v1.size());
  for (int i = 0; i < v1.size(); ++i) res[i] = v1[i] - v2[i];
  return res;
}

float operator*(const vf& v1, const vf& v2) {
  if (v1.size() != v2.size())
    LOG(FATAL) << "Inputs are of unequal length during dot product!";

  float res = 0.0;
  for (int i = 0; i < v1.size(); ++i) res += v1[i] * v2[i];
  return res;
}

void check_matmul(const vf& v1, const vf& v2, size_t m, size_t k, size_t n) {
  if (v1.size() != m * k)
    LOG(FATAL) << "Input dimension mismatch during matrix multiplication!";
  if (v2.size() != k * n)
    LOG(FATAL) << "Input dimension mismatch during matrix multiplication!";
}

void generate_covar_cpu(vf& covar, const vf& x1, const vf& x2, size_t d,
                        size_t n_blocks, float lengthscale, float noise,
                        float output_scale) {
  auto n1 = x1.size() / n_blocks / d, n2 = x2.size() / n_blocks / d;
  for (int k = 0; k < n_blocks; ++k) {
    for (int i = 0; i < n1; ++i) {
      for (int j = 0; j < n2; ++j) {
        auto xi1 = k * n1 * d + i * d, xi2 = k * n2 * d + j * d;
        auto val = output_scale * rbf(x1, x2, xi1, xi2, d, lengthscale);
        covar[k * n1 * n2 + i * n2 + j] = i != j ? val : val + noise;
      }
    }
  }
}

void generate_confidence_region_cpu(vf& upper, vf& lower, const vf& mean,
                                    const vf& K, float noise) {
  auto n = mean.size();
  for (int i = 0; i < n; ++i) {
    upper[i] = mean[i] + 2 * std::sqrt(K[i * n + i] + noise);
    lower[i] = mean[i] - 2 * std::sqrt(K[i * n + i] + noise);
  }
}
