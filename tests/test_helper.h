#pragma once

#include <limits>
#include <vector>

using vf = std::vector<float>;

void compare_vector(const vf& v1, const vf& v2,
                    float eps = std::numeric_limits<float>::epsilon());

vf generate_random_vector(size_t n, size_t d, float val_min = -1.0,
                          float val_max = 1.0);

vf operator-(const vf& v1, const vf& v2);

// dot product
float operator*(const vf& v1, const vf& v2);

void check_matmul(const vf& v1, const vf& v2, size_t m, size_t k, size_t n);

template <bool transpose1 = false, bool transpose2 = false>
vf matmul(const vf& v1, const vf& v2, size_t m, size_t k, size_t n) {
  check_matmul(v1, v2, m, k, n);

  vf res(m * n, 0.0);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int l = 0; l < k; ++l) {
        auto v1_il = transpose1 ? v1[l * m + i] : v1[i * k + l];
        auto v2_lj = transpose2 ? v2[j * k + l] : v2[l * n + j];
        res[i * n + j] += v1_il * v2_lj;
      }
    }
  }
  return res;
}

void generate_covar_cpu(vf& covar, const vf& x1, const vf& x2, size_t d,
                        size_t n_blocks, float lengthscale, float noise,
                        float output_scale);

void generate_confidence_region_cpu(vf& upper, vf& lower, const vf& mean,
                                    const vf& K, float noise);
