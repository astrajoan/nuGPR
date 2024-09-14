#include <gtest/gtest.h>

#include "device_test_helper.h"
#include "test_helper.h"

TEST(DeviceFunctionTests, generateSymmetricCovarTest) {
  size_t n = 1000, d = 1, b = 1;
  auto x = generate_random_vector(n, d, -10.0, 10.0);

  float lengthscale = 0.5, noise = 0.1, output_scale = 1.0;
  // calculate covar on CPU
  vf covar1(n * n);
  generate_covar_cpu(covar1, x, x, d, b, lengthscale, noise, output_scale);

  // calculate covar on GPU
  vf covar2(n * n);
  generate_covar_gpu(covar2, x, x, d, b, lengthscale, noise, output_scale);

  // compare two generated covar
  compare_vector(covar1, covar2);
}

TEST(DeviceFunctionTests, generateRectangularCovarTest) {
  size_t n1 = 1000, n2 = 1100, d = 1, b = 1;
  auto x1 = generate_random_vector(n1, d, -10.0, 10.0);
  auto x2 = generate_random_vector(n2, d, -10.0, 10.0);

  // here noise has to be 0.0, because it's not symmetric!
  float lengthscale = 0.5, noise = 0.0, output_scale = 1.0;
  // calculate covar on CPU
  vf covar1(n1 * n2);
  generate_covar_cpu(covar1, x1, x2, d, b, lengthscale, noise, output_scale);

  // calculate covar on GPU
  vf covar2(n1 * n2);
  generate_covar_gpu(covar2, x1, x2, d, b, lengthscale, noise, output_scale);

  // compare two generated covar
  compare_vector(covar1, covar2);
}

TEST(DeviceFunctionTests, generateMultiDimCovarTest) {
  // generate two random vectors
  size_t n1 = 1000, n2 = 1100, d = 5, b = 1;
  auto x1 = generate_random_vector(n1, d, -10.0, 10.0);
  auto x2 = generate_random_vector(n2, d, -10.0, 10.0);

  // here noise has to be 0.0, because it's not symmetric!
  float lengthscale = 0.5, noise = 0.0, output_scale = 1.0;
  // calculate covar on CPU
  vf covar1(n1 * n2);
  generate_covar_cpu(covar1, x1, x2, d, b, lengthscale, noise, output_scale);

  // calculate covar on GPU
  vf covar2(n1 * n2);
  generate_covar_gpu(covar2, x1, x2, d, b, lengthscale, noise, output_scale);

  // compare two generated covar
  compare_vector(covar1, covar2);
}

TEST(DeviceFunctionTests, generateMultiBlockCovarTest) {
  // generate two random vectors
  size_t n = 200, d = 1, b = 10;
  auto x1 = generate_random_vector(n * b, d, -10.0, 10.0);
  auto x2 = generate_random_vector(n * b, d, -10.0, 10.0);

  float lengthscale = 0.5, noise = 0.1, output_scale = 1.0;
  // calculate covar on CPU
  vf covar1(n * n * b);
  generate_covar_cpu(covar1, x1, x2, d, b, lengthscale, noise, output_scale);

  // calculate covar on GPU
  vf covar2(n * n * b);
  generate_covar_gpu(covar2, x1, x2, d, b, lengthscale, noise, output_scale);

  // compare two generated covar
  compare_vector(covar1, covar2);
}

TEST(DeviceFunctionTests, generateConfidenceRegionTest) {
  size_t n = 1000, d = 1, b = 1;
  auto x = generate_random_vector(n, d, -10.0, 10.0);

  float lengthscale = 0.5, noise = 0.1, output_scale = 1.0;
  vf covar(n * n);
  generate_covar_cpu(covar, x, x, d, b, lengthscale, noise, output_scale);

  auto mean = generate_random_vector(n, d, 0.0, 1.0);

  // calculate confidence region on CPU
  vf upper1(n), lower1(n);
  generate_confidence_region_cpu(upper1, lower1, mean, covar, 0.0);

  // calculate confidence region on GPU
  vf upper2(n), lower2(n);
  generate_confidence_region_gpu(upper2, lower2, mean, covar, n, 0.0);

  compare_vector(upper1, upper2);
  compare_vector(lower1, lower2);
}
