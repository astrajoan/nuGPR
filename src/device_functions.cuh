#pragma once

/**
 * @brief suppprts covar generation types for different diagonal operations
 *  - kNoOp: default, no operation
 *  - kAddNoise: add noise to the diagonal of the covar
 *  - kSetDiag: set the diagonal of the covar to `noise`
 */
enum class CovarType { kNoOp, kAddNoise, kSetDiag };

/**
 * @brief calculate the rbf kernel between two elements s1[0] and s2[col]
 *  - s1 is accessed contiguously, s2 is accessed with a stride of ns
 *
 * @param s1 pointer to the first dimension of the first element
 * @param s2 pointer to the start of all elements in shm for s2 (original)
 * @param ns number of elements in s2
 * @param col column number of the current element in s2
 */
__device__ inline float rbf(float* s1, float* s2, int ns, int d, int col,
                            float lengthscale) {
  float norm2 = 0.0;
  for (int i = col, j = 0; i < ns * d && j < d; i += ns, ++j)
    norm2 += (s1[j] - s2[i]) * (s1[j] - s2[i]);
  float power = -norm2 / (2 * lengthscale * lengthscale);
  return exp(power);
}

/**
 * @brief apply noise to one diagonal element within one block of covar
 *
 * @param K pointer to the start of the current block within covar
 * @param idx index of the diagonal element within the current block
 * @param ld leading dimension of entire covar
 * @param noise noise to apply to the diagonal element
 *
 * @return the new value of the diagonal element after the operation
 */
template <CovarType type>
__device__ float apply_diag_noise(float* K, int idx, int ld, float noise) {
  float val = K[idx * ld + idx];
  if constexpr (type == CovarType::kAddNoise) {
    val += noise;
  } else if constexpr (type == CovarType::kSetDiag) {
    val = noise;
  }
  K[idx * ld + idx] = val;
  return val;
}

/**
 * @brief calculate the confidence region given the covar and mean
 *
 * @param upper upper bound to be written
 * @param lower lower bound to be written
 * @param K full covariance matrix (without noise)
 * @param mean mean of the input data
 * @param n covar size (leading dimension)
 * @param noise squared noise to apply to the diagonal of the covar
 */
template <CovarType type = CovarType::kAddNoise>
__global__ void generate_confidence_region(float* upper, float* lower, float* K,
                                           float* mean, int n, float noise) {
  int bx = blockIdx.x, bd = blockDim.x, tx = threadIdx.x;
  int idx = bx * bd + tx;

  K += bx * bd * n + bx * bd;
  if (idx < n) {
    float sigma = sqrtf(apply_diag_noise<type>(K, tx, n, noise));
    upper[idx] = mean[idx] + 2 * sigma;
    lower[idx] = mean[idx] - 2 * sigma;
  }
}

/**
 * @brief generate covar / block diagonal covar in device memory
 *  - if called with 3D grid: one 2D grid -> one cluster in covar
 *  - one block -> one ms * ns block in one cluster, ms * d in x1, ns * d in x2
 *  - one thread -> ms * ns / bd elements in covar, with a stride of bd
 *
 * @param n1 number of elements to include in x1 in each **cluster**
 * @param n2 number of elements to include in x2 in each **cluster**
 * @param d input element dimension
 */
template <int ms, int ns, CovarType type = CovarType::kNoOp>
__global__ void generate_covar(float* covar, const float* x1, const float* x2,
                               int n1, int n2, int d, float lengthscale,
                               float noise, float output_scale) {
  extern __shared__ float s1[];
  float* s2 = s1 + ms * d;

  // for 3D (clustered) covar, bz = cluster number
  // for 2D (no cluster) covar, bz = 0 constant
  int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
  int tx = threadIdx.x, bd = blockDim.x;

  // move x1, x2 to start of current block
  x1 += bz * n1 * d + by * ms * d;
  x2 += bz * n2 * d + bx * ns * d;
  // ns{12} = number of elements in x1 and x2 to calculate for current block
  // l{12} = length of x1 and x2 to load and use for current block (including
  // dimension) specific case for last row and column of block
  int ns1 = min(ms, n1 - by * ms), ns2 = min(ns, n2 - bx * ns);
  int l1 = ns1 * d, l2 = ns2 * d;

  // load current block of x1 and x2 into shared memory
  for (int i = tx; i < l1; i += bd) s1[i] = x1[i];
  // load x2 so that one dimension of all elements are packed together
  // [[d0 of all elements], [d1 of all elements], ..., [d(d-1) of all elements]]
  // i % d = dimension of current float within current element
  // i / d = position of current element
  for (int i = tx; i < l2; i += bd) s2[(i % d) * ns2 + i / d] = x2[i];
  __syncthreads();

  // move covar to start of current block
  covar += bz * n1 * n2 + by * ms * n2 + bx * ns;
  for (int i = tx; i < ns1 * ns2; i += bd) {
    // row and column of current element within current block
    int row = i / ns2, col = i % ns2;
    float* s1_shifted = s1 + row * d;
    covar[row * n2 + col] =
        output_scale * rbf(s1_shifted, s2, ns2, d, col, lengthscale);
  }
  __syncthreads();

  // explicitly skip the if condition when no operation is needed
  if constexpr (type != CovarType::kNoOp) {
    if (bx == by && tx < ns2) apply_diag_noise<type>(covar, tx, n2, noise);
  }
}
