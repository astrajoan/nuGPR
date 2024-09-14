#include <chrono>
#include <numeric>
#include <random>
#include <vector>

#include "device_common.cuh"
#include "device_functions.cuh"
#include "device_test_helper.h"

namespace {

template <CovarType type>
void call_gencovar_kernel(float* covar, float* x1, float* x2, size_t n1,
                          size_t n2, size_t d, size_t n_blocks,
                          float lengthscale, float noise, float output_scale) {
  constexpr int ms = 64, ns = 64;
  constexpr int block = 256;
  dim3 grid((n2 + ns - 1) / ns, (n1 + ms - 1) / ms, n_blocks);
  auto sm_size = (ms + ns) * d * sizeof(float);
  generate_covar<ms, ns, type><<<grid, block, sm_size>>>(
      covar, x1, x2, n1, n2, d, lengthscale, noise, output_scale);
}

}  // namespace

void dmemcpy_wrapper(const float* cpu, float* gpu, size_t sz, bool to_gpu) {
  CHECK(to_gpu) << "Invalid transfer type for this call";
  dmemcpy<TransferType::kHostToDevice>(gpu, cpu, sz);
}

void dmemcpy_wrapper(float* cpu, const float* gpu, size_t sz, bool to_gpu) {
  CHECK(!to_gpu) << "Invalid transfer type for this call";
  dmemcpy<TransferType::kDeviceToHost>(cpu, gpu, sz);
}

void dmemcpy_wrapper(float* cpu, float* gpu, size_t sz, bool to_gpu) {
  if (to_gpu)
    dmemcpy_wrapper(static_cast<const float*>(cpu), gpu, sz, true);
  else
    dmemcpy_wrapper(cpu, static_cast<const float*>(gpu), sz, false);
}

void init_mock_data(GPData& data, size_t ones_sz) {
  data.ones.resize(ones_sz);
  dmemset(data.ones.data(), 1.0f, ones_sz);
}

void init_mock_data(GPData& data, const vf& train_x, const vf& train_y,
                    const vf& test_x, const vf& rep_x, const vf& rand_z,
                    size_t n_train, size_t n_test, size_t n_rand,
                    size_t n_blocks, size_t d) {
  data.n_train = n_train, data.n_test = n_test, data.n_rand = n_rand;
  data.n_blocks = n_blocks, data.x_dim = d;

  data.train_x.resize(n_train, d);
  data.train_y.resize(n_train);
  data.test_x.resize(n_test, d);
  data.rep_x.resize(n_blocks, d);
  data.rand_z.resize(n_train, n_rand);
  data.ones.resize(std::max(std::max(n_train, n_test), n_rand));
  dmemset(data.ones.data(), 1.0f, data.ones.size());

  dmemcpy<TransferType::kHostToDevice>(
      data.train_x.data(), train_x.data(), n_train * d, data.train_y.data(),
      train_y.data(), n_train, data.test_x.data(), test_x.data(), n_test * d,
      data.rep_x.data(), rep_x.data(), n_blocks * d, data.rand_z.data(),
      rand_z.data(), n_train * n_rand);
}

void generate_covar_gpu(vf& covar, const vf& x1, const vf& x2, size_t d,
                        size_t n_blocks, float lengthscale, float noise,
                        float output_scale) {
  auto n1 = x1.size() / n_blocks / d, n2 = x2.size() / n_blocks / d;
  mmf d_covar(n1, n2), d_x1(x1.size()), d_x2(x2.size());

  dmemcpy<TransferType::kHostToDevice>(d_x1.data(), x1.data(), x1.size(),
                                       d_x2.data(), x2.data(), x2.size());

  if (n1 == n2) {
    call_gencovar_kernel<CovarType::kAddNoise>(
        d_covar.data(), d_x1.data(), d_x2.data(), n1, n2, d, n_blocks,
        lengthscale, noise, output_scale);
  } else {
    call_gencovar_kernel<CovarType::kNoOp>(  // !
        d_covar.data(), d_x1.data(), d_x2.data(), n1, n2, d, n_blocks,
        lengthscale, noise, output_scale);
  }

  dmemcpy<TransferType::kDeviceToHost>(covar.data(), d_covar.data(),
                                       covar.size());
}

void generate_confidence_region_gpu(vf& upper, vf& lower, const vf& mean,
                                    const vf& covar, size_t n, float noise) {
  mmf d_covar(n, n), d_mean(n), d_upper(n), d_lower(n);

  dmemcpy<TransferType::kHostToDevice>(d_covar.data(), covar.data(), n * n,
                                       d_mean.data(), mean.data(), n);

  constexpr int ns = 32;
  constexpr int block = 32;
  auto grid = (n + ns - 1) / ns;
  generate_confidence_region<<<grid, block>>>(
      d_upper.data(), d_lower.data(), d_covar.data(), d_mean.data(), n, noise);

  dmemcpy<TransferType::kDeviceToHost>(upper.data(), d_upper.data(), n,
                                       lower.data(), d_lower.data(), n);
}

bool chol_softness(const vf& A, vf& L, size_t n) {
  // iterate over each row in A
  for (int i = 0; i < n; ++i) {
    // iterate over each column until the ith in A
    for (int j = 0; j <= i; ++j) {
      float sum = 0.0;
      // dot product of ith and jth row until column j
      for (int k = 0; k < j; ++k) sum += L[i * n + k] * L[j * n + k];
      float val = A[i * n + j] - sum;

      if (i == j && val < 0.0) return false;
      L[i * n + j] = i == j ? std::sqrt(val) : val / L[j * n + j];
    }
  }
  return true;
}

void syevd_test_impl() {
  size_t n_blocks = 10, d = 1, sz = 2000;
  vf h_rep_x{
      -10.166123, -7.901061, -5.627124, -3.365081, -1.141672,
      1.143030,   3.388243,  5.643469,  7.896815,  10.165068,
  };

  vf A(n_blocks * n_blocks), L(n_blocks * n_blocks);
  vf lengthscales(sz), res(sz);
  for (int i = 0; i < sz; ++i) lengthscales[i] = (i + 100) * 1e-3;

  mmf rep_x(n_blocks), covar(n_blocks, n_blocks), sigma(n_blocks * sz);
  dmemcpy<TransferType::kHostToDevice>(rep_x.data(), h_rep_x.data(), n_blocks);

  auto start_ts = std::chrono::high_resolution_clock::now();

  ManagedMemory<int> info(1);
  for (int i = 0; i < sz; ++i) {
    call_gencovar_kernel<CovarType::kNoOp>(covar.data(), rep_x.data(),
                                           rep_x.data(), n_blocks, n_blocks, d,
                                           1, lengthscales[i], 0.0, 1.0);

    auto sigma_block = sigma.data() + i * n_blocks;
    size_t d_sz, h_sz;
    safe_cusolver(cusolverDnXsyevd_bufferSize, -1, nullptr,
                  CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n_blocks,
                  CUDA_R_32F, covar.data(), n_blocks, CUDA_R_32F, sigma_block,
                  CUDA_R_32F, &d_sz, &h_sz);

    ManagedMemory<char> d_work(d_sz);
    std::vector<char> h_work(h_sz);
    safe_cusolver(cusolverDnXsyevd, -1, nullptr, CUSOLVER_EIG_MODE_VECTOR,
                  CUBLAS_FILL_MODE_LOWER, n_blocks, CUDA_R_32F, covar.data(),
                  n_blocks, CUDA_R_32F, sigma_block, CUDA_R_32F, d_work.data(),
                  d_sz, h_work.data(), h_sz, info.data());
  }

  auto dur_us = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::high_resolution_clock::now() - start_ts);
  LOG(ERROR) << "syevd total time " << dur_us.count() / 1e3 << " ms";
  LOG(ERROR) << "per solve time " << dur_us.count() / 1e3 / sz << " ms";

  start_ts = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < sz; ++i) {
    call_gencovar_kernel<CovarType::kNoOp>(covar.data(), rep_x.data(),
                                           rep_x.data(), n_blocks, n_blocks, d,
                                           1, lengthscales[i], 0.0, 1.0);

    dmemcpy<TransferType::kDeviceToHost>(A.data(), covar.data(), A.size());

    float l = 0, r = A[0], eps = 1e-6, mid;
    while (l < r - eps) {
      mid = l * 0.5 + r * 0.5;
      for (int i = 0; i < n_blocks; ++i) A[i * n_blocks + i] = mid;
      if (chol_softness(A, L, n_blocks))
        r = mid;
      else
        l = mid;
    }
    res[i] = mid;
  }

  dur_us = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::high_resolution_clock::now() - start_ts);
  LOG(ERROR) << "potrf total time " << dur_us.count() / 1e3 << " ms";
  LOG(ERROR) << "per solve time " << dur_us.count() / 1e3 / sz << " ms";

  vf h_sigma(sigma.size());
  dmemcpy<TransferType::kDeviceToHost>(h_sigma.data(), sigma.data(),
                                       sigma.size());

  for (int i = 0; i < sz; ++i) {
    LOG(ERROR) << "lengthscale = " << lengthscales[i]
               << ", syevd = " << 1.0 - h_sigma[i * n_blocks]
               << ", potrf = " << res[i];
  }
}

void syevdx_test_impl() {
  vf h_mat{1.5, -0.5, 0, 0, 0, 0, 0, 0, -0.5, 1.5, 0, 0, 0, 0, 0, 0, 0, 0,
           3.5, -0.5, 0, 0, 0, 0, 0, 0, -0.5, 3.5, 0, 0, 0, 0, 0, 0, 0, 0,
           5.5, -0.5, 0, 0, 0, 0, 0, 0, -0.5, 5.5, 0, 0, 0, 0, 0, 0, 0, 0,
           7.5, -0.5, 0, 0, 0, 0, 0, 0, -0.5, 7.5};

  cusolverDnParams_t params;
  cusolverDnCreateParams(&params);

  constexpr int64_t n = 8, k = 4, p = 4, niters = 8;
  mmf mat(n, n), S(k), U(n, k), V(n, k);
  dmemcpy<TransferType::kHostToDevice>(mat.data(), h_mat.data(), mat.size());
  dmemset(S.data(), 0.0f, S.size(), U.data(), 0.0f, U.size(), V.data(), 0.0f,
          V.size());

  size_t d_sz, h_sz;
  safe_cusolver(cusolverDnXgesvdr_bufferSize, -1, params, 'S', 'S', n, n, k, p,
                niters, CUDA_R_32F, mat.data(), n, CUDA_R_32F, S.data(),
                CUDA_R_32F, U.data(), n, CUDA_R_32F, V.data(), n, CUDA_R_32F,
                &d_sz, &h_sz);

  ManagedMemory<char> d_work(d_sz);
  std::vector<char> h_work(h_sz);
  ManagedMemory<int> info(1);
  safe_cusolver(cusolverDnXgesvdr, -1, params, 'S', 'S', n, n, k, p, niters,
                CUDA_R_32F, mat.data(), n, CUDA_R_32F, S.data(), CUDA_R_32F,
                U.data(), n, CUDA_R_32F, V.data(), n, CUDA_R_32F, d_work.data(),
                d_sz, h_work.data(), h_sz, info.data());

  vf h_x1{9, 10, 11, 12, 13, 14, 15, 16};
  mmf x1(n, 1), tmp(n, 1), z1(n, 1), z2(n, 1);
  dmemcpy<TransferType::kHostToDevice>(x1.data(), h_x1.data(), x1.size());

  const float one = 1.0, zero = 0.0;
  int64_t rs_ = k, b_ = n, n_blocks_ = 1;
  safe_cublas(cublasSgemmStridedBatched, -1, CUBLAS_OP_N, CUBLAS_OP_N, x1.n(),
              rs_, b_, &one, x1.data(), x1.n(), b_ * x1.n(), V.data(), b_,
              b_ * rs_, &zero, tmp.data(), tmp.n(), rs_ * tmp.n(), n_blocks_);

  safe_cublas(cublasSdgmm, -1, CUBLAS_SIDE_RIGHT, tmp.n(), tmp.m(), tmp.data(),
              tmp.n(), S.data(), 1, tmp.data(), tmp.n());

  safe_cublas(cublasSgemmStridedBatched, -1, CUBLAS_OP_N, CUBLAS_OP_T, z1.n(),
              b_, rs_, &one, tmp.data(), tmp.n(), rs_ * tmp.n(), U.data(), b_,
              b_ * rs_, &zero, z1.data(), z1.n(), b_ * z1.n(), n_blocks_);

  safe_cublas(cublasSgemv, -1, CUBLAS_OP_N, n, n, &one, mat.data(), n,
              x1.data(), 1, &zero, z2.data(), 1);

  vf h_mat2(h_mat.size()), h_S(S.size()), h_U(U.size()), h_V(V.size());
  vf h_z1(z1.size());
  dmemcpy<TransferType::kDeviceToHost>(
      h_mat2.data(), mat.data(), mat.size(), h_S.data(), S.data(), S.size(),
      h_U.data(), U.data(), U.size(), h_V.data(), V.data(), V.size(),
      h_z1.data(), z1.data(), z1.size());

  cusolverDnDestroyParams(params);

  {
    std::stringstream ss;
    ss << "h_mat2:\n";
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        ss << h_mat2[i * n + j] << "\t";
      }
      ss << "\n";
    }
    LOG(ERROR) << ss.str();
  }
  {
    std::stringstream ss;
    ss << "h_S:\n";
    for (size_t i = 0; i < k; ++i) {
      ss << h_S[i] << "\t";
    }
    LOG(ERROR) << ss.str();
  }
  {
    std::stringstream ss;
    ss << "h_U:\n";
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < k; ++j) {
        ss << h_U[i * k + j] << "\t";
      }
      ss << "\n";
    }
    LOG(ERROR) << ss.str();
  }
  {
    std::stringstream ss;
    ss << "h_V:\n";
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < k; ++j) {
        ss << h_V[i * k + j] << "\t";
      }
      ss << "\n";
    }
    LOG(ERROR) << ss.str();
  }
  {
    std::stringstream ss;
    ss << "h_z1:\n";
    for (size_t i = 0; i < n; ++i) {
      ss << h_z1[i] << "\t";
    }
    LOG(ERROR) << ss.str();
  }
}
