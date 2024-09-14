#include <chrono>

#include "device_common.cuh"
#include "gpmodel/precond.h"

DEFINE_bool(enable_cuda_streams, false, "Enable multiple CUDA streams");
DEFINE_bool(optimize_memory, false, "Optimize memory but limit some features");

namespace {

__global__ void log_and_scale(float* arr, int n, float scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) arr[idx] = logf(arr[idx]) * scale;
}

__global__ void trim_unused(float* R, int n, int b) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int rel_idx = idx % (b * b), i = rel_idx / b, j = rel_idx % b;
  if (idx < n && i > j) R[idx] = 0.0f;
}

__global__ void inv_horner(float* arr, int n, const float* x, const float* cs,
                           int csz) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float res = cs[0];
    for (int i = 1; i < csz; ++i) res = res * x[idx] + cs[i];
    arr[idx] = 1.0f / res;
  }
}

__global__ void balance(float* arr, int n, float* arr2) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float avg = (arr[idx] + arr2[idx]) * 0.5;
    arr[idx] = arr2[idx] = avg;
  }
}

template <typename Kernel, typename... Args>
void call_transform_kernel(Kernel kern, float* arr, int n, Args... args) {
  constexpr int block = 256;
  dim3 grid((n + block - 1) / block);
  kern<<<grid, block>>>(arr, n, args...);
}

}  // namespace

void BatchPreconditioner::solve_impl(const mmf& x, mmf& z) const {
  CHECK(x.m() == b_ * n_blocks_ && x.m() == z.m() && x.n() == z.n());
  CHECK(x.data() != z.data());

  mmf tmp(x.m(), x.n());

  // The below two-precond-pass solve should be a baseline that always works
  precond_impl(x, tmp, true);
  precond_impl(tmp, z, false);
}

void CholeskyPreconditioner::decompose() {
  if (!FLAGS_optimize_memory) R_.resize(A_.m(), A_.n());

  auto n = b_ * n_blocks_;
  auto portf_base = FLAGS_optimize_memory ? R_inv_.data() : R_.data();
  safe_cublas(cublasScopy, -1, A_.size(), A_.data(), 1, portf_base, 1);

  size_t d_sz, h_sz;
  mmf diags(n);
  ManagedMemory<int> info(1);
  for (int i = 0; i < n_blocks_; ++i) {
    auto potrf_block = portf_base + i * b_ * b_;
    auto R_inv_block = R_inv_.data() + i * b_ * b_;
    auto diags_block = diags.data() + i * b_;

    safe_cusolver(cusolverDnXpotrf_bufferSize, i, nullptr,
                  CUBLAS_FILL_MODE_LOWER, b_, CUDA_R_32F, potrf_block, b_,
                  CUDA_R_32F, &d_sz, &h_sz);

    ManagedMemory<char> d_work(d_sz);
    std::vector<char> h_work(h_sz);
    safe_cusolver(cusolverDnXpotrf, i, nullptr, CUBLAS_FILL_MODE_LOWER, b_,
                  CUDA_R_32F, potrf_block, b_, CUDA_R_32F, d_work.data(), d_sz,
                  h_work.data(), h_sz, info.data());

    safe_cublas(cublasScopy, i, b_, potrf_block, b_ + 1, diags_block, 1);
    if (!FLAGS_optimize_memory)
      safe_cublas(cublasScopy, i, b_ * b_, potrf_block, 1, R_inv_block, 1);

    safe_cusolver(cusolverDnXtrtri_bufferSize, i, CUBLAS_FILL_MODE_LOWER,
                  CUBLAS_DIAG_NON_UNIT, b_, CUDA_R_32F, R_inv_block, b_, &d_sz,
                  &h_sz);

    d_work.resize(d_sz);
    h_work.resize(h_sz);
    safe_cusolver(cusolverDnXtrtri, i, CUBLAS_FILL_MODE_LOWER,
                  CUBLAS_DIAG_NON_UNIT, b_, CUDA_R_32F, R_inv_block, b_,
                  d_work.data(), d_sz, h_work.data(), h_sz, info.data());
  }

  if (FLAGS_enable_cuda_streams) cudaDeviceSynchronize();

  call_transform_kernel(log_and_scale, diags.data(), n, 2.0f);
  call_transform_kernel(trim_unused, R_inv_.data(), n * b_, b_);

  logdet_ = 0.0;
  safe_cublas(cublasSdot, -1, n, diags.data(), 1, data_.ones.data(), 1,
              &logdet_.value());
}

void CholeskyPreconditioner::precond_impl(const mmf& x, mmf& z,
                                          bool transpose) const {
  CHECK(x.m() == b_ * n_blocks_ && x.m() == z.m() && x.n() == z.n());

  const float one = 1.0, zero = 0.0;
  auto R_op = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
  safe_cublas(cublasSgemmStridedBatched, -1, CUBLAS_OP_N, R_op, x.n(), b_, b_,
              &one, x.data(), x.n(), b_ * x.n(), R_inv_.data(), b_, b_ * b_,
              &zero, z.data(), z.n(), b_ * z.n(), n_blocks_);
}

void CholeskyPreconditioner::solve_impl(const mmf& x, mmf& z) const {
  if (FLAGS_optimize_memory)
    LOG(FATAL) << "Cholesky solve not avaliable in optimize memory mode";

  CHECK(x.m() == b_ * n_blocks_ && x.m() == z.m() && x.n() == z.n());
  CHECK(x.data() != z.data());

  const float one = 1.0, zero = 0.0;
  mmf zt(b_, z.n());
  ManagedMemory<int> info(1);
  for (int i = 0; i < n_blocks_; ++i) {
    auto R_block = R_.data() + i * b_ * b_;
    auto x_block = x.data() + i * b_ * x.n();
    auto z_block = z.data() + i * b_ * z.n();

    // z_tmp = x^T
    safe_cublas(cublasSgeam, i, CUBLAS_OP_T, CUBLAS_OP_N, b_, zt.n(), &one,
                x_block, x.n(), &zero, x_block, b_, zt.data(), b_);

    // solve R * z_tmp = x^T
    safe_cusolver(cusolverDnXpotrs, i, nullptr, CUBLAS_FILL_MODE_LOWER, b_,
                  zt.n(), CUDA_R_32F, R_block, b_, CUDA_R_32F, zt.data(), b_,
                  info.data());

    // z = z_tmp^T
    safe_cublas(cublasSgeam, i, CUBLAS_OP_T, CUBLAS_OP_N, z.n(), b_, &one,
                zt.data(), b_, &zero, zt.data(), zt.n(), z_block, z.n());
  }

  if (FLAGS_enable_cuda_streams) cudaDeviceSynchronize();
}

void GesvdrPreconditioner::decompose() {
  auto start_ts = std::chrono::high_resolution_clock::now();

  cusolverDnParams_t params;
  CHECK(cusolverDnCreateParams(&params) == CUSOLVER_STATUS_SUCCESS);

  mmf tmp(A_.m(), A_.n());
  safe_cublas(cublasScopy, -1, A_.size(), A_.data(), 1, tmp.data(), 1);

  const int64_t p = 0, niters = 8;
  size_t d_sz, h_sz;
  ManagedMemory<int> info(1);
  for (int i = 0; i < n_blocks_; ++i) {
    auto A_block = tmp.data() + i * b_ * b_;
    auto S_block = S_.data() + i * rs_;
    auto U_block = U_.data() + i * b_ * rs_;
    auto V_block = V_.data() + i * b_ * rs_;

    safe_cusolver(cusolverDnXgesvdr_bufferSize, i, params, 'S', 'S', b_, b_,
                  rs_, p, niters, CUDA_R_32F, A_block, b_, CUDA_R_32F, S_block,
                  CUDA_R_32F, U_block, b_, CUDA_R_32F, V_block, b_, CUDA_R_32F,
                  &d_sz, &h_sz);

    ManagedMemory<char> d_work(d_sz);
    std::vector<char> h_work(h_sz);
    safe_cusolver(cusolverDnXgesvdr, i, params, 'S', 'S', b_, b_, rs_, p,
                  niters, CUDA_R_32F, A_block, b_, CUDA_R_32F, S_block,
                  CUDA_R_32F, U_block, b_, CUDA_R_32F, V_block, b_, CUDA_R_32F,
                  d_work.data(), d_sz, h_work.data(), h_sz, info.data());
  }

  if (FLAGS_enable_cuda_streams) cudaDeviceSynchronize();

  call_transform_kernel(balance, U_.data(), U_.size(), V_.data());

  CHECK(cusolverDnDestroyParams(params) == CUSOLVER_STATUS_SUCCESS);

  // Using gesvdr we do not need to compensate for logdet, so just set it to 0
  logdet_ = 0.0;

  auto dur_us = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::high_resolution_clock::now() - start_ts);
  LOG(INFO) << "gesvdr took " << dur_us.count() / 1e3 << " ms";
}

void GesvdrPreconditioner::set_coeffs(vf coeffs) {
  mmf cs(coeffs.size());
  dmemcpy<TransferType::kHostToDevice>(cs.data(), coeffs.data(), cs.size());
  call_transform_kernel(inv_horner, P_.data(), P_.size(), S_.data(), cs.data(),
                        cs.size());
}

void GesvdrPreconditioner::solve_impl(const mmf& x, mmf& z) const {
  CHECK(logdet_.has_value());
  CHECK(x.m() == b_ * n_blocks_ && x.m() == z.m() && x.n() == z.n());
  CHECK(x.data() != z.data());

  mmf tmp(n_blocks_ * rs_, x.n());

  const float one = 1.0, zero = 0.0;
  // tmp = U^T * x
  safe_cublas(cublasSgemmStridedBatched, -1, CUBLAS_OP_N, CUBLAS_OP_N, x.n(),
              rs_, b_, &one, x.data(), x.n(), b_ * x.n(), V_.data(), b_,
              b_ * rs_, &zero, tmp.data(), tmp.n(), rs_ * tmp.n(), n_blocks_);

  // tmp = diag(P) * tmp
  safe_cublas(cublasSdgmm, -1, CUBLAS_SIDE_RIGHT, tmp.n(), tmp.m(), tmp.data(),
              tmp.n(), P_.data(), 1, tmp.data(), tmp.n());

  // z = V * tmp
  safe_cublas(cublasSgemmStridedBatched, -1, CUBLAS_OP_N, CUBLAS_OP_T, z.n(),
              b_, rs_, &one, tmp.data(), tmp.n(), rs_ * tmp.n(), U_.data(), b_,
              b_ * rs_, &zero, z.data(), z.n(), b_ * z.n(), n_blocks_);
}
