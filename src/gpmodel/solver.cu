#include "device_common.cuh"
#include "gpmodel/solver.h"

namespace {

__global__ void calc_alpha_beta(float* r1, float* r2, float* res, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) res[idx] = r2[idx] == 0.0 ? 0.0 : r1[idx] / r2[idx];
}

__global__ void check_convergence(float* r, float* mask, float tol, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) mask[idx] = r[idx] < tol ? 0.0 : 1.0;
}

template <typename Kernel, typename... Args>
void call_cg_kernel(Kernel kern, int n, Args... args) {
  constexpr int block = 32;
  dim3 grid((n + block - 1) / block);
  kern<<<grid, block>>>(args..., n);
}

}  // namespace

void BatchCGSolver::batch_dot(const mmf& m1, const mmf& m2, mmf& res) const {
  CHECK(m1.m() == m2.m() && m1.n() == m2.n() && m1.n() == res.size());

  mmf tmp(m1.m(), m1.n());

  const float one = 1.0, zero = 0.0;
  // tmp = A (element-wise-multiply) B
  safe_cublas(cublasSdgmm, -1, CUBLAS_SIDE_LEFT, m1.size(), 1, m1.data(),
              m1.size(), m2.data(), 1, tmp.data(), tmp.size());
  // res = 1^T * tmp
  safe_cublas(cublasSgemm, -1, CUBLAS_OP_N, CUBLAS_OP_N, tmp.n(), 1, tmp.m(),
              &one, tmp.data(), tmp.n(), data_.ones.data(), tmp.m(), &zero,
              res.data(), res.size());
}

void BatchCGSolver::bcg(const DeviceRep& A, mmf& x, const mmf& b) const {
  CHECK(x.m() == b.m() && x.n() == b.n());
  CHECK(x.data() != b.data());

  auto m = x.m(), n = x.n();
  mmf residual(m, n), p(m, n), Kp(m, n), tmp(m, n);
  mmf alpha(n), beta(n), r_prev(n), r_cur(n), mask(n);

  // x = 0
  dmemset(x.data(), 0.0f, x.size());
  // residual = b
  safe_cublas(cublasScopy, -1, b.size(), b.data(), 1, residual.data(), 1);
  // p = b
  safe_cublas(cublasScopy, -1, b.size(), b.data(), 1, p.data(), 1);
  // r_prev = residual (batch-dot) residual
  batch_dot(residual, residual, r_prev);

  const float one = 1.0, negone = -1.0;
  size_t i = 0;
  float prev_remain = n, remain = n;
  while (i++ < max_iter_) {
    // Kp = A * p
    A.left_mm(p, Kp);
    // r_cur = p (batch-dot) Kp
    batch_dot(p, Kp, r_cur);
    // alpha = r_prev (element-wise-divide) r_cur
    call_cg_kernel(calc_alpha_beta, n, r_prev.data(), r_cur.data(),
                   alpha.data());

    // tmp = Kp * diag(alpha)
    safe_cublas(cublasSdgmm, -1, CUBLAS_SIDE_LEFT, n, m, Kp.data(), n,
                alpha.data(), 1, tmp.data(), n);
    // residual -= tmp
    safe_cublas(cublasSaxpy, -1, tmp.size(), &negone, tmp.data(), 1,
                residual.data(), 1);
    // tmp = p * diag(alpha)
    safe_cublas(cublasSdgmm, -1, CUBLAS_SIDE_LEFT, n, m, p.data(), n,
                alpha.data(), 1, tmp.data(), n);
    // x += tmp
    safe_cublas(cublasSaxpy, -1, tmp.size(), &one, tmp.data(), 1, x.data(), 1);

    // r_cur = residual (batch-dot) residual
    batch_dot(residual, residual, r_cur);
    // mask = (r_cur >= tol * tol)
    call_cg_kernel(check_convergence, n, r_cur.data(), mask.data(),
                   tol_ * tol_);
    // remain = sum(mask) = mask * 1
    safe_cublas(cublasSdot, -1, n, mask.data(), 1, data_.ones.data(), 1,
                &remain);

    if (remain == 0) break;
    if (remain < prev_remain) {
      // mask converged columns in p and residual to 0
      safe_cublas(cublasSdgmm, -1, CUBLAS_SIDE_LEFT, n, m, p.data(), n,
                  mask.data(), 1, p.data(), n);
      safe_cublas(cublasSdgmm, -1, CUBLAS_SIDE_LEFT, n, m, residual.data(), n,
                  mask.data(), 1, residual.data(), n);
      prev_remain = remain;
    }

    // beta = r_cur (element-wise-divide) r_prev
    call_cg_kernel(calc_alpha_beta, n, r_cur.data(), r_prev.data(),
                   beta.data());
    // p *= diag(beta)
    safe_cublas(cublasSdgmm, -1, CUBLAS_SIDE_LEFT, n, m, p.data(), n,
                beta.data(), 1, p.data(), n);
    // p += residual
    safe_cublas(cublasSaxpy, -1, residual.size(), &one, residual.data(), 1,
                p.data(), 1);
    // r_prev = r_cur
    safe_cublas(cublasScopy, -1, n, r_cur.data(), 1, r_prev.data(), 1);
  }

  LOG(INFO) << "Number of CG iterations: " << std::min(i, max_iter_);
}

void BatchCGSolver::zbcg(const DeviceRep& A, const BatchPreconditioner& bp,
                         mmf& x, const mmf& b) const {
  CHECK(x.m() == b.m() && x.n() == b.n());
  CHECK(x.data() != b.data());

  auto m = x.m(), n = x.n();
  mmf residual(m, n), p(m, n), Kp(m, n), z(m, n), tmp(m, n);
  mmf alpha(n), beta(n), r_prev(n), r_cur(n), mask(n);

  // x = 0
  dmemset(x.data(), 0.0f, x.size());
  // residual = b
  safe_cublas(cublasScopy, -1, b.size(), b.data(), 1, residual.data(), 1);
  // solve M * z = residual
  bp.solve(residual, z);
  // p = z
  safe_cublas(cublasScopy, -1, z.size(), z.data(), 1, p.data(), 1);
  // r_prev = residual (batch-dot) z
  batch_dot(residual, z, r_prev);

  const float one = 1.0, negone = -1.0;
  size_t i = 0;
  float prev_remain = n, remain = n;
  while (i++ < max_iter_) {
    // Kp = A * p
    A.left_mm(p, Kp);
    // r_cur = p (batch-dot) Kp
    batch_dot(p, Kp, r_cur);
    // alpha = r_prev (element-wise-divide) r_cur
    call_cg_kernel(calc_alpha_beta, n, r_prev.data(), r_cur.data(),
                   alpha.data());

    // tmp = Kp * diag(alpha)
    safe_cublas(cublasSdgmm, -1, CUBLAS_SIDE_LEFT, n, m, Kp.data(), n,
                alpha.data(), 1, tmp.data(), n);
    // residual -= tmp
    safe_cublas(cublasSaxpy, -1, tmp.size(), &negone, tmp.data(), 1,
                residual.data(), 1);
    // tmp = p * diag(alpha)
    safe_cublas(cublasSdgmm, -1, CUBLAS_SIDE_LEFT, n, m, p.data(), n,
                alpha.data(), 1, tmp.data(), n);
    // x += tmp
    safe_cublas(cublasSaxpy, -1, tmp.size(), &one, tmp.data(), 1, x.data(), 1);

    // r_cur = residual (batch-dot) residual
    batch_dot(residual, residual, r_cur);
    // mask = (r_cur >= tol * tol)
    call_cg_kernel(check_convergence, n, r_cur.data(), mask.data(),
                   tol_ * tol_);
    // remain = sum(mask) = mask * 1
    safe_cublas(cublasSdot, -1, n, mask.data(), 1, data_.ones.data(), 1,
                &remain);

    if (remain == 0) break;
    if (remain < prev_remain) {
      // mask converged columns in p, residual, and z to 0
      safe_cublas(cublasSdgmm, -1, CUBLAS_SIDE_LEFT, n, m, p.data(), n,
                  mask.data(), 1, p.data(), n);
      safe_cublas(cublasSdgmm, -1, CUBLAS_SIDE_LEFT, n, m, residual.data(), n,
                  mask.data(), 1, residual.data(), n);
      prev_remain = remain;
    }

    // solve M * z = residual
    bp.solve(residual, z);
    {
      vf h(10, 0.0);
      dmemcpy<TransferType::kDeviceToHost>(h.data(), residual.data(), h.size());
      std::stringstream ss;
      ss << "residual = ";
      for (auto x : h) ss << x << " ";
      LOG(INFO) << ss.str();
    }
    {
      vf h(10, 0.0);
      dmemcpy<TransferType::kDeviceToHost>(h.data(), z.data(), h.size());
      std::stringstream ss;
      ss << "z = ";
      for (auto x : h) ss << x << " ";
      LOG(INFO) << ss.str();
    }
    // r_cur = residual (batch-dot) z
    batch_dot(residual, z, r_cur);
    // beta = r_cur (element-wise-divide) r_prev
    call_cg_kernel(calc_alpha_beta, n, r_cur.data(), r_prev.data(),
                   beta.data());
    // p *= diag(beta)
    safe_cublas(cublasSdgmm, -1, CUBLAS_SIDE_LEFT, n, m, p.data(), n,
                beta.data(), 1, p.data(), n);
    // p += z
    safe_cublas(cublasSaxpy, -1, z.size(), &one, z.data(), 1, p.data(), 1);
    // r_prev = r_cur
    safe_cublas(cublasScopy, -1, n, r_cur.data(), 1, r_prev.data(), 1);
  }

  LOG(INFO) << "Number of CG iterations: " << std::min(i, max_iter_);

  LOG(FATAL) << "intentional crash";
}

/**
 * @details the steps to solve preconditioned batched CG are:
 *  1. use precond to transform b to b_hat
 *  2. use CG to solve precond(A) * x_hat = b_hat
 *  3. use precond to transform x_hat to x
 */
void BatchCGSolver::pbcg(const DeviceRep& A, const BatchPreconditioner& bp,
                         mmf& x, const mmf& b, bool precond_done) const {
  CHECK(x.m() == b.m() && x.n() == b.n());
  CHECK(x.data() != b.data());

  auto m = x.m(), n = x.n();
  mmf b_hat(m, n), x_hat(m, n);

  // step 1: b_hat = precond(b) with transpose
  bp.precond(b, b_hat, true);
  // step 2: now A_hat precond(A), call bcg to solve A_hat * x_hat = b_hat
  if (precond_done) {
    bcg(A, x_hat, b_hat);
  } else {
    // The smush applies preconditioner to A and lets bcg do left_mm properly
    DRPrecondSmush A_hat(data_, A, bp);
    bcg(A_hat, x_hat, b_hat);
  }
  // step 3: finally, x = precond(x_hat)
  bp.precond(x_hat, x, false);
}
