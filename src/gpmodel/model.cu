#include <numeric>
#include <sstream>

#include "device_common.cuh"
#include "device_functions.cuh"
#include "gpmodel/model.h"

DEFINE_bool(rep, true, "Use representation points for off-diagonal blocks");
DEFINE_bool(super, false, "Use super Cholesky for train forward");
DEFINE_bool(combine_cg, false, "Combine K^{-1} * y and logdet CG procedures");

namespace {

__global__ void simple_add(float* covar, int n, float val) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) covar[idx] += val;
}

}  // namespace

GPModel::GPModel(const GPData& data)
    : data_(data),
      lengthscale_(0.0),
      noise_(0.0),
      output_scale_(0.0),
      solve_(data.n_train),
      decompose_(true),
      gamma_(1.0),
      delta_(0.0) {
  covar_ = std::make_unique<DRCovarODLR>(data, data.n_train, data.n_blocks);
  if (!FLAGS_rep) covar_full_ = std::make_unique<DRCovar>(data, data.n_train);
  cgsolver_ = std::make_unique<BatchCGSolver>(data, 200, 0.01);
}

float GPModel::train_forward() {
  LOG(INFO) << "Running forward propagation in training mode";

  auto n_train = data_.n_train, x_dim = data_.x_dim, n_blocks = data_.n_blocks;
  auto block_sz = n_train / n_blocks;
  // ms, ns = block row and column size, number of elements per block
  constexpr int ms = 64, ns = 64;
  constexpr int block = 256;
  auto sm_size = (ms + ns) * x_dim * sizeof(float);

  if (!FLAGS_rep) {
    dim3 grid((n_train + ns - 1) / ns, (n_train + ms - 1) / ms);
    generate_covar<ms, ns, CovarType::kAddNoise><<<grid, block, sm_size>>>(
        covar_full_->K_data(), data_.train_x.data(), data_.train_x.data(),
        n_train, n_train, x_dim, lengthscale(), noise(), output_scale());
  }

  // launch 3D grid, each 2D grid handles one cluster of covar_
  dim3 grid0((block_sz + ns - 1) / ns, (block_sz + ms - 1) / ms, n_blocks);
  generate_covar<ms, ns, CovarType::kAddNoise><<<grid0, block, sm_size>>>(
      covar_->K_diag_data(), data_.train_x.data(), data_.train_x.data(),
      block_sz, block_sz, x_dim, lengthscale(), noise(), output_scale());

  // compute low rank representation of off-diagonal blocks
  constexpr int rs = 8;
  dim3 grid1((n_blocks + rs - 1) / rs, (n_blocks + rs - 1) / rs);
  sm_size = 2 * rs * x_dim * sizeof(float);
  auto tmp_os = output_scale();
  generate_covar<rs, rs, CovarType::kSetDiag><<<grid1, block, sm_size>>>(
      covar_->K_odlr_data(), data_.rep_x.data(), data_.rep_x.data(), n_blocks,
      n_blocks, x_dim, lengthscale(), 0.0, FLAGS_super ? 1.0 : tmp_os);

  if (FLAGS_super) {
    auto cpu_sz = n_blocks * n_blocks;
    dmemcpy<TransferType::kDeviceToHost>(chol_cpu().K_odlr_data(),
                                         covar_->K_odlr_data(), cpu_sz);

    auto comp = chol_cpu().binary_search_chol();
    // safe_cublas(cublasSaxpy, -1, n_blocks, &comp, data_.ones.data(), 1,
    //             covar_->K_odlr_data(), n_blocks + 1);
    auto diag_sz = covar_->K_diag().size();
    simple_add<<<(diag_sz + block - 1) / block, block>>>(
        covar_->K_diag_data(), diag_sz, comp * tmp_os);
    safe_cublas(cublasSscal, -1, cpu_sz, &tmp_os, covar_->K_odlr_data(), 1);
  }

  if (FLAGS_combine_cg)
    return -log_prob_combined() / data_.n_train;
  else
    return -log_prob() / data_.n_train;
}

float GPModel::log_prob_combined() {
  auto n_train = data_.n_train, n_blocks = data_.n_blocks;
  auto n_rand = data_.n_rand;

  if (decompose_) precond().decompose();

  SuperCholeskySmush chol_smush(data_, get_covar(), precond(), n_train,
                                n_blocks, gamma_, delta_);

  // prepare for combined CG
  const vf Q_coeffs{0.25, 1.0, 0.25};
  SuperCholeskyHornerSmush schs(data_, chol_smush, 1, n_rand - 1, Q_coeffs);

  mmf lhs(n_train, n_rand), rhs(n_train, n_rand), solve2(n_train);
  safe_cublas(cublasScopy, -1, data_.rand_z.size(), data_.rand_z.data(), 1,
              rhs.data(), 1);
  precond().precond(data_.train_y, solve2, true);
  safe_cublas(cublasScopy, -1, n_train, solve2.data(), 1,
              rhs.data() + n_rand - 1, rhs.n());

  cgsolver_->bcg(schs, lhs, rhs);

  safe_cublas(cublasScopy, -1, n_train, lhs.data() + n_rand - 1, lhs.n(),
              solve2.data(), 1);
  precond().precond(solve2, solve_, false);

  // solve_ = K^{-1} * y
  float y_Kinv_y;
  safe_cublas(cublasSdot, -1, n_train, data_.train_y.data(), 1, solve_.data(),
              1, &y_Kinv_y);

  const vf P_coeffs{0.75, 0.0, -0.75};
  DRHornerSmush horner_smush_P(data_, chol_smush, P_coeffs);

  mmf q(n_train, n_rand), r(n_rand);
  horner_smush_P.left_mm(lhs, q);
  cgsolver_->batch_dot(data_.rand_z, q, r);
  mmf k_invs(n_rand - 1);
  dmemset(k_invs.data(), 1.0f / (n_rand - 1), n_rand - 1);

  // compute logdet
  float logdet_K;
  safe_cublas(cublasSdot, -1, n_rand - 1, r.data(), 1, k_invs.data(), 1,
              &logdet_K);
  logdet_K += precond().logdet();

  if (FLAGS_super && param_ == "noise" && binder()) {
    auto mul = p0_ / p1_, inv_mul = p1_ / p0_;
    auto step = inv_softplus(output_scale() * mul) - output_scale_;
    auto loss = -0.5 * (inv_mul * y_Kinv_y + n_train * log(mul) + logdet_K +
                        n_train * log(2 * M_PI));
    auto e1 = -loss / n_train;
    binder()->grad1 = (e1 - binder()->e0) / step;
  }

  return -0.5 * (y_Kinv_y + logdet_K + n_train * log(2 * M_PI));
}

float GPModel::log_prob() {
  auto n_train = data_.n_train, n_blocks = data_.n_blocks;

  if (decompose_) precond().decompose();

  SuperCholeskySmush chol_smush(data_, get_covar(), precond(), n_train,
                                n_blocks, gamma_, delta_);

  // solve_ = K^{-1} * y
  cgsolver_->pbcg(chol_smush, precond(), solve_, data_.train_y, true);

  // y_Ki_y = y^T * K^{-1} * y = y (dot) solve_
  float y_Kinv_y;
  safe_cublas(cublasSdot, -1, n_train, data_.train_y.data(), 1, solve_.data(),
              1, &y_Kinv_y);

  // compute logdet
  auto logdet_K = logdet(chol_smush);

  if (FLAGS_super && param_ == "noise" && binder()) {
    auto mul = p0_ / p1_, inv_mul = p1_ / p0_;
    auto step = inv_softplus(output_scale() * mul) - output_scale_;
    auto loss = -0.5 * (inv_mul * y_Kinv_y + n_train * log(mul) + logdet_K +
                        n_train * log(2 * M_PI));
    auto e1 = -loss / n_train;
    binder()->grad1 = (e1 - binder()->e0) / step;
  }

  return -0.5 * (y_Kinv_y + logdet_K + n_train * log(2 * M_PI));
}

float GPModel::logdet(const DeviceRep& dr_covar) {
  auto n_train = data_.n_train, n_rand = data_.n_rand;

  // x = polynomial(K)^{-1} * rand_z
  // const vf Q_coeffs{3.0 / 27, 1.0, 1.0, 3.0 / 27};
  const vf Q_coeffs{0.25, 1.0, 0.25};
  // const vf Q_coeffs{0.8, 0.4};
  // const vf Q_coeffs{0.5, 0.5};

  DRHornerSmush horner_smush_Q(data_, dr_covar, Q_coeffs);
  mmf x(n_train, n_rand);
  cgsolver_->bcg(horner_smush_Q, x, data_.rand_z);

  // q = poloynomial(K) * x
  // const vf P_coeffs{11.0 / 27, 1.0, -1.0, -11.0 / 27};
  const vf P_coeffs{0.75, 0.0, -0.75};
  // const vf P_coeffs{0.2, 0.8, -1.0};
  // const vf P_coeffs{1.0, -1.0};

  DRHornerSmush horner_smush_P(data_, dr_covar, P_coeffs);
  mmf q(n_train, n_rand);
  horner_smush_P.left_mm(x, q);

  // r = rand_z (batch-dot) q
  mmf r(n_rand);
  cgsolver_->batch_dot(data_.rand_z, q, r);

  // res = r (dot) k_invs
  mmf k_invs(n_rand);
  dmemset(k_invs.data(), 1.0f / n_rand, n_rand);
  float res;
  safe_cublas(cublasSdot, -1, n_rand, r.data(), 1, k_invs.data(), 1, &res);

  return res + precond().logdet();
}

void GPMain::posterior_forward() {
  LOG(INFO) << "Running forward propagation in posterior mode";

  auto n_train = data_.n_train, n_test = data_.n_test, x_dim = data_.x_dim;

  // ************************ cublas version ************************
  // mmf K_s(n_train, n_test), K_ss(n_test, n_test), K(n_test, n_test),
  //     K_tmp(n_train, n_test);

  // // generate covar
  // constexpr int ms = 64, ns = 64;
  // constexpr int block = 256;
  // dim3 grid_Ks((n_test + ns - 1) / ns, (n_train + ms - 1) / ms);
  // auto sm_size = (ms + ns) * x_dim * sizeof(float);
  // generate_covar<ms, ns><<<grid_Ks, block, sm_size>>>(
  //     K_s.data(), data_.train_x.data(), data_.test_x.data(), n_train, n_test,
  //     x_dim, lengthscale(), noise(), output_scale());

  // dim3 grid_Kss((n_test + ns - 1) / ns, (n_test + ms - 1) / ms);
  // generate_covar<ms, ns><<<grid_Kss, block, sm_size>>>(
  //     K_ss.data(), data_.test_x.data(), data_.test_x.data(), n_test, n_test,
  //     x_dim, lengthscale(), noise(), output_scale());

  // const float alpha = 1.0, beta = 0.0, neg_beta = -1.0;

  // // calculate mean
  // // mu = K_s^T * solve_ (solve_ = Ky^-1 * y, saved during training)
  // safe_call_cublas(cublasSgemv, -1, CUBLAS_OP_N, n_test, n_train,
  //                  &alpha, K_s.data(), n_test, solve_.data(), 1, &beta,
  //                  result_.mean.data(), 1);

  // // calculate posterior covar (without sigma * I)
  // // K_tmp = Ky^-1 * K_s
  // cgsolver_.bcg(get_covar(), K_tmp, K_s);
  // // K = K_s^T * K_tmp
  // safe_cublas(cublasSgemm, -1, CUBLAS_OP_N, CUBLAS_OP_T,
  //                  n_test, n_test, n_train, &alpha, K_s.data(), n_test,
  //                  K_tmp.data(), n_test, &beta, K.data(), n_test);

  // *************************** dr version ***************************
  DRCovar K_s(data_, n_train, n_test);
  mmf K_ss(n_test, n_test), K(n_test, n_test), K_tmp(n_train, n_test);

  // generate covar
  constexpr int ms = 64, ns = 64;
  constexpr int block = 256;
  dim3 grid_Ks((n_test + ns - 1) / ns, (n_train + ms - 1) / ms);
  auto sm_size = (ms + ns) * x_dim * sizeof(float);
  generate_covar<ms, ns><<<grid_Ks, block, sm_size>>>(
      K_s.K_data(), data_.train_x.data(), data_.test_x.data(), n_train, n_test,
      x_dim, lengthscale(), noise(), output_scale());

  dim3 grid_Kss((n_test + ns - 1) / ns, (n_test + ms - 1) / ms);
  generate_covar<ms, ns><<<grid_Kss, block, sm_size>>>(
      K_ss.data(), data_.test_x.data(), data_.test_x.data(), n_test, n_test,
      x_dim, lengthscale(), noise(), output_scale());

  // recalculate K_y (TODO: save some computation)
  if (FLAGS_rep) {
    covar_full_ = std::make_unique<DRCovar>(data_, n_train);

    dim3 grid((n_train + ns - 1) / ns, (n_train + ms - 1) / ms);
    generate_covar<ms, ns, CovarType::kAddNoise><<<grid, block, sm_size>>>(
        covar_full_->K_data(), data_.train_x.data(), data_.train_x.data(),
        n_train, n_train, x_dim, lengthscale(), noise(), output_scale());

    cgsolver_->pbcg(*covar_full_, precond(), solve_, data_.train_y);
  }

  // calculate mean
  // mu = K_s^T * solve_ (solve_ = Ky^-1 * y, saved during training)
  K_s.left_mm(solve_, result_.mean, true);

  // calculate posterior covar (without sigma * I)
  // K_tmp = Ky^-1 * K_s
  cgsolver_->pbcg(*covar_full_, precond(), K_tmp, K_s.K());
  // K = K_s^T * K_tmp
  K_s.left_mm(K_tmp, K, true);

  const float alpha = 1.0, neg_beta = -1.0;
  // *******************************************************************
  // K = K_ss - K
  safe_cublas(cublasSgeam, -1, CUBLAS_OP_N, CUBLAS_OP_N, n_test, n_test, &alpha,
              K_ss.data(), n_test, &neg_beta, K.data(), n_test, K.data(),
              n_test);

  // calculate confidence region
  // 1d grid, each block handles one diagonal block of size ns * ns
  // each thread handles one element
  generate_confidence_region<<<(n_test + ns - 1) / ns, ns>>>(
      result_.upper.data(), result_.lower.data(), K.data(), result_.mean.data(),
      n_test, noise());

  if (FLAGS_rep) covar_full_.reset();
}
