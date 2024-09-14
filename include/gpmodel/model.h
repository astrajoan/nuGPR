#pragma once

#include <cmath>
#include <functional>
#include <list>
#include <map>

#include "gpdata/gp.h"
#include "gpmodel/solver.h"
#include "gpmodel/super_cholesky.h"

DECLARE_bool(rep);
DECLARE_bool(super);

using msf = std::map<std::string, float>;
using msrf = std::map<std::string, std::reference_wrapper<float>>;
using msls = std::map<std::string, std::list<size_t>>;

inline float softplus(float x, float lower_bound = 0.0) {
  return std::log(1 + std::exp(x)) + lower_bound;
}

inline float inv_softplus(float x, float lower_bound = 0.0) {
  return std::log(-std::expm1(lower_bound - x)) + x;
}

/*
 * holds model parameters, mean, covar and confidence region
 *
 * owns:
 *   - device data -> float* to device memory: mean, covar, confidence region
 *   - constant reference access to GPData
 *
 * types:
 *   - main     : model for training and updating parameters
 *   - temporary: one-time model for calculating gradient
 */
class GPModel {
 public:
  explicit GPModel(const GPData& data);

  virtual ~GPModel() = default;

  // parameters
  float lengthscale() const { return softplus(lengthscale_); }
  float noise() const { return softplus(noise_, kNoiseLowerBound); }
  float output_scale() const { return softplus(output_scale_); }

  void set_params(float lengthscale, float noise, float output_scale) {
    CHECK_GT(lengthscale, 0.0);
    CHECK_GT(noise, kNoiseLowerBound);
    CHECK_GT(output_scale, 0.0);

    lengthscale_ = inv_softplus(lengthscale);
    noise_ = inv_softplus(noise, kNoiseLowerBound);
    output_scale_ = inv_softplus(output_scale);
  }

  // raw parameters
  void register_params(msrf& params_ref) {
    params_ref.emplace("lengthscale", std::ref(lengthscale_));
    params_ref.emplace("noise", std::ref(noise_));
    params_ref.emplace("output_scale", std::ref(output_scale_));
  }

  virtual float train_forward();
  virtual SuperNoiseOutputscaleBinder* binder() { return nullptr; }

  const DeviceRep& get_covar() const {
    if (FLAGS_rep) return *covar_;
    return *covar_full_;
  }

 protected:
  virtual SuperCholeskyCPUSolver& chol_cpu() const = 0;
  virtual BatchPreconditioner& precond() const = 0;

  float log_prob();
  float log_prob_combined();
  float logdet(const DeviceRep& dr_covar);

  // necessary parameters and model owned data structures
  const GPData& data_;
  float lengthscale_, noise_, output_scale_;
  mmf solve_;
  std::unique_ptr<DRCovarODLR> covar_;
  std::unique_ptr<DRCovar> covar_full_;
  std::unique_ptr<BatchCGSolver> cgsolver_;

  // metadata options to control and diverge the train forward process
  bool decompose_;
  std::string param_;
  float gamma_, delta_, p0_, p1_;

 private:
  static constexpr float kNoiseLowerBound = 1e-4;

  GPModel(const GPModel&) = delete;
  GPModel& operator=(const GPModel&) = delete;
  GPModel(GPModel&&) = delete;
  GPModel& operator=(GPModel&&) = delete;
};

class GPTemporary : public GPModel {
 public:
  GPTemporary(const GPData& data, SuperCholeskyCPUSolver& chol_cpu,
              BatchPreconditioner& precond, float raw_lengthscale,
              float raw_noise, float raw_output_scale)
      : GPModel(data), chol_cpu_(chol_cpu), precond_(precond) {
    lengthscale_ = raw_lengthscale;
    noise_ = raw_noise;
    output_scale_ = raw_output_scale;
  }

  SuperNoiseOutputscaleBinder* binder() override { return &binder_; }

  void set_train_opts(bool main_decompose, std::string param, float step) {
    decompose_ = false;
    param_ = std::move(param);
    if (FLAGS_super && param_ == "noise") {
      p1_ = noise();
      noise_ -= step, p0_ = noise(), noise_ += step;
      if (main_decompose)
        gamma_ = 1.0, delta_ = p1_ - p0_;
      else
        gamma_ = delta_ = 0.0;
    } else if (FLAGS_super && main_decompose && param_ == "output_scale") {
      p1_ = output_scale();
      output_scale_ -= step, p0_ = output_scale(), output_scale_ += step;
      gamma_ = p1_ / p0_, delta_ = (1.0 - gamma_) * noise();
    } else {
      gamma_ = delta_ = 0.0;
    }
  }

 protected:
  SuperCholeskyCPUSolver& chol_cpu() const override { return chol_cpu_; }
  BatchPreconditioner& precond() const override { return precond_; }

 private:
  SuperCholeskyCPUSolver& chol_cpu_;
  BatchPreconditioner& precond_;
  SuperNoiseOutputscaleBinder binder_;
};

class GPMain : public GPModel {
 public:
  GPMain(const GPData& data, GPResult& result)
      : GPModel(data), result_(result) {
    chol_cpu_ = std::make_unique<SuperCholeskyCPUSolver>(data.n_blocks);
    precond_ = std::make_unique<SuperCholeskyPreconditioner>(
        data, covar_->K_diag(), data.n_train, data.n_blocks, chol_cpu());
  }

  // calculates mean and confidence region for test data
  void posterior_forward();

  void set_train_opts(bool main_decompose) {
    decompose_ = main_decompose;
    param_.clear();
    if (FLAGS_super && main_decompose)
      gamma_ = 1.0, delta_ = 0.0;
    else
      gamma_ = delta_ = 0.0;
  }

  virtual std::unique_ptr<GPTemporary> shallow_copy() const {
    return std::make_unique<GPTemporary>(  // !
        data_, chol_cpu(), precond(), lengthscale_, noise_, output_scale_);
  }

 protected:
  SuperCholeskyCPUSolver& chol_cpu() const override { return *chol_cpu_; }
  BatchPreconditioner& precond() const override { return *precond_; }

 private:
  std::unique_ptr<SuperCholeskyCPUSolver> chol_cpu_;
  std::unique_ptr<BatchPreconditioner> precond_;
  // for getting mean, covar and confidence region during posterior forward
  GPResult& result_;
};
