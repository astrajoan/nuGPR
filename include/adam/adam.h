#pragma once

#include <random>

#include "gpmodel/model.h"

/**
 * @brief Adam optimizer which uses numerical gradient for training Gaussian
 * process model
 *
 * can be used in a model agnostic way
 */

class NumericalAdam {
 public:
  NumericalAdam(GPMain& model, float beta1, float beta2, float alpha)
      : gen_(rd_()),
        model_(model),
        beta1_(beta1),
        beta2_(beta2),
        alpha_(alpha) {
    // store model parameter references in params_ref_
    // default initialize grads_, steps_, m_, v_ values for each parameter
    model_.register_params(params_ref_);
    for (const auto& [name, _] : params_ref_) {
      grads_[name] = m_[name] = v_[name] = 0.0;
      steps_[name] = 0.5;
    }
  }

  explicit NumericalAdam(GPMain& model)
      : NumericalAdam(model, 0.9, 0.999, 1e-3) {}

  ~NumericalAdam() = default;

  void calc_gradients(vf* lptr = nullptr, float grad_eps = kMinGradEps,
                      float step_eps = kMinStepEps);

  void step();

 private:
  // calculate gradient for one parameter and update numerical gradient step
  void gradient(msrf& tmp_refs, const string& name, float base, float grad_eps,
                float step_eps, GPTemporary* m);

  void record_hist(const string& name, size_t cnt) {
    if (hist_[name].size() >= kMaxHistItems) hist_[name].pop_front();
    hist_[name].push_back(cnt);
  }

  bool skip_grad_hist(const string& name) {
    float prob = 0.0;
    for (auto x : hist_[name])
      if (x == 2) prob = 1.0 - 1.0 / kMaxHistItems;
    std::bernoulli_distribution dist(prob);
    return dist(gen_);
  }

  // gradient tolerance between two step sizes for numerical gradient
  static constexpr float kMinGradEps = 1e-3;
  // minimum step size for calculating numerical gradient
  static constexpr float kMinStepEps = 1e-6;
  // maximum number of histogram items per parameter
  static constexpr size_t kMaxHistItems = 10;

  GPMain& model_;

  // for random generation
  std::random_device rd_;
  std::mt19937 gen_;

  // decay rates and learing rate
  float beta1_ = 0.0, beta2_ = 0.0, alpha_ = 0.0;
  // beta1^t and beta2^t for Adam (t = time step)
  float beta1_tp_ = 1.0, beta2_tp_ = 1.0;
  // first and second momentum
  msf m_, v_;
  // whether main model has decomposed needs to be recorded
  bool main_decompose_;

  // references to parameters of model
  msrf params_ref_;
  // numerical gradient for each parameter
  msf grads_;
  // step size to calculate numerical gradient from previous epoch
  msf steps_;
  // histogram for previous gradient steps
  msls hist_;
};
