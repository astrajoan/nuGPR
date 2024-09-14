#include <glog/logging.h>

#include "adam/adam.h"

DEFINE_bool(resetstep, false, "Reset step in numerical gradient each epoch");
DEFINE_bool(bind_grad, false, "Bind noise and output_scale gradients");
DEFINE_bool(skip_grad, false, "Skip gradient validation based on histogram");
DEFINE_int32(decompose_iters, 1, "Decompose every x iterations");

/**
 * @brief calculate numerical gradient for a single parameter
 *
 * @param tmp_refs map to reference of parameters to temporary models
 * @param name name of the parameter to calculate gradient
 * @param base current value of the parameter in the model
 * @param grad_eps stopping criteria for gradient calculation
 * @param step_eps minimum step size for gradient calculation
 * @param m temporary model for changing to calculate gradient
 */
void NumericalAdam::gradient(msrf& tmp_refs, const string& name, float base,
                             float grad_eps, float step_eps, GPTemporary* m) {
  auto step = steps_[name] * 2;
  float grad1 = std::numeric_limits<float>::infinity(), grad2 = 0.0, e1;
  auto binder = m->binder();
  binder->grad1 = grad1, binder->grad2 = grad2;

  size_t cnt = 0;
  while (step > step_eps * 2 && std::fabs(grad1 - grad2) > grad_eps) {
    grad2 = grad1;
    binder->grad2 = binder->grad1;
    step *= 0.5;

    tmp_refs.at(name).get() = base + step;
    m->set_train_opts(main_decompose_, name, step);
    e1 = m->train_forward();
    grad1 = (e1 - binder->e0) / step;

    if (++cnt == 1 && FLAGS_skip_grad && skip_grad_hist(name)) {
      LOG(INFO) << "Skipping gradient validation for parameter " << name;
      grad2 = grad1;
      binder->grad2 = binder->grad1;
      step *= 0.5;
    }
  }

  tmp_refs.at(name).get() = base;
  record_hist(name, cnt);
  grads_[name] = (grad1 + grad2) * 0.5;
  if (!FLAGS_resetstep)
    steps_[name] = step * 2;  // compensate for step / 2 in last iteration
}

/**
 * @brief calculate numerical gradients for all parameters stored in params_ref_
 *
 * @param grad_eps stopping criteria for gradient calculation
 * @param step_eps minimum step size for gradient calculation
 */
void NumericalAdam::calc_gradients(vf* lptr, float grad_eps, float step_eps) {
  LOG(INFO) << "Calculating numerical gradients for lengthscale, noise, and "
               "output_scale";

  // temporary model for calculating numerical gradient
  auto m = model_.shallow_copy();
  auto binder = m->binder();
  msrf tmp_refs;
  m->register_params(tmp_refs);

  // base case
  main_decompose_ = !lptr || (lptr->size() / 3) % FLAGS_decompose_iters == 0;
  model_.set_train_opts(main_decompose_);
  binder->e0 = model_.train_forward();
  if (lptr) lptr->push_back(binder->e0);

  for (const auto& [name, param] : params_ref_)
    if (!FLAGS_bind_grad || name != "output_scale")
      gradient(tmp_refs, name, param.get(), grad_eps, step_eps, m.get());

  if (FLAGS_bind_grad)
    grads_["output_scale"] = (binder->grad1 + binder->grad2) * 0.5;
}

/**
 * @brief update parameters stored in params_ref_
 */
void NumericalAdam::step() {
  LOG(INFO) << "Updating parameters for lengthscale, noise, and output_scale";

  beta1_tp_ *= beta1_;
  beta2_tp_ *= beta2_;

  for (const auto& [name, param] : params_ref_) {
    m_[name] = beta1_ * m_[name] + (1 - beta1_) * grads_[name];
    v_[name] = beta2_ * v_[name] + (1 - beta2_) * grads_[name] * grads_[name];

    float m_hat = m_[name] / (1 - beta1_tp_);
    float v_hat = v_[name] / (1 - beta2_tp_);

    param.get() -= alpha_ * m_hat / (std::sqrt(v_hat) + 1e-8);
  }
}
