#include "adam/adam.h"

#include <gtest/gtest.h>

#include <random>

#include "device_test_helper.h"
#include "test_helper.h"

namespace {

// ground truth of parameters
static constexpr float gt_lengthscale = 0.75;
static constexpr float gt_noise = 0.5;
static constexpr float gt_output_scale = 0.25;

float quad(const vf& mock_x, const vf& mock_y, float lengthscale, float noise,
           float output_scale) {
  // calculate total loss
  auto n_mock = mock_x.size();
  float loss = 0.0;
  for (int i = 0; i < n_mock; ++i) {
    auto x = mock_x[i], y = mock_y[i];
    auto pred_y = lengthscale * x * x + noise * x + output_scale;
    loss += (y - pred_y) * (y - pred_y);
  }

  // divide by number of training data to get MSE loss
  return loss / n_mock;
}

vf generate_quadratic_with_noise(const vf& mock_x, float sigma) {
  auto n_mock = mock_x.size();
  vf mock_y(n_mock);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dis(0.0, sigma);

  for (int i = 0; i < n_mock; ++i) {
    float x = mock_x[i];
    float gt_y = gt_lengthscale * x * x + gt_noise * x + gt_output_scale;
    mock_y[i] = gt_y + dis(gen);
  }

  return mock_y;
}

class QuadraticTemporary : public GPTemporary {
 public:
  QuadraticTemporary(const GPData& data, SuperCholeskyCPUSolver& chol_cpu,
                     BatchPreconditioner& precond, float rls, float rnoise,
                     float ros, const vf& mock_x, const vf& mock_y)
      : GPTemporary(data, chol_cpu, precond, rls, rnoise, ros),
        mock_x_(mock_x),
        mock_y_(mock_y) {}

  float train_forward() override {
    // float train_forward(TrainOpts opts = {}) override {
    return quad(mock_x_, mock_y_, lengthscale(), noise(), output_scale());
  }

 private:
  const vf &mock_x_, mock_y_;
};

class QuadraticMain : public GPMain {
 public:
  QuadraticMain(const GPData& data, GPResult& result, const vf& mock_x,
                const vf& mock_y)
      : GPMain(data, result), mock_x_(mock_x), mock_y_(mock_y) {}

  float train_forward() override {
    // float train_forward(TrainOpts opts = {}) override {
    return quad(mock_x_, mock_y_, lengthscale(), noise(), output_scale());
  }

  std::unique_ptr<GPTemporary> shallow_copy() const override {
    float rls = lengthscale_, rnoise = noise_, ros = output_scale_;
    return std::make_unique<QuadraticTemporary>(
        data_, chol_cpu(), precond(), rls, rnoise, ros, mock_x_, mock_y_);
  }

 private:
  const vf &mock_x_, mock_y_;
};

}  // namespace

TEST(AdamTests, quadraticOptimizationTest) {
  // Create an artificial dataset with 1000 random training samples
  vf mock_x = generate_random_vector(1000, 1);
  vf mock_y = generate_quadratic_with_noise(mock_x, 0.1);

  GPData data;
  init_mock_data(data, mock_x.size());
  GPResult result(data);

  // Create model, optimizer, and register parameters
  QuadraticMain model(data, result, mock_x, mock_y);
  NumericalAdam optim(model, 0.9, 0.999, 0.05);
  model.set_params(0.1, 0.2, 0.7);

  // Train for 50 epochs and output the loss each time
  float prev_loss = model.train_forward();
  for (int epoch = 0; epoch < 100; ++epoch) {
    optim.calc_gradients();
    optim.step();
    float loss = model.train_forward();
    prev_loss = loss;
    LOG(INFO) << "Epoch " << epoch << ", loss = " << loss;
  }

  LOG(INFO) << "Final parameters: lengthscale = " << model.lengthscale()
            << ", noise = " << model.noise()
            << ", output_scale = " << model.output_scale();

  // Check that the final parameters are close enough to the ground truth
  EXPECT_NEAR(model.lengthscale(), gt_lengthscale, 0.1 * gt_lengthscale);
  EXPECT_NEAR(model.noise(), gt_noise, 0.1 * gt_noise);
  EXPECT_NEAR(model.output_scale(), gt_output_scale, 0.1 * gt_output_scale);
}
