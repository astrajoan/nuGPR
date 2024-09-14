#include <chrono>

#include "adam/adam.h"
#include "cuda_env.h"

DECLARE_bool(optimize_memory);
DEFINE_string(config, "../data/config.json", "Path to JSON config file");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  // Ensure log outputs are written to the console for more visible debugging
  FLAGS_logtostdout = true;
  google::InitGoogleLogging(argv[0]);

  // Read JSON config file
  const auto& config = read_json_config(FLAGS_config);

  // Read the total number of data points to estimate CUDA memory pool size
  json params;
  CHECK(get_field(config, "params", params));
  size_t n_total;
  CHECK(get_field(params, "n_total", n_total));

  // RMM memory pool initialization
  auto est_sz = n_total * n_total * sizeof(float);
  auto init_scalar = FLAGS_optimize_memory ? 0.2 : 1.0;
  auto max_scalar = FLAGS_optimize_memory ? 1.0 : 4.0;
  RMMResourceWrapper rmm(est_sz, init_scalar, max_scalar);

  // GP data (input) and result (output) initialization
  GPData data(FLAGS_config, config);
  GPResult result(data);

  // cuBLAS and cuSOLVER libraries and other CUDA structures initialization
  initialize_cuda_env(data.n_blocks);

  // Populate other user-defined parameters
  int epoches, contour_steps;
  float lr;
  CHECK(get_field(params, "epoches", epoches, "learning_rate", lr,
                  "contour_steps", contour_steps));

  json ls_params, ns_params, os_params;
  CHECK(get_field(params, "lengthscale", ls_params, "noise", ns_params,
                  "output_scale", os_params));

  float min_ls, max_ls, init_ls, min_ns, max_ns, init_ns, init_os;
  CHECK(get_field(ls_params, "min_value", min_ls, "max_value", max_ls,
                  "initial_value", init_ls));
  CHECK(get_field(ns_params, "min_value", min_ns, "max_value", max_ns,
                  "initial_value", init_ns));
  CHECK(get_field(os_params, "initial_value", init_os));

  // Model and ADAM optimizer initialization
  GPMain model(data, result);
  NumericalAdam optim(model, 0.9, 0.999, lr);

  // Contour plot generation
  auto ls_step = (max_ls - min_ls) / (contour_steps - 1);
  auto ns_step = (max_ns - min_ns) / (contour_steps - 1);
  for (int i = 0; i < contour_steps; ++i) {
    for (int j = 0; j < contour_steps; ++j) {
      auto ls = min_ls + i * ls_step;
      auto ns = min_ns + j * ns_step;
      LOG(INFO) << "Contour: lengthscale = " << ls << ", noise = " << ns;

      model.set_params(ls, ns, init_os);
      result.losses.push_back(model.train_forward());
    }
  }

  // Main training loop, starting from pre-defined initial hyperparameters
  model.set_params(init_ls, init_ns, init_os);
  auto start_ts = std::chrono::high_resolution_clock::now();

  for (int epoch = 0; epoch < epoches; ++epoch) {
    optim.calc_gradients(&result.contour);
    optim.step();
    LOG(INFO) << "Epoch " << epoch << ": lengthscale = " << model.lengthscale()
              << ", noise = " << model.noise()
              << ", output_scale = " << model.output_scale();

    result.contour.push_back(model.lengthscale());
    result.contour.push_back(model.noise());

    // maintain the order of contour as (lengthscale, noise, loss)
    auto sz = result.contour.size();
    std::swap(result.contour[sz - 3], result.contour[sz - 2]);
    std::swap(result.contour[sz - 2], result.contour[sz - 1]);
  }

  auto dur_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - start_ts);
  LOG(INFO) << "Total training time: " << dur_ms.count() / 1e3 << " seconds";

  if (!FLAGS_optimize_memory) {
    // Posterior mode (prediction) after training
    model.posterior_forward();

    // Save output data as .npy files to disk
    result.save_data(FLAGS_config, config);
  }

  // Clean up; RMM wrapper uses RAII, so we only need to to reset CUDA env
  finalize_cuda_env();

  return 0;
}
