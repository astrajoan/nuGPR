#include <functional>
#include <numeric>

#include "device_common.cuh"
#include "gpdata/gp.h"

void copy_cnpy_to_gpu(const cnpy::NpyArray& arr, mmf& data) {
  dmemcpy<TransferType::kHostToDevice>(data.data(), arr.data<float>(),
                                       data.size());
}

void save_cnpy_cpu(const fsp& path, const vf& data, const vsz& shape) {
  try {
    cnpy::npy_save(path.string(), data.data(), shape, "w");
  } catch (...) {
    LOG(FATAL) << "Error saving npy file: " << path;
  }
}

void save_cnpy_gpu(const fsp& path, mmf& data, const vsz& shape) {
  auto sz =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
  vf h_data(sz);
  dmemcpy<TransferType::kDeviceToHost>(h_data.data(), data.data(), sz);
  save_cnpy_cpu(path, h_data, shape);
}

void GPData::load_data(const string& config_path, const json& config) {
  LOG(INFO) << "Loading training and testing data for GP";

  json params;
  CHECK(get_field(config, "params", params));
  CHECK(get_field(params, "n_blocks", n_blocks));

  json paths;
  CHECK(get_field(config, "input_paths", paths));

  string train_x_path, train_y_path, test_x_path, test_y_path, rep_x_path,
      rand_z_path;
  CHECK(get_field(paths, "train_x", train_x_path, "train_y", train_y_path,
                  "test_x", test_x_path, "test_y", test_y_path, "rep_x",
                  rep_x_path, "rand_z", rand_z_path));

  auto base_dir = std::filesystem::path(config_path).parent_path();

  load_cnpy_gpu(base_dir / train_x_path, train_x, n_train, x_dim);
  load_cnpy_gpu(base_dir / train_y_path, train_y, n_train);
  load_cnpy_gpu(base_dir / test_x_path, test_x, n_test, x_dim);
  load_cnpy_gpu(base_dir / test_y_path, test_y, n_test);
  load_cnpy_gpu(base_dir / rep_x_path, rep_x, n_blocks, x_dim);
  load_cnpy_gpu(base_dir / rand_z_path, rand_z, n_train, n_rand);

  ones.resize(std::max(std::max(n_train, n_test), n_rand));
  dmemset(ones.data(), 1.0f, ones.size());
}

void GPData::set_ones(size_t n) {
  ones.resize(n);
  dmemset(ones.data(), 1.0f, n);
}

void GPResult::save_data(const string& config_path, const json& config) {
  LOG(INFO) << "Saving GP predictions and contour results";

  json paths;
  CHECK(get_field(config, "output_paths", paths));

  string mean_path, covar_path, upper_path, lower_path, contour_path,
      losses_path;
  CHECK(get_field(paths, "mean", mean_path, "covar", covar_path, "upper",
                  upper_path, "lower", lower_path, "contour", contour_path,
                  "losses", losses_path));

  auto base_dir = std::filesystem::path(config_path).parent_path();

  if (contour.size())
    save_cnpy_cpu(base_dir / contour_path, contour, {contour.size()});
  if (losses.size())
    save_cnpy_cpu(base_dir / losses_path, losses, {losses.size()});

  save_cnpy_gpu(base_dir / mean_path, mean, {n_test});
  // save_cnpy_gpu(base_dir / covar_path, covar, {n_train, n_train});
  save_cnpy_gpu(base_dir / upper_path, upper, {n_test});
  save_cnpy_gpu(base_dir / lower_path, lower, {n_test});
}
