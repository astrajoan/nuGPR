#pragma once

#include <cnpy.h>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

#include "cuda_env.h"

using fsp = std::filesystem::path;
using json = nlohmann::json;
using std::string;
using vf = std::vector<float>;
using vsz = std::vector<size_t>;

inline bool get_field(const json& data) { return true; }  // base case

template <typename T, typename... Args>  // recursive
bool get_field(const json& data, const string& key, T& val, Args&... args) {
  try {
    val = data[key];
    return get_field(data, args...);
  } catch (...) {
    LOG(ERROR) << "Error reading key: " << key << " in json object";
  }
  return false;
}

inline json read_json_config(const string& path) {
  LOG(INFO) << "Loading GP data and parameters from JSON file: " << path;

  std::ifstream json_file(path);
  try {
    return json::parse(json_file);
  } catch (...) {
    LOG(FATAL) << "Failed to parse JSON file: " << path;
  }
  return {};  // we will never reach here
}

template <size_t... Idxs, typename... Args>
void assign(const vsz& shape, std::index_sequence<Idxs...>, Args&... dims) {
  (..., (dims = shape[Idxs]));
}

// necessary as a non-template function since CUDA code is involved
void copy_cnpy_to_gpu(const cnpy::NpyArray& arr, mmf& data);

template <typename... Args>
void load_cnpy_gpu(const fsp& path, mmf& data, Args&... dims) {
  try {
    auto arr = cnpy::npy_load(path.string());
    if (sizeof...(dims) != arr.shape.size()) {
      LOG(ERROR) << "dims size = " << sizeof...(dims)
                 << ", arr.shape.size() = " << arr.shape.size();
      LOG(FATAL) << "Dimension mismatch on npy array assignment: " << path;
    }
    assign(arr.shape, std::make_index_sequence<sizeof...(dims)>{}, dims...);

    data.resize(dims...);
    copy_cnpy_to_gpu(arr, data);
  } catch (...) {
    LOG(FATAL) << "Error loading npy file: " << path;
  }
}

void save_cnpy_cpu(const fsp& path, const vf& data, const vsz& shape);

void save_cnpy_gpu(const fsp& path, const mmf& data, const vsz& shape);

/*
 * holds train / test data (x-y) and random vector for logdet approximate
 *
 * owns:
 *   - host data   -> vector<float> of train/test x and y, rand_z (unused)
 *   - device data -> ManagedMemory<float> of train/test x and y, rand_z
 */
struct GPData {
  // host - currently empty, we directly load data onto GPU
  // vf train_x, train_y, test_x, test_y, rand_z;

  // number of training data, test data, random vectors
  size_t n_train = 0, n_test = 0, n_rand = 0;
  // input data dimension
  size_t x_dim = 1;
  // number of blocks in the input data
  size_t n_blocks = 1;

  // device
  mmf train_x, train_y, test_x, test_y, rep_x, rand_z, ones;

  GPData() = default;
  GPData(const string& config_path, const json& config) {
    load_data(config_path, config);
  }

  // rule of five defaults; GPData struct should be copyable and movable
  ~GPData() = default;
  GPData(const GPData&) = default;
  GPData(GPData&&) = default;
  GPData& operator=(const GPData&) = default;
  GPData& operator=(GPData&&) = default;

  // load dataset into device memory
  void load_data(const string& config_path, const json& config);

  void set_ones(size_t n);
};

/*
 * holds results on both host and device, can be saved for external plotting
 */
struct GPResult {
  // number of training data and test data
  size_t n_train = 0, n_test = 0;

  // losses and contour are always on host
  vf losses, contour;

  // device data
  mmf mean, covar, upper, lower;

  GPResult() = default;

  explicit GPResult(const GPData& data)
      : n_train(data.n_train),
        n_test(data.n_test),
        mean(data.n_test),
        // covar(data.n_train, data.n_train),
        upper(data.n_test),
        lower(data.n_test) {}

  // rule of five defaults; GPResult struct should be copyable and movable
  ~GPResult() = default;
  GPResult(const GPResult&) = default;
  GPResult(GPResult&&) = default;
  GPResult& operator=(const GPResult&) = default;
  GPResult& operator=(GPResult&&) = default;

  // save both host and device results to output cnpy files
  void save_data(const string& config_path, const json& config);
};
