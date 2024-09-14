#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "rmm.h"

DECLARE_bool(enable_cuda_streams);

/**
 * @brief the entire program should have only one copy of CUDA streams
 *
 * @return vector of CUDA streams
 */
inline std::vector<cudaStream_t>& streams() {
  static std::vector<cudaStream_t> vec;
  return vec;
}

/**
 * @brief the entire program should have only one copy of cuBLAS handles
 *
 * @return vector of cuBLAS handles
 */
inline std::vector<cublasHandle_t>& cublas_handles() {
  static std::vector<cublasHandle_t> vec;
  return vec;
}

/**
 * @brief the entire program should have only one copy of cuSOLVER handles
 *
 * @return vector of cuSOLVER handles
 */
inline std::vector<cusolverDnHandle_t>& cusolver_handles() {
  static std::vector<cusolverDnHandle_t> vec;
  return vec;
}

/**
 * @brief calls CUDA builtin functions with safety checks
 *
 * @tparam Func CUDA builtin function
 * @tparam Args argument pack to the CUDA builtin function
 * @param func the CUDA builtin function being called
 * @param args arguments to func
 */
template <typename Func, typename... Args>
void safe_cuda(Func func, Args... args) {
  cudaError_t err = func(args...);
  if (err != cudaSuccess)
    LOG(FATAL) << "Encountered CUDA error: " << cudaGetErrorString(err);
}

/**
 * @brief calls cuBLAS functions with safety checks
 *
 * @tparam Func cuBLAS function
 * @tparam Args argument pack to the cuBLAS function
 * @param func the cuBLAS function being called
 * @param sid stream id of the current call, -1 if using default stream
 * @param args arguments to func
 */
template <typename Func, typename... Args>
void safe_cublas(Func func, int sid, Args... args) {
  cublasStatus_t err = func(cublas_handles()[sid + 1], args...);
  if (err != CUBLAS_STATUS_SUCCESS)
    LOG(FATAL) << "Encountered cuBLAS error with status code: " << err;
}

/**
 * @brief calls cuSOLVER functions with safety checks
 *
 * @tparam Func cuSOLVER function
 * @tparam Args argument pack to the cuSOLVER function
 * @param func the cuSOLVER function being called
 * @param sid stream id of the current call, -1 if using default stream
 * @param args arguments to func
 */
template <typename Func, typename... Args>
void safe_cusolver(Func func, int sid, Args... args) {
  cusolverStatus_t err = func(cusolver_handles()[sid + 1], args...);
  if (err != CUSOLVER_STATUS_SUCCESS)
    LOG(FATAL) << "Encountered cuSOLVER error with status code: " << err;
}

/**
 * @brief creates CUDA structs, called at the beginning of entire program
 *
 * @param n_streams number of CUDA streams, same as the number of handles for
 *                  cuBLAS and cuSOLVER
 */
inline void initialize_cuda_env(int n_streams) {
  streams().clear();
  stream_views().clear();
  cublas_handles().clear();
  cusolver_handles().clear();

  for (int i = 0; i < n_streams + 1; ++i) {
    streams().emplace_back();
    // The first position is reserved for the default stream (aka. stream 0)
    if (i == 0 || !FLAGS_enable_cuda_streams)
      streams().back() = 0;
    else
      safe_cuda(cudaStreamCreate, &streams().back());

    stream_views().emplace_back(streams().back());

    CHECK(cublasCreate(&cublas_handles().emplace_back()) ==
          CUBLAS_STATUS_SUCCESS);
    safe_cublas(cublasSetStream, i - 1, streams().back());

    CHECK(cusolverDnCreate(&cusolver_handles().emplace_back()) ==
          CUSOLVER_STATUS_SUCCESS);
    safe_cusolver(cusolverDnSetStream, i - 1, streams().back());
  }
}

/**
 * @brief destroys CUDA structs, called at the end of entire program
 */
inline void finalize_cuda_env() {
  for (int i = 0; i < streams().size(); ++i) {
    // Only the streams called with cudaStreamCreate need to be destroyed
    if (i != 0 && FLAGS_enable_cuda_streams)
      safe_cuda(cudaStreamDestroy, streams()[i]);
    safe_cublas(cublasDestroy, i - 1);
    safe_cusolver(cusolverDnDestroy, i - 1);
  }
}
