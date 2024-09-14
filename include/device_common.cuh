#pragma once

#include "cuda_env.h"

enum class TransferType { kHostToDevice, kDeviceToHost };

// cudaMalloc is a special case that does not work well with safe_call_cuda
// hence writing this specialization explicitly
template <typename T>
void safe_cuda_malloc(T** d_ptr, size_t sz) {
  cudaError_t err = cudaMalloc(d_ptr, sz * sizeof(T));
  if (err != cudaSuccess)
    LOG(FATAL) << "CUDA malloc has failed: " << cudaGetErrorString(err);
}

// recursively call cudaMalloc on any number of pair {device pointer, size}
template <typename T>  // base
void dmalloc(T** d_ptr, size_t sz) {
  safe_cuda_malloc(d_ptr, sz);
}

template <typename T, typename... Args>  // recursive
void dmalloc(T** d_ptr, size_t sz, Args... args) {
  safe_cuda_malloc(d_ptr, sz);
  dmalloc(args...);
}

// recursively call cudaFree on any number of device pointers
template <typename T>  // base
void dfree(T* d_ptr) {
  safe_cuda(cudaFree, d_ptr);
}

template <typename T, typename... Args>  // recursive
void dfree(T* d_ptr, Args... args) {
  safe_cuda(cudaFree, d_ptr);
  dfree(args...);
}

// recursively call cudaMemcpy on any number of group
// {dest pointer, src pointer, size}
// type is the data transfer type, use with compile time constant (enum class)
template <TransferType type>  // base
void dmemcpy() {}

template <TransferType type, typename T, typename... Args>  // recursive
void dmemcpy(T* d_ptr1, const T* d_ptr2, size_t sz, Args... args) {
  if constexpr (type == TransferType::kHostToDevice) {
    safe_cuda(cudaMemcpy, d_ptr1, d_ptr2, sz * sizeof(T),
              cudaMemcpyHostToDevice);
  } else if constexpr (type == TransferType::kDeviceToHost) {
    safe_cuda(cudaMemcpy, d_ptr1, d_ptr2, sz * sizeof(T),
              cudaMemcpyDeviceToHost);
  }

  dmemcpy<type>(args...);
}

template <typename T>
void dmemset_impl(T* d_ptr, T value, size_t sz) {
  static_assert(std::is_arithmetic<T>::value,
                "dmemset only works with arithmetic data types");

  if (value == T(0)) {
    safe_cuda(cudaMemset, d_ptr, value, sz * sizeof(T));
  } else {
    std::vector<T> tmp_values(sz, value);
    dmemcpy<TransferType::kHostToDevice>(d_ptr, tmp_values.data(), sz);
  }
}

// recursively call cudaMemset on any number of group
// {dest pointer, value, size}
// the underlying cudaMemset may or may not be applicable, if not, we allocate
// temporary vector and do a dmemcpy
template <typename T>  // base
void dmemset(T* d_ptr, T value, size_t sz) {
  dmemset_impl(d_ptr, value, sz);
}

template <typename T, typename... Args>  // recursive
void dmemset(T* d_ptr, T value, size_t sz, Args... args) {
  dmemset_impl(d_ptr, value, sz);
  dmemset(args...);
}
