#pragma once

#ifdef __INTELLISENSE__

namespace rmm::mr {

template <typename Upstream>
class pool_memory_resource {
 public:
  pool_memory_resource(Upstream* upstream, size_t init_sz, size_t max_sz) {}
};

template <typename Upstream>
void set_current_device_resource(pool_memory_resource<Upstream>* mr) {}

}  // namespace rmm::mr

#else  // __INTELLISENSE__

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#endif  // __INTELLISENSE__

#include <memory>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <utility>
#include <vector>

/**
 * @brief The entire program should have only one copy of CUDA stream views
 *
 * @return vector of CUDA stream views
 */
inline std::vector<rmm::cuda_stream_view>& stream_views() {
  static std::vector<rmm::cuda_stream_view> vec;
  return vec;
}

/**
 * @brief simple wrapper around rmm::device_buffer to manage temporary memory
 */
template <typename T>
class ManagedMemory {
 public:
  ManagedMemory() = default;

  ManagedMemory(size_t m, size_t n)
      : buf_(m * n * sizeof(T), rmm::cuda_stream_default), dim_(m, n) {}

  ManagedMemory(size_t m, size_t n, int sid)
      : buf_(m * n * sizeof(T), stream_views()[sid + 1]), dim_(m, n) {}

  explicit ManagedMemory(size_t n) : ManagedMemory(n, 1) {}

  ~ManagedMemory() = default;
  ManagedMemory(ManagedMemory&& other) = default;
  ManagedMemory& operator=(ManagedMemory&& other) = default;

  ManagedMemory(const ManagedMemory& other)
      : buf_(other.buf_, other.buf_.stream()), dim_(other.dim_) {}

  ManagedMemory& operator=(const ManagedMemory& other) {
    if (this != &other) {
      buf_ = rmm::device_buffer(other.buf_, other.buf_.stream());
      dim_ = other.dim_;
    }
    return *this;
  }

  T* data() { return static_cast<T*>(buf_.data()); }
  const T* data() const { return static_cast<const T*>(buf_.data()); }

  size_t m() const { return dim_.first; }
  size_t n() const { return dim_.second; }
  size_t size() const { return m() * n(); }

  void reshape(size_t m, size_t n) {
    CHECK(m * n == size()) << "Cannot reshape to a different size";
    dim_.first = m, dim_.second = n;
  }
  void reshape(size_t n) { reshape(n, 1); }

  void resize(size_t m, size_t n) {
    buf_.resize(m * n * sizeof(T), buf_.stream());
    dim_.first = m, dim_.second = n;
  }
  void resize(size_t n) { resize(n, 1); }

 private:
  rmm::device_buffer buf_;
  std::pair<size_t, size_t> dim_;
};

using mmf = ManagedMemory<float>;  // convenient alias for float memories

/**
 * @brief RAII-based exception-free wrapper around RMM resources
 *
 * @details Only one copy of this class is supposed to exist in the program
 */
class RMMResourceWrapper {
 public:
  explicit RMMResourceWrapper(size_t est_sz)
      : RMMResourceWrapper(est_sz, kInitScalar, kMaxScalar) {}

  RMMResourceWrapper(size_t est_sz, float init_scalar, float max_scalar) {
    try {
      upstream_ = std::make_unique<cuda_resource_t>();
      pool_ = std::make_unique<pool_resource_t>(
          upstream_.get(), init_scalar * est_sz, max_scalar * est_sz);
      rmm::mr::set_current_device_resource(pool_.get());
    } catch (...) {
      LOG(FATAL) << "Failed to initialize CUDA RMM resources";
    }
  }

  ~RMMResourceWrapper() {
    try {
      rmm::mr::set_current_device_resource(nullptr);
      pool_.reset();
      upstream_.reset();
    } catch (...) {
      LOG(FATAL) << "Failed to destroy CUDA RMM resources";
    }
  }

 private:
  using cuda_resource_t = rmm::mr::cuda_memory_resource;
  using pool_resource_t = rmm::mr::pool_memory_resource<cuda_resource_t>;

  RMMResourceWrapper() = delete;
  RMMResourceWrapper(const RMMResourceWrapper&) = delete;
  RMMResourceWrapper& operator=(const RMMResourceWrapper&) = delete;
  RMMResourceWrapper(RMMResourceWrapper&&) = delete;
  RMMResourceWrapper& operator=(RMMResourceWrapper&&) = delete;

  static constexpr float kInitScalar = 1.0;
  static constexpr float kMaxScalar = 4.0;

  std::unique_ptr<cuda_resource_t> upstream_;
  std::unique_ptr<pool_resource_t> pool_;
};
