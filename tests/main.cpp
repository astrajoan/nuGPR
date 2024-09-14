#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "cuda_env.h"

namespace {

class RmmTestEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    // Tests should not allocate matrices larger than 2000 * 2000 in size
    rmm_ = std::make_unique<RMMResourceWrapper>(2000 * 2000 * sizeof(float));
  }

  void TearDown() override { rmm_.reset(); }

 private:
  std::unique_ptr<RMMResourceWrapper> rmm_;
};

}  // namespace

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  // Ensure log outputs are written to the console for more visible debugging
  FLAGS_logtostdout = true;
  google::InitGoogleLogging(argv[0]);

  ::testing::AddGlobalTestEnvironment(new RmmTestEnvironment);

  return RUN_ALL_TESTS();
}
