cmake_minimum_required(VERSION 3.22)

project(nuGPR LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -g -lineinfo")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# cmake enabled packages
find_package(gflags CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(rmm CONFIG REQUIRED)

# cnpy package
set(CNPY_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../packages")
find_library(CNPY_LIB cnpy PATHS "${CNPY_ROOT_DIR}/lib")

include_directories(src include "${CNPY_ROOT_DIR}/include")

# GTest package
enable_testing()
add_subdirectory(tests)

add_library(gp_data src/gpdata/gp.cu)
target_link_libraries(gp_data
    PUBLIC gflags glog::glog CUDA::cublas CUDA::cusolver rmm::rmm "${CNPY_LIB}")
set_target_properties(gp_data PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(gp_model_utils
    src/gpmodel/device_rep.cu
    src/gpmodel/precond.cu
    src/gpmodel/solver.cu
    src/gpmodel/super_cholesky.cu)
target_link_libraries(gp_model_utils PUBLIC gp_data)
set_target_properties(gp_model_utils PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(gp_core src/adam/adam.cu src/gpmodel/model.cu)
target_link_libraries(gp_core PUBLIC gp_model_utils)
set_target_properties(gp_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(nuGPR main.cpp)
target_link_libraries(nuGPR PRIVATE gp_core)
