cmake_minimum_required(VERSION 3.22)

# test packages
find_package(GTest CONFIG REQUIRED)

include(GoogleTest)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}" ${GTEST_INCLUDE_DIRS})

add_library(test_helper test_helper.cpp)
target_link_libraries(test_helper PUBLIC glog::glog GTest::gtest)

add_library(dt_helper device_test_helper.cu)
target_link_libraries(dt_helper PUBLIC gp_core)
set_target_properties(dt_helper PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(all_tests
    main.cpp
    adam.test.cpp
    device_functions.test.cpp
    device_rep.test.cpp
    precond_solver.test.cpp
    model.test.cpp)
target_link_libraries(all_tests PRIVATE test_helper dt_helper)

gtest_discover_tests(all_tests)
