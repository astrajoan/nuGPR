#pragma once

#include "gpdata/gp.h"

void dmemcpy_wrapper(const float* cpu, float* gpu, size_t sz, bool to_gpu);
void dmemcpy_wrapper(float* cpu, const float* gpu, size_t sz, bool to_gpu);
void dmemcpy_wrapper(float* cpu, float* gpu, size_t sz, bool to_gpu);

void init_mock_data(GPData& data, size_t ones_sz);
void init_mock_data(GPData& data, const vf& train_x, const vf& train_y,
                    const vf& test_x, const vf& rep_x, const vf& rand_z,
                    size_t n_train, size_t n_test, size_t n_rand,
                    size_t n_blocks, size_t d);

void generate_covar_gpu(vf& covar, const vf& x1, const vf& x2, size_t d,
                        size_t n_blocks, float lengthscale, float noise,
                        float output_scale);

void generate_confidence_region_gpu(vf& upper, vf& lower, const vf& mean,
                                    const vf& covar, size_t n, float noise);

void syevd_test_impl();
void syevdx_test_impl();
