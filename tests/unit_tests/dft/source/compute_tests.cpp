/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  For your convenience, a copy of the License has been included in this
*  repository.
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
**************************************************************************/

#include <iostream>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "test_helper.hpp"
#include "test_common.hpp"
#include <gtest/gtest.h>

#include "compute_inplace.hpp"
#include "compute_inplace_real_real.hpp"
#include "compute_out_of_place.hpp"
#include "compute_out_of_place_real_real.hpp"

extern std::vector<sycl::device *> devices;

namespace {

class ComputeTests : public ::testing::TestWithParam<std::tuple<sycl::device *, std::int64_t>> {};

std::vector<std::int64_t> lengths{ 8, 21, 32 };

#define INSTANTIATE_TEST(PRECISION, DOMAIN, DIMENSIONS, PLACE, LAYOUT, STORAGE)                                        \
TEST_P(ComputeTests, DOMAIN ## _ ## PRECISION ## _ ## DIMENSIONS ## D_ ## PLACE ## _ ## LAYOUT ## STORAGE) {           \
    auto test = DFT_Test<oneapi::mkl::dft::precision::PRECISION, oneapi::mkl::dft::domain::DOMAIN, DIMENSIONS>{        \
        std::get<0>(GetParam()), std::get<1>(GetParam())                                                               \
    };                                                                                                                 \
    EXPECT_TRUEORSKIP(test.test_ ## PLACE ## _ ## LAYOUT ## STORAGE());                                                \
}

#define INSTANTIATE_TEST_DIMENSIONS(PRECISION, DOMAIN, PLACE, LAYOUT, STORAGE)    \
INSTANTIATE_TEST(PRECISION, DOMAIN, 1, PLACE, LAYOUT, STORAGE) \
INSTANTIATE_TEST(PRECISION, DOMAIN, 2, PLACE, LAYOUT, STORAGE) \
INSTANTIATE_TEST(PRECISION, DOMAIN, 3, PLACE, LAYOUT, STORAGE)

#define INSTANTIATE_TEST_DIMENSIONS_PRECISION_DOMAIN(PLACE, LAYOUT, STORAGE) \
INSTANTIATE_TEST_DIMENSIONS(SINGLE, COMPLEX, PLACE, LAYOUT, STORAGE)         \
INSTANTIATE_TEST_DIMENSIONS(SINGLE, REAL, PLACE, LAYOUT, STORAGE)            \
INSTANTIATE_TEST_DIMENSIONS(DOUBLE, COMPLEX, PLACE, LAYOUT, STORAGE)         \
INSTANTIATE_TEST_DIMENSIONS(DOUBLE, REAL, PLACE, LAYOUT, STORAGE)

#define INSTANTIATE_TEST_DIMENSIONS_PRECISION_DOMAIN_PLACE_LAYOUT(STORAGE)      \
INSTANTIATE_TEST_DIMENSIONS_PRECISION_DOMAIN(in_place, , STORAGE)               \
INSTANTIATE_TEST_DIMENSIONS_PRECISION_DOMAIN(in_place, real_real_, STORAGE)     \
INSTANTIATE_TEST_DIMENSIONS_PRECISION_DOMAIN(out_of_place, , STORAGE)           \
INSTANTIATE_TEST_DIMENSIONS_PRECISION_DOMAIN(out_of_place, real_real_, STORAGE)

INSTANTIATE_TEST_DIMENSIONS_PRECISION_DOMAIN_PLACE_LAYOUT(buffer)
INSTANTIATE_TEST_DIMENSIONS_PRECISION_DOMAIN_PLACE_LAYOUT(USM)

INSTANTIATE_TEST_SUITE_P(ComputeTestSuite, ComputeTests,
                         ::testing::Combine(testing::ValuesIn(devices), testing::ValuesIn(lengths)),
                         ::DimensionsDeviceNamePrint());

} // anonymous namespace
