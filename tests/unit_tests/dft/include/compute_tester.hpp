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

#ifndef ONEMKL_COMPUTE_TESTER_HPP
#define ONEMKL_COMPUTE_TESTER_HPP

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include "oneapi/mkl.hpp"
#include "test_helper.hpp"
#include "test_common.hpp"
#include "reference_dft.hpp"

template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain, int dimms = 1>
struct DFT_Test {
    using descriptor_t = oneapi::mkl::dft::descriptor<precision, domain>;

    template <typename ElemT>
    using usm_allocator_t = sycl::usm_allocator<ElemT, sycl::usm::alloc::shared, 64>;

    using PrecisionType =
        typename std::conditional_t<precision == oneapi::mkl::dft::precision::SINGLE, float,
                                    double>;

    using InputType = typename std::conditional_t<domain == oneapi::mkl::dft::domain::REAL,
                                                  PrecisionType, std::complex<PrecisionType>>;
    using OutputType = std::complex<PrecisionType>;

    enum class TestType { buffer, usm };

    const std::int64_t size;
    const std::int64_t size_total;
    const std::int64_t conjugate_even_size;
    static constexpr int error_margin = 10;

    sycl::device *dev;
    sycl::queue sycl_queue;
    sycl::context cxt;

    std::vector<InputType> input;
    std::vector<PrecisionType> input_re;
    std::vector<PrecisionType> input_im;
    std::vector<OutputType> out_host_ref;

    DFT_Test(sycl::device *dev, std::int64_t size);

    bool skip_test(TestType type);
    bool init(TestType type);

    int test_in_place_buffer();
    int test_in_place_real_real_buffer();
    int test_out_of_place_buffer();
    int test_out_of_place_real_real_buffer();
    int test_in_place_USM();
    int test_in_place_real_real_USM();
    int test_out_of_place_USM();
    int test_out_of_place_real_real_USM();
};

template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain, int dimms>
DFT_Test<precision, domain, dimms>::DFT_Test(sycl::device *dev, std::int64_t size)
        : dev{ dev },
          size{ static_cast<std::int64_t>(size) },
          size_total{ static_cast<std::int64_t>(std::round(std::pow(size, dimms)))},
          conjugate_even_size{ 2 * (size / 2 + 1) },
          sycl_queue{ *dev, exception_handler },
          cxt{ sycl_queue.get_context() } {
    input = std::vector<InputType>(size_total);
    input_re = std::vector<PrecisionType>(size_total);
    input_im = std::vector<PrecisionType>(size_total);

    out_host_ref = std::vector<OutputType>(size_total);
    rand_vector(input, size_total, 1);

    if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
        for (int i = 0; i < input.size(); ++i) {
            input_re[i] = { input[i] };
            input_im[i] = 0;
        }
    }
    else {
        for (int i = 0; i < input.size(); ++i) {
            input_re[i] = { input[i].real() };
            input_im[i] = { input[i].imag() };
        }
    }
}

template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain, int dimms>
bool DFT_Test<precision, domain, dimms>::skip_test(TestType type) {
    if constexpr (precision == oneapi::mkl::dft::precision::DOUBLE) {
        if (!sycl_queue.get_device().has(sycl::aspect::fp64)) {
            std::cout << "Device does not support double precision." << std::endl;
            return true;
        }
    }

    if (type == TestType::usm &&
        !sycl_queue.get_device().has(sycl::aspect::usm_shared_allocations)) {
        std::cout << "Device does not support usm shared allocations." << std::endl;
        return true;
    }

    return false;
}

template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain, int dimms>
bool DFT_Test<precision, domain, dimms>::init(TestType type) {
    reference<InputType, OutputType, dimms>::forward_dft(input, out_host_ref);
    return !skip_test(type);
}

#endif //ONEMKL_COMPUTE_TESTER_HPP
