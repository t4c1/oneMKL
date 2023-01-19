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

#ifndef ONEMKL_COMPUTE_INPLACE_HPP
#define ONEMKL_COMPUTE_INPLACE_HPP

#include "compute_tester.hpp"

template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain, int dimms>
int DFT_Test<precision, domain, dimms>::test_in_place_buffer() {
    if (!init(TestType::buffer)) {
        return test_skipped;
    }

    descriptor_t descriptor = descriptor_factory<descriptor_t, dimms>::get(size);
    descriptor.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                         oneapi::mkl::dft::config_value::INPLACE);


    const size_t container_size =
        domain == oneapi::mkl::dft::domain::REAL ? conjugate_even_size : size;
    const size_t container_size_total = 
        container_size * static_cast<size_t>(std::pow(size, dimms-1));

    sycl::buffer<InputType, 1> inout_dev {sycl::range<1>(container_size_total)};
    std::vector<InputType> out_host(container_size_total);

    if constexpr(domain == oneapi::mkl::dft::domain::REAL){
        if constexpr(dimms == 1){
            copy_to_device(sycl_queue, input, inout_dev);
        }
        if constexpr(dimms == 2){
            std::vector<InputType> strided(container_size_total);
            for(int i=0;i<size;i++){
                for(int j=0;j<size;j++){
                    strided[i * (size+2) + j] = input[i*size + j];
                }
            }
            copy_to_device(sycl_queue, strided, inout_dev);
            std::array<std::int64_t, dimms + 1> real_strides{0, 2*(size/2 + 1), 1};
            std::array<std::int64_t, dimms + 1> complex_strides{0, (size/2 + 1), 1};
            descriptor.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, real_strides.data());
            descriptor.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, complex_strides.data());
        }
        if constexpr(dimms == 3){
            std::vector<InputType> strided(container_size_total);
            for(int i=0;i<size*size;i++){
                for(int j=0;j<size;j++){
                    strided[i * (size+2) + j] = input[i*size + j];
                }
            }
            copy_to_device(sycl_queue, strided, inout_dev);
            std::array<std::int64_t, dimms + 1> real_strides{0, 2 * size * (size/2 + 1), 2 * (size/2 +1), 1};
            std::array<std::int64_t, dimms + 1> complex_strides{0, size * (size/2 + 1), (size/2 + 1), 1};
            descriptor.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, real_strides.data());
            descriptor.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, complex_strides.data());
        }
    } else{
        copy_to_device(sycl_queue, input, inout_dev);
    }

    commit_descriptor(descriptor, sycl_queue);

    try {
        oneapi::mkl::dft::compute_forward<descriptor_t, InputType>(descriptor, inout_dev);
    }
    catch (oneapi::mkl::unimplemented &e) {
        std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
        return test_skipped;
    }

    copy_to_host(sycl_queue, inout_dev, out_host);

    if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
        std::vector<InputType> out_host_ref_conjugate = std::vector<InputType>(container_size_total);
        for (int j = 0; j < container_size_total/container_size; j++) {
            for (int i = 0; i < container_size; i += 2) {
                out_host_ref_conjugate[j*container_size + i] = out_host_ref[j*size + i / 2].real();
                out_host_ref_conjugate[j*container_size + i + 1] = out_host_ref[j*size + i / 2].imag();
            }
        }
        EXPECT_TRUE(check_equal_vector(out_host.data(), out_host_ref_conjugate.data(),
                                       out_host.size(), 1, error_margin, std::cout));
    }
    else {
        EXPECT_TRUE(check_equal_vector(out_host.data(), out_host_ref.data(), out_host.size(), 1,
                                       error_margin, std::cout));
    }

    descriptor_t descriptor_back = descriptor_factory<descriptor_t, dimms>::get(size);
    descriptor_back.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                              oneapi::mkl::dft::config_value::INPLACE);
    descriptor_back.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, (1.0 / size_total));
    if constexpr(domain == oneapi::mkl::dft::domain::REAL){
        if constexpr(dimms == 2){
            std::array<std::int64_t, dimms + 1> complex_strides{0, (size/2 + 1), 1};
            std::array<std::int64_t, dimms + 1> real_strides{0, 2*(size/2 + 1), 1};
            descriptor_back.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, complex_strides.data());
            descriptor_back.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, real_strides.data());
        }
        if constexpr(dimms == 3){
            std::array<std::int64_t, dimms + 1> complex_strides{0, size * (size/2 + 1), (size/2 +1), 1};
            std::array<std::int64_t, dimms + 1> real_strides{0, 2 * size * (size/2 + 1), 2 * (size/2 + 1), 1};
            descriptor_back.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, complex_strides.data());
            descriptor_back.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, real_strides.data());
        }
    }
    commit_descriptor(descriptor_back, sycl_queue);

    try {
        oneapi::mkl::dft::compute_backward<std::remove_reference_t<decltype(descriptor_back)>,
                                           InputType>(descriptor_back, inout_dev);
    }
    catch (oneapi::mkl::unimplemented &e) {
        std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
        return test_skipped;
    }

    copy_to_host(sycl_queue, inout_dev, out_host);

    if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
        for (int j = 0; j < size_total/size; j++) {
            EXPECT_TRUE(check_equal_vector(out_host.data() + j*(size+2), input.data() + j*size, size, 1, error_margin,
                                    std::cout));
        }
    } else{
        EXPECT_TRUE(check_equal_vector(out_host.data(), input.data(), input.size(), 1, error_margin,
                                   std::cout));
    }

    return !::testing::Test::HasFailure();
}

template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain, int dimms>
int DFT_Test<precision, domain, dimms>::test_in_place_USM() {
    if (!init(TestType::usm)) {
        return test_skipped;
    }

    const size_t container_size =
        domain == oneapi::mkl::dft::domain::REAL ? conjugate_even_size : size;
    const size_t container_size_total = 
        container_size * static_cast<size_t>(std::pow(size, dimms-1));

    auto ua_input = usm_allocator_t<InputType>(cxt, *dev);
    std::vector<InputType, decltype(ua_input)> inout(container_size_total, ua_input);

    descriptor_t descriptor = descriptor_factory<descriptor_t, dimms>::get(size);
    descriptor.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                         oneapi::mkl::dft::config_value::INPLACE);

    if constexpr(domain == oneapi::mkl::dft::domain::REAL){
        if constexpr(dimms == 1){
            std::copy(input.begin(), input.end(), inout.begin());
        }
        if constexpr(dimms == 2){
            for(int i=0;i<size;i++){
                for(int j=0;j<size;j++){
                    inout[i * (size+2) + j] = input[i*size + j];
                }
            }
            std::array<std::int64_t, dimms + 1> real_strides{0, 2*(size/2 + 1), 1};
            std::array<std::int64_t, dimms + 1> complex_strides{0, (size/2 + 1), 1};
            descriptor.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, real_strides.data());
            descriptor.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, complex_strides.data());
        }
        if constexpr(dimms == 3){
            for(int i=0;i<size*size;i++){
                for(int j=0;j<size;j++){
                    inout[i * (size+2) + j] = input[i*size + j];
                }
            }
            std::array<std::int64_t, dimms + 1> real_strides{0, 2 * size * (size/2 + 1), 2 * (size/2 +1), 1};
            std::array<std::int64_t, dimms + 1> complex_strides{0, size * (size/2 + 1), (size/2 + 1), 1};
            descriptor.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, real_strides.data());
            descriptor.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, complex_strides.data());
        }
    } else{
        std::copy(input.begin(), input.end(), inout.begin());
    }

    commit_descriptor(descriptor, sycl_queue);

    try {
        std::vector<sycl::event> dependencies;
        sycl::event done = oneapi::mkl::dft::compute_forward<descriptor_t, InputType>(
            descriptor, inout.data(), dependencies);
        done.wait();
    }
    catch (oneapi::mkl::unimplemented &e) {
        std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
        return test_skipped;
    }

    if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
        std::vector<InputType> out_host_ref_conjugate = std::vector<InputType>(container_size_total);
        for (int j = 0; j < container_size_total/container_size; j++) {
            for (int i = 0; i < container_size; i += 2) {
                out_host_ref_conjugate[j*container_size + i] = out_host_ref[j*size + i / 2].real();
                out_host_ref_conjugate[j*container_size + i + 1] = out_host_ref[j*size + i / 2].imag();
            }
        }
        EXPECT_TRUE(check_equal_vector(inout.data(), out_host_ref_conjugate.data(), inout.size(), 1,
                                       error_margin, std::cout));
    }
    else {
        EXPECT_TRUE(check_equal_vector(inout.data(), out_host_ref.data(), inout.size(), 1,
                                       error_margin, std::cout));
    }

    descriptor_t descriptor_back = descriptor_factory<descriptor_t, dimms>::get(size);
    descriptor_back.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                              oneapi::mkl::dft::config_value::INPLACE);
    descriptor_back.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, (1.0 / size_total));
    if constexpr(domain == oneapi::mkl::dft::domain::REAL){
        if constexpr(dimms == 2){
            std::array<std::int64_t, dimms + 1> complex_strides{0, (size/2 + 1), 1};
            std::array<std::int64_t, dimms + 1> real_strides{0, 2*(size/2 + 1), 1};
            descriptor_back.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, complex_strides.data());
            descriptor_back.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, real_strides.data());
        }
        if constexpr(dimms == 3){
            std::array<std::int64_t, dimms + 1> complex_strides{0, size * (size/2 + 1), (size/2 +1), 1};
            std::array<std::int64_t, dimms + 1> real_strides{0, 2 * size * (size/2 + 1), 2 * (size/2 + 1), 1};
            descriptor_back.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, complex_strides.data());
            descriptor_back.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, real_strides.data());
        }
    }
    commit_descriptor(descriptor_back, sycl_queue);

    try {
        std::vector<sycl::event> dependencies;
        sycl::event done =
            oneapi::mkl::dft::compute_backward<std::remove_reference_t<decltype(descriptor_back)>,
                                               InputType>(descriptor_back, inout.data());
        done.wait();
    }
    catch (oneapi::mkl::unimplemented &e) {
        std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
        return test_skipped;
    }

    if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
        for (int j = 0; j < size_total/size; j++) {
            EXPECT_TRUE(check_equal_vector(inout.data() + j*(size+2), input.data() + j*size, size, 1, error_margin,
                                    std::cout));
        }
    } else{
    EXPECT_TRUE(
        check_equal_vector(inout.data(), input.data(), input.size(), 1, error_margin, std::cout));
    }

    return !::testing::Test::HasFailure();
}

#endif //ONEMKL_COMPUTE_INPLACE_HPP
