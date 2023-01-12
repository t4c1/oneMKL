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

#ifndef ONEMKL_REFERENCE_DFT_HPP
#define ONEMKL_REFERENCE_DFT_HPP

template <typename TypeIn, typename TypeOut>
void reference_forward_dft_impl(TypeIn* in, TypeOut* out, size_t N, size_t stride) {
    static_assert(is_complex<TypeOut>());

    double TWOPI = 2.0 * std::atan(1.0) * 4.0;

    std::complex<double> out_temp; /* Do the calculations using double */
    for (int k = 0; k < N; k++) {
        out[k*stride] = 0;
        out_temp = 0;
        for (int n = 0; n < N; n++) {
            if constexpr (is_complex<TypeIn>()) {
                out_temp += static_cast<std::complex<double>>(in[n*stride]) *
                            std::complex<double>{ std::cos(n * k * TWOPI / N),
                                                  -std::sin(n * k * TWOPI / N) };
            }
            else {
                out_temp +=
                    std::complex<double>{ static_cast<double>(in[n*stride]) * std::cos(n * k * TWOPI / N),
                                          static_cast<double>(-in[n*stride]) *
                                              std::sin(n * k * TWOPI / N) };
            }
        }
        out[k*stride] = static_cast<TypeOut>(out_temp);
    }
}

template <typename TypeIn, typename TypeOut, int dimms>
struct reference{};

template <typename TypeIn, typename TypeOut>
struct reference<TypeIn, TypeOut, 1>{
    static void forward_dft(std::vector<TypeIn> &in, std::vector<TypeOut> &out) {
        reference_forward_dft_impl(in.data(), out.data(), out.size(), 1);
    }
};

template <typename TypeIn, typename TypeOut>
struct reference<TypeIn, TypeOut, 2>{
    static void forward_dft(std::vector<TypeIn> &in, std::vector<TypeOut> &out) {
        std::vector<std::complex<double>> tmp(out.size());
        size_t N = static_cast<size_t>(std::round(std::pow(out.size(), 0.5)));
        for(size_t i=0;i<N*N;i+=N){
            reference_forward_dft_impl(in.data()+i, tmp.data()+i, N, 1);
        }
        for(size_t i=0;i<N;i++){
            reference_forward_dft_impl(tmp.data()+i, out.data()+i, N, N);
        }
    }
};

template <typename TypeIn, typename TypeOut>
struct reference<TypeIn, TypeOut, 3>{
    static void forward_dft(std::vector<TypeIn> &in, std::vector<TypeOut> &out) {
        std::vector<std::complex<double>> tmp1(out.size());
        std::vector<std::complex<double>> tmp2(out.size());
        size_t N = static_cast<size_t>(std::round(std::pow(out.size(), 1.0 / 3.0)));
        for(size_t i=0;i<N*N*N;i+=N){
            reference_forward_dft_impl(in.data()+i, tmp1.data()+i, N, 1);
        }
        for(size_t j=0;j<N*N*N;j+=N*N){
            for(size_t i=0;i<N;i++){
                reference_forward_dft_impl(tmp1.data()+i+j, tmp2.data()+i+j, N, N);
            }
        }
        for(size_t i=0;i<N*N;i++){
            reference_forward_dft_impl(tmp2.data()+i, out.data()+i, N, N*N);
        }
    }
};

#endif //ONEMKL_REFERENCE_DFT_HPP
