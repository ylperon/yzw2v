#pragma once

#include <cinttypes>

namespace yzw2v {
    namespace num {
        void Fill(float* v, const uint32_t v_size, const float value) noexcept;
        void Zeroize(float* v, const uint32_t v_size) noexcept;
        void Prefetch(const float* v) noexcept;

        void DivideVector(float* v, const uint32_t v_size, const float divisor) noexcept;

        void MultiplyVector(float* v, const uint32_t v_size, const float multiple) noexcept;

        void AddVector(float* v, const uint32_t v_size, const float* summand) noexcept;
        void AddVector(float* v, const uint32_t v_size,
                       const float* summand, const float summand_multiple) noexcept;

        float ScalarProduct(const float* lhs, const uint32_t lhs_size, const float* rhs) noexcept;
    }
}
