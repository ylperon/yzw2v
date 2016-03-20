#include "numeric.h"

void yzw2v::num::Fill(float* v, const uint32_t v_size, const float value) noexcept {
    for (auto i = uint32_t{}; i < v_size; ++i) {
        v[i] = value;
    }
}

void yzw2v::num::Zeroize(float* v, const uint32_t v_size) noexcept {
    Fill(v, v_size, 0.0f);
}

void yzw2v::num::DivideVector(float* v, const uint32_t v_size, const float divisor) noexcept {
    for (auto i = uint32_t{}; i < v_size; ++i) {
        v[i] /= divisor;
    }
}

void yzw2v::num::MultiplyVector(float* v, const uint32_t v_size, const float multiple) noexcept {
    for (auto i = uint32_t{}; i < v_size; ++i) {
        v[i] *= multiple;
    }
}

void yzw2v::num::AddVector(float* v, const uint32_t v_size,
                           const float* const summand) noexcept {
    for (auto i = uint32_t{}; i < v_size; ++i) {
        v[i] += summand[i];
    }
}

void yzw2v::num::AddVector(float* v, const uint32_t v_size,
                           const float* const summand, const float summand_multiple) noexcept {
    for (auto i = uint32_t{}; i < v_size; ++i) {
        v[i] += summand_multiple * summand[i];
    }
}

float yzw2v::num::ScalarProduct(const float* lhs, const uint32_t lhs_size,
                                const float* rhs) noexcept {
    auto res = float{};
    for (auto i = uint32_t{}; i < lhs_size; ++i) {
        res += lhs[i] * rhs[i];
    }

    return res;
}
