#include "numeric.h"

#include "mem.h"

#include <xmmintrin.h>

void yzw2v::num::Fill(float* v, const uint32_t v_size, const float value) noexcept {
    const auto wide_value = _mm_set1_ps(value);
    for (const auto* const v_end = v + mem::RoundSizeUpByVecSize(v_size); v < v_end; v += 4) {
        _mm_store_ps(v, wide_value);
    }
}

void yzw2v::num::Zeroize(float* const v, const uint32_t v_size) noexcept {
    Fill(v, v_size, 0.0f);
}

void yzw2v::num::DivideVector(float* v, const uint32_t v_size, const float divisor) noexcept {
    const auto wide_divisor = _mm_set1_ps(divisor);
    for (const auto* const v_end = v + mem::RoundSizeUpByVecSize(v_size); v < v_end; v += 4) {
        _mm_store_ps(v, _mm_div_ps(_mm_load_ps(v), wide_divisor));
    }
}

void yzw2v::num::MultiplyVector(float* v, const uint32_t v_size, const float multiple) noexcept {
    const auto wide_multiple = _mm_set1_ps(multiple);
    for (const auto* const v_end = v + mem::RoundSizeUpByVecSize(v_size); v < v_end; v += 4) {
        _mm_store_ps(v, _mm_mul_ps(_mm_load_ps(v), wide_multiple));
    }
}

void yzw2v::num::AddVector(float* v, const uint32_t v_size, const float* summand) noexcept {
    for (const auto* const v_end = v + mem::RoundSizeUpByVecSize(v_size); v < v_end; summand += 4, v += 4) {
        _mm_store_ps(v, _mm_add_ps(_mm_load_ps(v), _mm_load_ps(summand)));
    }
}

void yzw2v::num::AddVector(float* v, const uint32_t v_size,
                           const float* summand, const float summand_multiple) noexcept {
    const auto wide_summand_multiple = _mm_set1_ps(summand_multiple);
    for (const auto* const v_end = v + mem::RoundSizeUpByVecSize(v_size); v < v_end; summand += 4, v += 4) {
        _mm_store_ps(v, _mm_add_ps(_mm_load_ps(v),
                                   _mm_mul_ps(wide_summand_multiple,
                                              _mm_load_ps(summand)
                                              )
                                   )
                     );
    }
}

float yzw2v::num::ScalarProduct(const float* v, const uint32_t v_size,
                                const float* rhs) noexcept {
    __m128 wide_res = {};
    for (const auto* const v_end = v + mem::RoundSizeUpByVecSize(v_size); v < v_end; v += 4, rhs += 4) {
        wide_res = _mm_add_ps(wide_res, _mm_mul_ps(_mm_load_ps(v), _mm_load_ps(rhs)));
    }

    return wide_res[0] + wide_res[1] + wide_res[2] + wide_res[3];
}
