#include "numeric.h"

#include "assume_aligned.h"
#include "mem.h"

#include <xmmintrin.h>

void yzw2v::num::Fill(float* v, const uint32_t v_size, const float value) noexcept {
    v = YZ_ASSUME_ALIGNED(v, 128);
    const auto wide_value = _mm_set1_ps(value);
    for (const auto* const v_end = v + mem::RoundSizeUpByVecSize(v_size); v < v_end; v += 4) {
        _mm_store_ps(v, wide_value);
    }
}

void yzw2v::num::Zeroize(float* const v, const uint32_t v_size) noexcept {
    Fill(v, v_size, 0.0f);
}

void yzw2v::num::DivideVector(float* v, const uint32_t v_size, const float divisor) noexcept {
    v = YZ_ASSUME_ALIGNED(v, 128);
    const auto wide_divisor = _mm_set1_ps(divisor);
    for (const auto* const v_end = v + mem::RoundSizeUpByVecSize(v_size); v < v_end; v += 4) {
        const auto x = _mm_load_ps(v);
        const auto y = _mm_div_ps(x, wide_divisor);
        _mm_store_ps(v, y);
    }
}

void yzw2v::num::MultiplyVector(float* v, const uint32_t v_size, const float multiple) noexcept {
    v = YZ_ASSUME_ALIGNED(v, 128);
    const auto wide_multiple = _mm_set1_ps(multiple);
    for (const auto* const v_end = v + mem::RoundSizeUpByVecSize(v_size); v < v_end; v += 4) {
        const auto x = _mm_load_ps(v);
        const auto y = _mm_mul_ps(x, wide_multiple);
        _mm_store_ps(v, y);
    }
}

void yzw2v::num::AddVector(float* v, const uint32_t v_size, const float* summand) noexcept {
    v = YZ_ASSUME_ALIGNED(v, 128);
    summand = YZ_ASSUME_ALIGNED(summand, 128);
    for (const auto* const v_end = v + mem::RoundSizeUpByVecSize(v_size); v < v_end; summand += 4, v += 4) {
        const auto x = _mm_load_ps(v);
        const auto y = _mm_load_ps(summand);
        const auto z = _mm_add_ps(x, y);
        _mm_store_ps(v, z);
    }
}

void yzw2v::num::AddVector(float* v, const uint32_t v_size,
                           const float* summand, const float summand_multiple) noexcept {
    v = YZ_ASSUME_ALIGNED(v, 128);
    summand = YZ_ASSUME_ALIGNED(summand, 128);
    const auto wide_summand_multiple = _mm_set1_ps(summand_multiple);
    for (const auto* const v_end = v + mem::RoundSizeUpByVecSize(v_size); v < v_end; summand += 4, v += 4) {
        const auto x = _mm_load_ps(summand);
        const auto y = _mm_load_ps(v);
        const auto z = _mm_mul_ps(x, wide_summand_multiple);
        const auto w = _mm_add_ps(y, z);
        _mm_store_ps(v, w);
    }
}

float yzw2v::num::ScalarProduct(const float* v, const uint32_t v_size,
                                const float* rhs) noexcept {
    v = YZ_ASSUME_ALIGNED(v, 128);
    rhs = YZ_ASSUME_ALIGNED(rhs, 128);
    __m128 wide_res = {};
    for (const auto* const v_end = v + mem::RoundSizeUpByVecSize(v_size); v < v_end; v += 4, rhs += 4) {
        const auto x = _mm_load_ps(v);
        const auto y = _mm_load_ps(rhs);
        const auto z = _mm_mul_ps(x, y);
        wide_res = _mm_add_ps(wide_res, z);
    }

    return wide_res[0] + wide_res[1] + wide_res[2] + wide_res[3];
}
