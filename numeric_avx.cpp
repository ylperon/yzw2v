#include "numeric.h"

#include "assume_aligned.h"
#include "mem.h"

#include <immintrin.h>

void yzw2v::num::Fill(float* v, const uint32_t v_size, const float value) noexcept {
    v = YZ_ASSUME_ALIGNED(v, 256);
    const auto wide_value = _mm256_set1_ps(value);
    for (const auto* const v_end = v + mem::RoundSizeUpByVecSize(v_size); v < v_end; v += 8) {
        _mm256_store_ps(v, wide_value);
    }
}

void yzw2v::num::Zeroize(float* const v, const uint32_t v_size) noexcept {
    Fill(v, v_size, 0.0f);
}

void yzw2v::num::DivideVector(float* v, const uint32_t v_size, const float divisor) noexcept {
    v = YZ_ASSUME_ALIGNED(v, 256);
    const auto wide_divisor = _mm256_set1_ps(divisor);
    for (const auto* const v_end = v + mem::RoundSizeUpByVecSize(v_size); v < v_end; v += 8) {
        _mm256_store_ps(v, _mm256_div_ps(_mm256_load_ps(v), wide_divisor));
    }
}

void yzw2v::num::MultiplyVector(float* v, const uint32_t v_size, const float multiple) noexcept {
    v = YZ_ASSUME_ALIGNED(v, 256);
    const auto wide_multiple = _mm256_set1_ps(multiple);
    for (const auto* const v_end = v + mem::RoundSizeUpByVecSize(v_size); v < v_end; v += 8) {
        _mm256_store_ps(v, _mm256_mul_ps(_mm256_load_ps(v), wide_multiple));
    }
}

void yzw2v::num::AddVector(float* v, const uint32_t v_size, const float* summand) noexcept {
    v = YZ_ASSUME_ALIGNED(v, 256);
    for (const auto* const v_end = v + mem::RoundSizeUpByVecSize(v_size); v < v_end; summand += 8, v += 8) {
        _mm256_store_ps(v, _mm256_add_ps(_mm256_load_ps(v), _mm256_load_ps(summand)));
    }
}

void yzw2v::num::AddVector(float* v, const uint32_t v_size,
                           const float* summand, const float summand_multiple) noexcept {
    v = YZ_ASSUME_ALIGNED(v, 256);
    summand = YZ_ASSUME_ALIGNED(summand, 256);
    const auto wide_summand_multiple = _mm256_set1_ps(summand_multiple);
    for (const auto* const v_end = v + mem::RoundSizeUpByVecSize(v_size); v < v_end; summand += 8, v += 8) {
        _mm256_store_ps(v, _mm256_add_ps(_mm256_load_ps(v),
                                         _mm256_mul_ps(wide_summand_multiple,
                                                       _mm256_load_ps(summand)
                                                       )
                                         )
                        );
    }
}

float yzw2v::num::ScalarProduct(const float* v, const uint32_t v_size,
                                const float* rhs) noexcept {
    v = YZ_ASSUME_ALIGNED(v, 256);
    rhs = YZ_ASSUME_ALIGNED(rhs, 256);
    __m256 wide_res = {};
    for (const auto* const v_end = v + mem::RoundSizeUpByVecSize(v_size); v < v_end; v += 8, rhs += 8) {
        wide_res = _mm256_add_ps(wide_res, _mm256_mul_ps(_mm256_load_ps(v), _mm256_load_ps(rhs)));
    }

    return wide_res[0] + wide_res[1] + wide_res[2] + wide_res[3]
           + wide_res[4] + wide_res[5] + wide_res[6] + wide_res[7];
}
