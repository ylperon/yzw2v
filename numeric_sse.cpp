#include "numeric.h"

#include "assume.h"
#include "assume_aligned.h"
#include "mem.h"

#include <xmmintrin.h>

void yzw2v::num::Prefetch(const float* v) noexcept {
    v = YZ_ASSUME_ALIGNED(v, 128);
    _mm_prefetch(v, _MM_HINT_T0);
    _mm_prefetch(v + 4, _MM_HINT_T0);
    _mm_prefetch(v + 8, _MM_HINT_T0);
    _mm_prefetch(v + 12, _MM_HINT_T0);
}

void yzw2v::num::Fill(float* v, const uint32_t v_size, const float value) noexcept {
    const auto v_size_rounded_up = mem::RoundSizeUpByVecSize(v_size);
    v = YZ_ASSUME_ALIGNED(v, 128);
    const auto wide_value = _mm_set1_ps(value);
    for (const auto* const v_end = v + v_size_rounded_up; v < v_end; v += 4) {
        _mm_store_ps(v, wide_value);
    }
}

void yzw2v::num::Zeroize(float* const v, const uint32_t v_size) noexcept {
    Fill(v, v_size, 0.0f);
}

void yzw2v::num::DivideVector(float* v, const uint32_t v_size, const float divisor) noexcept {
    const auto v_size_rounded_up = mem::RoundSizeUpByVecSize(v_size);
    v = YZ_ASSUME_ALIGNED(v, 128);
    const auto wide_divisor = _mm_set1_ps(divisor);
    for (const auto* const v_end = v + v_size_rounded_up; v < v_end; v += 4) {
        _mm_store_ps(v, _mm_div_ps(_mm_load_ps(v), wide_divisor));
    }
}

void yzw2v::num::MultiplyVector(float* v, const uint32_t v_size, const float multiple) noexcept {
    const auto v_size_rounded_up = mem::RoundSizeUpByVecSize(v_size);
    v = YZ_ASSUME_ALIGNED(v, 128);
    const auto wide_multiple = _mm_set1_ps(multiple);
    for (const auto* const v_end = v + v_size_rounded_up; v < v_end; v += 4) {
        _mm_store_ps(v, _mm_mul_ps(_mm_load_ps(v), wide_multiple));
    }
}

void yzw2v::num::AddVector(float* v, const uint32_t v_size, const float* summand) noexcept {
    const auto v_size_rounded_up = mem::RoundSizeUpByVecSize(v_size);
    v = YZ_ASSUME_ALIGNED(v, 128);
    summand = YZ_ASSUME_ALIGNED(summand, 128);
    for (const auto* const v_end = v + v_size_rounded_up; v < v_end; summand += 4, v += 4) {
        _mm_store_ps(v, _mm_add_ps(_mm_load_ps(v), _mm_load_ps(summand)));
    }
}

void yzw2v::num::AddVector(float* v, const uint32_t v_size,
                           const float* summand, const float summand_multiple) noexcept {
    const auto v_size_rounded_up = mem::RoundSizeUpByVecSize(v_size);
    v = YZ_ASSUME_ALIGNED(v, 128);
    summand = YZ_ASSUME_ALIGNED(summand, 128);
    const auto wide_summand_multiple = _mm_set1_ps(summand_multiple);
    for (const auto* const v_end = v + v_size_rounded_up; v < v_end; summand += 4, v += 4) {
        _mm_store_ps(v, _mm_add_ps(_mm_load_ps(v),
                                   _mm_mul_ps(wide_summand_multiple, _mm_load_ps(summand))
                                  )
                    );
    }
}

float yzw2v::num::ScalarProduct(const float* v, const uint32_t v_size,
                                const float* rhs) noexcept {
    const auto v_size_rounded_up = mem::RoundSizeUpByVecSize(v_size);
    v = YZ_ASSUME_ALIGNED(v, 128);
    rhs = YZ_ASSUME_ALIGNED(rhs, 128);
    __m128 wide_res = {};
    for (const auto* const v_end = v + v_size_rounded_up; v < v_end; v += 4, rhs += 4) {
        wide_res = _mm_add_ps(wide_res, _mm_mul_ps(_mm_load_ps(v), _mm_load_ps(rhs)));
    }

    return wide_res[0] + wide_res[1] + wide_res[2] + wide_res[3];
}
