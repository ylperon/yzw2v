#include "numeric.h"

#include <xmmintrin.h>

void yzw2v::num::Fill(float* v, const uint32_t v_size, const float value) noexcept {
    const auto* const v_end = v + v_size;
    const auto wide_value = _mm_set1_ps(value);
    for (const auto* const v_this_end = v_end - 3; v < v_this_end; v += 4) {
        _mm_prefetch(v + 4, _MM_HINT_T0);
        _mm_store_ps(v, wide_value);
    }

    for (; v < v_end;) {
        *(v++) = value;
    }
}

void yzw2v::num::Zeroize(float* const v, const uint32_t v_size) noexcept {
    Fill(v, v_size, 0.0f);
}

void yzw2v::num::DivideVector(float* v, const uint32_t v_size, const float divisor) noexcept {
    const auto* const v_end = v + v_size;
    const auto wide_divisor = _mm_set1_ps(divisor);
    for (const auto* const v_this_end = v_end - 3; v < v_this_end; v += 4) {
        _mm_prefetch(v + 4, _MM_HINT_T0);
        _mm_store_ps(v, _mm_div_ps(_mm_load_ps(v), wide_divisor));
    }

    for (; v < v_end;) {
        *(v++) /= divisor;
    }
}

void yzw2v::num::MultiplyVector(float* v, const uint32_t v_size, const float multiple) noexcept {
    const auto* const v_end = v + v_size;
    const auto wide_multiple = _mm_set1_ps(multiple);
    for (const auto* const v_this_end = v_end - 3; v < v_this_end; v += 4) {
        _mm_prefetch(v + 4, _MM_HINT_T0);
        _mm_store_ps(v, _mm_mul_ps(_mm_load_ps(v), wide_multiple));
    }

    for (; v < v_end; ++v) {
        *(v++) *= multiple;
    }
}

void yzw2v::num::AddVector(float* v, const uint32_t v_size, const float* summand) noexcept {
    const auto* const v_end = v + v_size;
    for (const auto* const v_this_end = v_end - 3; v < v_this_end; summand += 4, v += 4) {
        _mm_prefetch(v + 4, _MM_HINT_T0);
        _mm_store_ps(v, _mm_add_ps(_mm_load_ps(v), _mm_load_ps(summand)));
    }

    for (; v < v_end;) {
        *(v++) += *(summand++);
    }
}

void yzw2v::num::AddVector(float* v, const uint32_t v_size,
                           const float* summand, const float summand_multiple) noexcept {
    const auto* const v_end = v + v_size;
    const auto wide_summand_multiple = _mm_set1_ps(summand_multiple);
    for (const auto* const v_this_end = v_end - 3; v < v_this_end; summand += 4, v += 4) {
        _mm_prefetch(v + 4, _MM_HINT_T0);
        _mm_prefetch(summand + 4, _MM_HINT_T0);
        _mm_store_ps(v, _mm_add_ps(_mm_load_ps(v),
                                   _mm_mul_ps(wide_summand_multiple,
                                              _mm_load_ps(summand)
                                              )
                                   )
                     );
    }

    for (; v < v_end;) {
        *(v++) += summand_multiple * *(summand++);
    }
}

float yzw2v::num::ScalarProduct(const float* v, const uint32_t v_size,
                                const float* rhs) noexcept {
    __m128 wide_res = {};
    const auto* const v_end = v + v_size;
    for (const auto* const v_this_end = v_end - 3; v < v_this_end; v += 4, rhs += 4) {
        _mm_prefetch(v + 4, _MM_HINT_T0);
        _mm_prefetch(rhs + 4, _MM_HINT_T0);
        wide_res = _mm_add_ps(wide_res, _mm_mul_ps(_mm_load_ps(v), _mm_load_ps(rhs)));
    }

    for (; v < v_end;) {
        wide_res[0] += *(v++) * *(rhs++);
    }

    return wide_res[0] + wide_res[1] + wide_res[2] + wide_res[3];
}
