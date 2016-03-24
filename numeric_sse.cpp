#include "numeric.h"

#include <xmmintrin.h>

static void Prefetch8(const float* const v) noexcept {
    _mm_prefetch(v, _MM_HINT_T0);
    _mm_prefetch(v + 4, _MM_HINT_T0);
    _mm_prefetch(v + 8, _MM_HINT_T0);
    _mm_prefetch(v + 12, _MM_HINT_T0);
    _mm_prefetch(v + 16, _MM_HINT_T0);
    _mm_prefetch(v + 20, _MM_HINT_T0);
    _mm_prefetch(v + 24, _MM_HINT_T0);
    _mm_prefetch(v + 28, _MM_HINT_T0);
}

static void Prefetch8(float* const v) noexcept {
    _mm_prefetch(v, _MM_HINT_T0);
    _mm_prefetch(v + 4, _MM_HINT_T0);
    _mm_prefetch(v + 8, _MM_HINT_T0);
    _mm_prefetch(v + 12, _MM_HINT_T0);
    _mm_prefetch(v + 16, _MM_HINT_T0);
    _mm_prefetch(v + 20, _MM_HINT_T0);
    _mm_prefetch(v + 24, _MM_HINT_T0);
    _mm_prefetch(v + 28, _MM_HINT_T0);
}

static void Fill8(float* const v, const __m128 wide_value) noexcept {
    _mm_stream_ps(v, wide_value);
    _mm_stream_ps(v + 4, wide_value);
    _mm_stream_ps(v + 8, wide_value);
    _mm_stream_ps(v + 12, wide_value);
    _mm_stream_ps(v + 16, wide_value);
    _mm_stream_ps(v + 20, wide_value);
    _mm_stream_ps(v + 24, wide_value);
    _mm_stream_ps(v + 28, wide_value);
}

static void DivideVector8(float* const v, const __m128 wide_divisor) noexcept {
    auto v0 = _mm_load_ps(v);
    auto v1 = _mm_load_ps(v + 4);
    auto v2 = _mm_load_ps(v + 8);
    auto v3 = _mm_load_ps(v + 12);
    auto v4 = _mm_load_ps(v + 16);
    auto v5 = _mm_load_ps(v + 20);
    auto v6 = _mm_load_ps(v + 24);
    auto v7 = _mm_load_ps(v + 28);

    v0 = _mm_div_ps(v0, wide_divisor);
    v1 = _mm_div_ps(v1, wide_divisor);
    v2 = _mm_div_ps(v2, wide_divisor);
    v3 = _mm_div_ps(v3, wide_divisor);
    v4 = _mm_div_ps(v4, wide_divisor);
    v5 = _mm_div_ps(v5, wide_divisor);
    v6 = _mm_div_ps(v6, wide_divisor);
    v7 = _mm_div_ps(v7, wide_divisor);

    _mm_stream_ps(v, v0);
    _mm_stream_ps(v + 4, v1);
    _mm_stream_ps(v + 8, v2);
    _mm_stream_ps(v + 12, v3);
    _mm_stream_ps(v + 16, v4);
    _mm_stream_ps(v + 20, v5);
    _mm_stream_ps(v + 24, v6);
    _mm_stream_ps(v + 28, v7);
}

static void MultiplyVector8(float* const v, const __m128 wide_divisor) noexcept {
    auto v0 = _mm_load_ps(v);
    auto v1 = _mm_load_ps(v + 4);
    auto v2 = _mm_load_ps(v + 8);
    auto v3 = _mm_load_ps(v + 12);
    auto v4 = _mm_load_ps(v + 16);
    auto v5 = _mm_load_ps(v + 20);
    auto v6 = _mm_load_ps(v + 24);
    auto v7 = _mm_load_ps(v + 28);

    v0 = _mm_mul_ps(v0, wide_divisor);
    v1 = _mm_mul_ps(v1, wide_divisor);
    v2 = _mm_mul_ps(v2, wide_divisor);
    v3 = _mm_mul_ps(v3, wide_divisor);
    v4 = _mm_mul_ps(v4, wide_divisor);
    v5 = _mm_mul_ps(v5, wide_divisor);
    v6 = _mm_mul_ps(v6, wide_divisor);
    v7 = _mm_mul_ps(v7, wide_divisor);

    _mm_stream_ps(v, v0);
    _mm_stream_ps(v + 4, v1);
    _mm_stream_ps(v + 8, v2);
    _mm_stream_ps(v + 12, v3);
    _mm_stream_ps(v + 16, v4);
    _mm_stream_ps(v + 20, v5);
    _mm_stream_ps(v + 24, v6);
    _mm_stream_ps(v + 28, v7);
}

static void AddVector8(float* const v, const float* const u) noexcept {
    auto v0 = _mm_load_ps(v);
    auto v1 = _mm_load_ps(v + 4);
    auto v2 = _mm_load_ps(v + 8);
    auto v3 = _mm_load_ps(v + 12);
    auto v4 = _mm_load_ps(v + 16);
    auto v5 = _mm_load_ps(v + 20);
    auto v6 = _mm_load_ps(v + 24);
    auto v7 = _mm_load_ps(v + 28);

    auto u0 = _mm_load_ps(u);
    auto u1 = _mm_load_ps(u + 4);
    auto u2 = _mm_load_ps(u + 8);
    auto u3 = _mm_load_ps(u + 12);
    auto u4 = _mm_load_ps(u + 16);
    auto u5 = _mm_load_ps(u + 20);
    auto u6 = _mm_load_ps(u + 24);
    auto u7 = _mm_load_ps(u + 28);

    v0 = _mm_mul_ps(v0, u0);
    v1 = _mm_mul_ps(v1, u1);
    v2 = _mm_mul_ps(v2, u2);
    v3 = _mm_mul_ps(v3, u3);
    v4 = _mm_mul_ps(v4, u4);
    v5 = _mm_mul_ps(v5, u5);
    v6 = _mm_mul_ps(v6, u6);
    v7 = _mm_mul_ps(v7, u7);

    _mm_stream_ps(v, v0);
    _mm_stream_ps(v + 4, v1);
    _mm_stream_ps(v + 8, v2);
    _mm_stream_ps(v + 12, v3);
    _mm_stream_ps(v + 16, v4);
    _mm_stream_ps(v + 20, v5);
    _mm_stream_ps(v + 24, v6);
    _mm_stream_ps(v + 28, v7);
}

static void AddVector8(float* const v, const float* const u, const __m128 wide_multiple) noexcept {
    auto v0 = _mm_load_ps(v);
    auto v1 = _mm_load_ps(v + 4);
    auto v2 = _mm_load_ps(v + 8);
    auto v3 = _mm_load_ps(v + 12);
    auto v4 = _mm_load_ps(v + 16);
    auto v5 = _mm_load_ps(v + 20);
    auto v6 = _mm_load_ps(v + 24);
    auto v7 = _mm_load_ps(v + 28);

    auto u0 = _mm_load_ps(u);
    auto u1 = _mm_load_ps(u + 4);
    auto u2 = _mm_load_ps(u + 8);
    auto u3 = _mm_load_ps(u + 12);
    auto u4 = _mm_load_ps(u + 16);
    auto u5 = _mm_load_ps(u + 20);
    auto u6 = _mm_load_ps(u + 24);
    auto u7 = _mm_load_ps(u + 28);

    u0 = _mm_mul_ps(u0, wide_multiple);
    u1 = _mm_mul_ps(u1, wide_multiple);
    u2 = _mm_mul_ps(u2, wide_multiple);
    u3 = _mm_mul_ps(u3, wide_multiple);
    u4 = _mm_mul_ps(u4, wide_multiple);
    u5 = _mm_mul_ps(u5, wide_multiple);
    u6 = _mm_mul_ps(u6, wide_multiple);
    u7 = _mm_mul_ps(u7, wide_multiple);

    v0 = _mm_mul_ps(v0, u0);
    v1 = _mm_mul_ps(v1, u1);
    v2 = _mm_mul_ps(v2, u2);
    v3 = _mm_mul_ps(v3, u3);
    v4 = _mm_mul_ps(v4, u4);
    v5 = _mm_mul_ps(v5, u5);
    v6 = _mm_mul_ps(v6, u6);
    v7 = _mm_mul_ps(v7, u7);

    _mm_stream_ps(v, v0);
    _mm_stream_ps(v + 4, v1);
    _mm_stream_ps(v + 8, v2);
    _mm_stream_ps(v + 12, v3);
    _mm_stream_ps(v + 16, v4);
    _mm_stream_ps(v + 20, v5);
    _mm_stream_ps(v + 24, v6);
    _mm_stream_ps(v + 28, v7);
}

static __m128 ScalarProduct8(const float* const v, const float* const u) noexcept {
    auto v0 = _mm_load_ps(v);
    auto v1 = _mm_load_ps(v + 4);
    auto v2 = _mm_load_ps(v + 8);
    auto v3 = _mm_load_ps(v + 12);
    auto v4 = _mm_load_ps(v + 16);
    auto v5 = _mm_load_ps(v + 20);
    auto v6 = _mm_load_ps(v + 24);
    auto v7 = _mm_load_ps(v + 28);

    auto u0 = _mm_load_ps(u);
    auto u1 = _mm_load_ps(u + 4);
    auto u2 = _mm_load_ps(u + 8);
    auto u3 = _mm_load_ps(u + 12);
    auto u4 = _mm_load_ps(u + 16);
    auto u5 = _mm_load_ps(u + 20);
    auto u6 = _mm_load_ps(u + 24);
    auto u7 = _mm_load_ps(u + 28);

    v0 = _mm_mul_ps(v0, u0);
    v1 = _mm_mul_ps(v1, u1);
    v2 = _mm_mul_ps(v2, u2);
    v3 = _mm_mul_ps(v3, u3);
    v4 = _mm_mul_ps(v4, u4);
    v5 = _mm_mul_ps(v5, u5);
    v6 = _mm_mul_ps(v6, u6);
    v7 = _mm_mul_ps(v7, u7);

    v0 = _mm_add_ps(v0, v4);
    v1 = _mm_add_ps(v1, v5);
    v2 = _mm_add_ps(v2, v6);
    v3 = _mm_add_ps(v3, v7);

    v0 = _mm_add_ps(v0, v2);
    v1 = _mm_add_ps(v1, v3);

    return _mm_add_ps(v0, v1);
}

void yzw2v::num::Fill(float* v, const uint32_t v_size, const float value) noexcept {
    const auto* const v_end = v + v_size;
    const auto wide_value = _mm_set1_ps(value);
    for (const auto* const v_this_end = v_end - 31; v < v_this_end; v += 32) {
        Prefetch8(v);
        Fill8(v, wide_value);
    }

    for (const auto* const v_this_end = v_end - 3; v < v_this_end; v += 4) {
        _mm_prefetch(v + 4, _MM_HINT_T0);
        _mm_stream_ps(v, wide_value);
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
    for (const auto* const v_this_end = v_end - 31; v < v_this_end; v += 32) {
        Prefetch8(v);
        DivideVector8(v, wide_divisor);
    }

    for (const auto* const v_this_end = v_end - 3; v < v_this_end; v += 4) {
        _mm_prefetch(v + 4, _MM_HINT_T0);
        _mm_stream_ps(v, _mm_div_ps(_mm_load_ps(v), wide_divisor));
    }

    for (; v < v_end;) {
        *(v++) /= divisor;
    }
}

void yzw2v::num::MultiplyVector(float* v, const uint32_t v_size, const float multiple) noexcept {
    const auto* const v_end = v + v_size;
    const auto wide_multiple = _mm_set1_ps(multiple);

    for (const auto* const v_this_end = v_end - 31; v < v_this_end; v += 32) {
        Prefetch8(v);
        MultiplyVector8(v, wide_multiple);
    }

    for (const auto* const v_this_end = v_end - 3; v < v_this_end; v += 4) {
        _mm_prefetch(v + 4, _MM_HINT_T0);
        _mm_stream_ps(v, _mm_mul_ps(_mm_load_ps(v), wide_multiple));
    }

    for (; v < v_end; ++v) {
        *(v++) *= multiple;
    }
}

void yzw2v::num::AddVector(float* v, const uint32_t v_size, const float* summand) noexcept {
    const auto* const v_end = v + v_size;
    for (const auto* const v_this_end = v_end - 31; v < v_this_end; summand += 32, v += 32) {
        Prefetch8(v);
        Prefetch8(summand);
        AddVector8(v, summand);
    }

    for (const auto* const v_this_end = v_end - 3; v < v_this_end; summand += 4, v += 4) {
        _mm_prefetch(v + 4, _MM_HINT_T0);
        _mm_stream_ps(v, _mm_add_ps(_mm_load_ps(v), _mm_load_ps(summand)));
    }

    for (; v < v_end;) {
        *(v++) += *(summand++);
    }
}

void yzw2v::num::AddVector(float* v, const uint32_t v_size,
                           const float* summand, const float summand_multiple) noexcept {
    const auto* const v_end = v + v_size;
    const auto wide_summand_multiple = _mm_set1_ps(summand_multiple);
    for (const auto* const v_this_end = v_end - 31; v < v_this_end; summand += 32, v += 32) {
        Prefetch8(v);
        Prefetch8(summand);
        AddVector8(v, summand, wide_summand_multiple);
    }

    for (const auto* const v_this_end = v_end - 3; v < v_this_end; summand += 4, v += 4) {
        _mm_prefetch(v + 4, _MM_HINT_T0);
        _mm_prefetch(summand + 4, _MM_HINT_T0);
        _mm_stream_ps(v, _mm_add_ps(_mm_load_ps(v),
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
    for (const auto* const v_this_end = v_end - 31; v < v_this_end; v += 32, rhs += 32) {
        Prefetch8(v);
        Prefetch8(rhs);
        const auto cur_res = ScalarProduct8(v, rhs);
        wide_res = _mm_add_ps(wide_res, cur_res);
    }

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
