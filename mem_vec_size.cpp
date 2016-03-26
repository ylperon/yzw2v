#include "mem.h"

#if defined(__AVX__)
const uint32_t yzw2v::mem::VEC_SIZE = 8;
#elif defined(__SSE__)
const uint32_t yzw2v::mem::VEC_SIZE = 4;
#else
const uint32_t yzw2v::mem::VEC_SIZE = 1;
#endif

uint32_t yzw2v::mem::RoundSizeUpByVecSize(const uint32_t size) noexcept {
    if (const auto remainder = size % VEC_SIZE) {
        return size + (VEC_SIZE - remainder);
    }

    return size;
}
