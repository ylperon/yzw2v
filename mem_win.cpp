#include "mem.h"

#include <stdexcept>

#include <cstdlib>
#include <cstring>

#include <malloc.h>

void yzw2v::mem::detail::Deleter::operator ()(void* const ptr) const noexcept {
    _aligned_free(ptr);
}

std::unique_ptr<float, yzw2v::mem::detail::Deleter>
yzw2v::mem::AllocateFloatForSIMD(const uint32_t size) {
    const auto actual_size = RoundSizeUpByVecSize(size);
    auto* const res = reinterpret_cast<float*>(_aligned_malloc(sizeof(float) * actual_size, 128));
    if (!res) {
        std::runtime_error{"aligned allocation failed"};
    }

    std::memset(res, 0, sizeof(float) * actual_size);

    return std::unique_ptr<float, yzw2v::mem::detail::Deleter>{res};
}
