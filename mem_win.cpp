#include "mem.h"

#include <cstdlib>

#include <malloc.h>

void yzw2v::mem::detail::Deleter::operator ()(void* const ptr) const noexcept {
    _aligned_free(ptr);
}

std::unique_ptr<float, yzw2v::mem::detail::Deleter>
yzw2v::mem::AllocateFloatForSIMD(const uint32_t size) {
    auto* const res = reinterpret_cast<float*>(_aligned_malloc(sizeof(float) * size, 128));
    if (!res) {
        std::runtime_error{"aligned allocation failed"};
    }

    return std::unique_ptr<float, yzw2v::mem::detail::Deleter>{res};
}
