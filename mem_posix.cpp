#include "mem.h"

#include <stdexcept>
#include <cstdlib>

void yzw2v::mem::detail::Deleter::operator ()(void* const ptr) const noexcept {
    std::free(ptr);
}

std::unique_ptr<float, yzw2v::mem::detail::Deleter>
yzw2v::mem::AllocateFloatForSIMD(const uint32_t size) {
    auto* res = static_cast<float*>(nullptr);
    const auto ret = posix_memalign(reinterpret_cast<void**>(&res), 128, sizeof(float) * size);
    if (ret) {
        throw std::runtime_error{"aligned allocation failed"};
    }

    return std::unique_ptr<float, yzw2v::mem::detail::Deleter>{res};
}
