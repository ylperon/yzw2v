#include "mem.h"

#include <stdexcept>

#include <cstdlib>
#include <cstring>

void yzw2v::mem::detail::Deleter::operator ()(void* const ptr) const noexcept {
    std::free(ptr);
}

std::unique_ptr<float, yzw2v::mem::detail::Deleter>
yzw2v::mem::AllocateFloatForSIMD(const uint32_t size) {
    auto* res = static_cast<float*>(nullptr);
    const auto actual_size = RoundSizeUpByVecSize(size);
    const auto ret = posix_memalign(reinterpret_cast<void**>(&res), sizeof(float) * VEC_SIZE, sizeof(float) * actual_size);
    if (ret) {
        throw std::runtime_error{"aligned allocation failed"};
    }

    std::memset(res, 0, sizeof(float) * actual_size);

    return std::unique_ptr<float, yzw2v::mem::detail::Deleter>{res};
}
