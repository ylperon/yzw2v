#pragma once

#include <memory>

#include <cstdint>

namespace yzw2v {
    namespace mem {
        namespace detail {
            class Deleter {
            public:
                void operator() (void* const ptr) const noexcept;
            };
        }

        extern const uint32_t VEC_SIZE;

        uint32_t RoundSizeUpByVecSize(const uint32_t size) noexcept;

        std::unique_ptr<float, detail::Deleter> AllocateFloatForSIMD(const uint32_t size);
    }
}
