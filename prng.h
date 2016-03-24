#pragma once

#include <limits>

#include <cstdint>

namespace yzw2v {
    namespace sampling {
        class PRNG {
        public:
            using result_type = uint64_t;

            explicit PRNG(const uint64_t state)
                : state_{state}
            {
            }

            uint64_t next() const noexcept {
                return state_ * uint64_t{25214903917} + uint64_t{11};
            }

            uint64_t operator()() noexcept {
                return state_ = next();
            }

            double real_0_inc_1_inc() noexcept {
                return (operator()() >> 11) * (1.0 / 9007199254740991.0);
            }

            double real_0_inc_1_exc() noexcept {
                return (operator()() >> 11) * (1.0 / 9007199254740992.0);
            }

            static constexpr uint64_t min() noexcept {
                return 0;
            }

            static constexpr uint64_t max() noexcept {
                return std::numeric_limits<uint64_t>::max();
            }

            void discard(uint64_t n) noexcept {
                for (; n; --n) {
                    // may be done in a more efficient manner
                    operator()();
                }
            }

        private:
            uint64_t state_;
        };
    }
}
