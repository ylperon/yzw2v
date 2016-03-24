#pragma once

#include <memory>

namespace yzw2v {
    namespace sampling {
        class PRNG;
    }

    namespace vocab {
        class Vocabulary;
    }
}

namespace yzw2v {
    namespace sampling {
        class UnigramDistribution {
        public:
            explicit UnigramDistribution(const vocab::Vocabulary& vocab);
            uint32_t operator()(PRNG& prng) const noexcept;

            void prefetch(const PRNG& prng) const noexcept;
            void prefetch(const PRNG& prng, const uint32_t steps) const noexcept;

        private:
            uint32_t size_;
            uint32_t* table_;
            uint32_t vocab_size_;
            std::unique_ptr<uint32_t[]> table_holder_;
        };
    }  // namespace sampling
}  // namespace yzw2v
