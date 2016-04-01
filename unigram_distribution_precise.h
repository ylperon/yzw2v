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
            uint32_t next(const PRNG& prng) const noexcept;

            void prefetch(const PRNG& prng) const noexcept;
            void prefetch(const PRNG& prng, const uint32_t steps) const noexcept;

        private:
            struct Entry {
                float prob;
                uint32_t alias;
            };

            uint32_t size_;
            Entry* table_;
            std::unique_ptr<Entry[]> table_holder_;
        };
    }  // namespace sampling
}  // namespace yzw2v
