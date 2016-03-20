#pragma once

#include "likely.h"

#include <algorithm>
#include <memory>

#include <cinttypes>

namespace yzw2v {
    namespace vocab {
        class Vocabulary;
    }
}

namespace yzw2v {
    namespace train {
        class UnigramDistribution {
        public:
            template <typename PRNG>
            UnigramDistribution(const uint32_t size, const yzw2v::vocab::Vocabulary& vocab,
                                PRNG&& prng);

            uint32_t size() const noexcept;
            uint32_t operator[](const uint32_t index) const noexcept;

        private:
            void Init(const yzw2v::vocab::Vocabulary& vocab) noexcept;

            uint32_t table_size_;
            const uint32_t* table_;
            std::unique_ptr<uint32_t[]> table_holder_;
        };
    }
}

template <typename PRNG>
yzw2v::train::UnigramDistribution::UnigramDistribution(const uint32_t size,
                                                       const yzw2v::vocab::Vocabulary& vocab,
                                                       PRNG&& prng)
    : table_size_{size}
    , table_holder_{new uint32_t[size]}
{
    table_ = table_holder_.get();
    Init(vocab);
    std::shuffle(table_holder_.get(), table_holder_.get() + table_size_, prng);
}
