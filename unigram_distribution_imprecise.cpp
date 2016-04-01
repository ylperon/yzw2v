#include "unigram_distribution_imprecise.h"

#include "prefetch.h"
#include "prng.h"
#include "vocabulary.h"

#include <cmath>

static constexpr uint32_t UNIGRAM_TABLE_SIZE = 100000000;

yzw2v::sampling::UnigramDistribution::UnigramDistribution(const vocab::Vocabulary& vocab)
    : size_{UNIGRAM_TABLE_SIZE}
    , vocab_size_{vocab.size()}
    , table_holder_{new uint32_t[UNIGRAM_TABLE_SIZE]}
{
    table_ = table_holder_.get();

    constexpr auto POWER = 0.75;
    const auto train_words_pow = [&vocab, POWER]{
        auto sum = double{};
        for (auto id = uint32_t{}; id < vocab.size(); ++id) {
            sum += std::pow(static_cast<double>(vocab.Count(id)), POWER);
        }

        return sum;
    }();

    auto id = uint32_t{};
    auto d1 = std::pow(static_cast<double>(vocab.Count(id)), POWER) / train_words_pow;
    for (auto index = uint32_t{}; index < size_; ++index) {
        table_[index] = id;
        if (static_cast<double>(index) / size_ > d1) {
            ++id;
            if (id >= vocab.size()) {
                id = vocab.size() - 1;
            }

            d1 += std::pow(static_cast<double>(vocab.Count(id)), POWER) / train_words_pow;
        }
    }
}

uint32_t yzw2v::sampling::UnigramDistribution::operator() (PRNG& prng) const noexcept {
    const auto prn = prng();
    if (const auto val = table_[prn % size_]) {
        return val;
    }

    return (prn % (vocab_size_ - 1)) + 1;
}

uint32_t yzw2v::sampling::UnigramDistribution::next(const PRNG& prng) const noexcept {
    const auto prn = prng.next();
    if (const auto val = table_[prn % size_]) {
        return val;
    }

    return (prn % (vocab_size_ - 1)) + 1;
}

void yzw2v::sampling::UnigramDistribution::prefetch(const PRNG& prng) const noexcept {
    YZ_PREFETCH_READ(table_ + (prng.next() % size_), 3);
}

void yzw2v::sampling::UnigramDistribution::prefetch(const PRNG& prng, const uint32_t steps) const noexcept {
    YZ_PREFETCH_READ(table_ + (prng.next(steps) % size_), 3);
}
