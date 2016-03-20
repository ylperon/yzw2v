#include "unigram_distribution.h"

#include "vocabulary.h"

#include <cmath>

uint32_t yzw2v::train::UnigramDistribution::size() const noexcept {
    return table_size_;
}

uint32_t yzw2v::train::UnigramDistribution::operator[](const uint32_t index) const noexcept {
    return table_[index];
}

void yzw2v::train::UnigramDistribution::Init(const yzw2v::vocab::Vocabulary& vocab) noexcept {
    const auto POWER = 0.75;
    const auto train_words_pow = [&vocab, POWER]{
        auto sum = double{};
        for (auto id = uint32_t{}; id < vocab.size(); ++id) {
            sum += std::pow(static_cast<double>(vocab.Count(id)), POWER);
        }

        return sum;
    }();

    auto* const table = table_holder_.get();
    auto id = uint32_t{};
    auto d1 = std::pow(static_cast<double>(vocab.Count(id)), POWER) / train_words_pow;
    for (auto index = uint32_t{}; index < table_size_; ++index) {
        table[index] = id;
        if (static_cast<double>(index) / table_size_ > d1) {
            ++id;
            if (id >= vocab.size()) {
                id = vocab.size() - 1;
            }

            d1 += std::pow(static_cast<double>(vocab.Count(id)), POWER) / train_words_pow;
        }
    }
}

