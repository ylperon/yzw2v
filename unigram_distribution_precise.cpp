#include "unigram_distribution_precise.h"

#include "vocabulary.h"
#include "prng.h"

#include <vector>
#include <cmath>

namespace {
    template <typename T>
    class KahanAccumulator {
    public:
        using value_type = T;

        explicit KahanAccumulator(const value_type value = {})
            : sum_{value}
            , compensation_{} {
        }

        template <typename Other>
        KahanAccumulator& operator =(const Other value) {
            sum_ = static_cast<value_type>(value);
            compensation_ = {};
            return *this;
        }

        template <typename Other>
        KahanAccumulator& operator+= (const Other value) {
            const value_type xxx = static_cast<value_type>(value) - compensation_;
            const value_type yyy = sum_ + xxx;
            compensation_ = (yyy - sum_) - xxx;
            sum_ = yyy;
            return *this;
        }

        value_type get() const {
            return sum_ + compensation_;
        }

        template <typename Other>
        operator Other() const {
            return get();
        }

    private:
        value_type sum_;
        value_type compensation_;
    };
}

namespace {
    struct PreciseEntry {
        double prob;
        uint32_t alias;
    };
}

static std::vector<PreciseEntry> GenerateTable(const yzw2v::vocab::Vocabulary& vocab) {
    constexpr auto POWER = 0.75;
    const auto sum = [&vocab, POWER]{
        auto res = KahanAccumulator<double>{};
        for (auto i = uint32_t{1}; i < vocab.size(); ++i) {
            res += vocab.Count(i);
        }

        return res.get();
    }();

    const auto size = vocab.size() - 1;
    auto table = std::vector<PreciseEntry>(size);
    for (auto i = uint32_t{1}; i < vocab.size() - 1; ++i) {
        table[i - 1].prob = vocab.Count(i) / sum;
    }

    auto small = std::vector<uint32_t>{};
    small.reserve(size);
    auto large = std::vector<uint32_t>{};
    large.reserve(size);
    for (auto i = uint32_t{0}; i < size; ++i) {
        const auto value = table[i].prob * size;
        if (value < 1.0) {
            small.push_back(i);
        } else {
            large.push_back(i);
        }
    }

    while (!small.empty() && !large.empty()) {
        const auto small_index = small.back();
        const auto large_index = large.back();
        small.pop_back();
        large.pop_back();
        table[small_index].alias = large_index;
        const auto large_prob_updated = (table[small_index].prob + table[large_index].prob) - 1.0;
        table[large_index].prob = large_prob_updated;
        if (large_prob_updated < 1.0) {
            small.push_back(large_index);
        } else {
            large.push_back(small_index);
        }
    }

    while (!large.empty()) {
        const auto index = large.back();
        large.pop_back();
        table[index].prob = 1.0;
    }

    while (!small.empty()) {
        const auto index = small.back();
        small.pop_back();
        table[index].prob = 1.0;
    }

    return table;
}

yzw2v::sampling::UnigramDistribution::UnigramDistribution(const vocab::Vocabulary& vocab)
    : size_{vocab.size() - 1}
    , table_holder_{new Entry[vocab.size() - 1]}  // ignore paragraph token
{
    table_ = table_holder_.get();
    const auto precise_table = GenerateTable(vocab);
    for (auto i = uint32_t{}; i < size_; ++i) {
        table_[i].prob = static_cast<float>(precise_table[i].prob);
        table_[i].alias = precise_table[i].alias;
    }
}

uint32_t yzw2v::sampling::UnigramDistribution::operator() (PRNG& prng) const noexcept {
    const auto index = static_cast<uint32_t>(size_ * prng.real_0_inc_1_exc());
    const auto prob = static_cast<float>(prng.real_0_inc_1_inc());
    if (prob < table_[index].prob) {
        return index + 1;
    }

    return table_[index].alias + 1;
}

const uint32_t* yzw2v::sampling::UnigramDistribution::nexp_ptr(const PRNG& prng) const noexcept {
    return nullptr;
}
