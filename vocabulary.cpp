#include "vocabulary.h"
#include "likely.h"

#include <algorithm>
#include <limits>

#include <cassert>
#include <cstring>

yzw2v::vocab::Token::Token(const char* const begin) noexcept
    : begin_{begin}
    , length_{static_cast<uint8_t>(std::strlen(begin))} {
}

yzw2v::vocab::Token::Token(const char* const begin, const char* const end) noexcept
    : begin_{begin}
    , length_{static_cast<uint8_t>(end - begin)} {
}

yzw2v::vocab::Token::Token(const char* const begin, const uint8_t length) noexcept
    : begin_{begin}
    , length_{length} {
}

bool yzw2v::vocab::Token::operator==(const Token& other) const noexcept {
    if (YZ_LIKELY(length_ != other.length_)) {
        return false;
    }

    return 0 == std::strncmp(begin_, other.begin_, length_);
}

bool yzw2v::vocab::Token::operator!=(const Token& other) const noexcept {
    if (YZ_LIKELY(length_ != other.length_)) {
        return true;
    }

    return 0 != std::strncmp(begin_, other.begin_, length_);
}

bool yzw2v::vocab::Token::operator<(const Token& other) const noexcept {
    return std::strncmp(begin_, other.begin_, std::max(length_, other.length_)) < 0;
}

bool yzw2v::vocab::Token::operator<=(const Token& other) const noexcept {
    return std::strncmp(begin_, other.begin_, std::max(length_, other.length_)) <= 0;
}

bool yzw2v::vocab::Token::operator>(const Token& other) const noexcept {
    return std::strncmp(begin_, other.begin_, std::max(length_, other.length_)) > 0;
}

bool yzw2v::vocab::Token::operator>=(const Token& other) const noexcept {
    return std::strncmp(begin_, other.begin_, std::max(length_, other.length_)) >= 0;
}

const char* yzw2v::vocab::Token::cbegin() const noexcept {
    return begin_;
}

const char* yzw2v::vocab::Token::cend() const noexcept {
    return begin_ + length_;
}

const char* yzw2v::vocab::Token::begin() const noexcept {
    return begin_;
}

const char* yzw2v::vocab::Token::end() const noexcept {
    return begin_ + length_;
}

uint8_t yzw2v::vocab::Token::length() const noexcept {
    return length_;
}

yzw2v::vocab::TokenInfo::TokenInfo(const Token& token_, const uint32_t count_) noexcept
    : token{token_}
    , count{count_} {
}

static constexpr size_t BLOCK_SIZE = 1024 * 1024 * 8; // 8 Mb

yzw2v::vocab::Vocabulary::Vocabulary(const uint32_t max_number_of_tokens)
    : max_number_of_tokens_{max_number_of_tokens}
    , hash_table_size_{max_number_of_tokens * 10 / 7}
    , pool_(BLOCK_SIZE)
    , hash_(hash_table_size_, INVALID_TOKEN_ID) {
    tokens_.reserve(max_number_of_tokens_);
}

static uint32_t Hash(const yzw2v::vocab::Token& token) {
    // let's trust guys from MSR http://stackoverflow.com/a/107657/2513489
    // perf doesn't show any real difference, but we have to be different from word2vec
    // implementation :)
    static constexpr uint32_t BASE = 101;

    auto hash = uint32_t{};
    const auto* p = token.cbegin();
    const auto* const p_end = token.cend();

#if 1
    static constexpr uint32_t BASE_P[] = {
        1,
        BASE,
        BASE * BASE,
        BASE * BASE * BASE,
        BASE * BASE * BASE * BASE
    };
    for (const auto* const p_this_end = p_end - 3; p < p_this_end; p += 4) {
        hash = hash * BASE_P[4]
               + BASE_P[3] * static_cast<uint8_t>(p[0])
               + BASE_P[2] * static_cast<uint8_t>(p[1])
               + BASE_P[1] * static_cast<uint8_t>(p[2])
               + BASE_P[0] * static_cast<uint8_t>(p[3]);
    }
#endif

    for (; p < p_end; ++p) {
        hash = hash * BASE + static_cast<uint8_t>(*p);
    }

    return hash;
}

uint32_t yzw2v::vocab::Vocabulary::size() const noexcept {
    return static_cast<uint32_t>(tokens_.size());
}

float yzw2v::vocab::Vocabulary::LoadFactor() const noexcept {
    return static_cast<float>(static_cast<double>(tokens_.size()) / hash_table_size_);
}

uint64_t yzw2v::vocab::Vocabulary::TextWordCount() const noexcept {
    auto count = uint64_t{};
    for (const auto& token : tokens_) {
        count += token.count;
    }

    return count;
}

bool yzw2v::vocab::Vocabulary::Has(const class Token& token) const noexcept {
    return INVALID_TOKEN_ID == ID(token);
}

bool yzw2v::vocab::Vocabulary::Has(const uint32_t id) const noexcept {
    return INVALID_TOKEN_ID != id && id < tokens_.size();
}

const yzw2v::vocab::TokenInfo& yzw2v::vocab::Vocabulary::Token(const uint32_t id) const noexcept {
    return tokens_[id];
}

uint32_t yzw2v::vocab::Vocabulary::Count(const uint32_t id) const noexcept {
    return tokens_[id].count;
}

uint32_t yzw2v::vocab::Vocabulary::ID(const yzw2v::vocab::Token& token) const noexcept {
    auto hash = Hash(token) % hash_table_size_;
    while (INVALID_TOKEN_ID != hash_[hash]) {
        if (token == tokens_[hash_[hash]].token) {
            return hash_[hash];
        }

        hash = (hash + 1) % hash_table_size_;
    }

    return INVALID_TOKEN_ID;
}

yzw2v::vocab::Vocabulary::const_iterator yzw2v::vocab::Vocabulary::cbegin() noexcept {
    return tokens_.cbegin();
}

yzw2v::vocab::Vocabulary::const_iterator yzw2v::vocab::Vocabulary::cend() noexcept {
    return tokens_.cend();
}

yzw2v::vocab::Vocabulary::const_reverse_iterator yzw2v::vocab::Vocabulary::crbegin() noexcept {
    return tokens_.crbegin();
}

yzw2v::vocab::Vocabulary::const_reverse_iterator yzw2v::vocab::Vocabulary::crend() noexcept {
    return tokens_.crend();
}

static yzw2v::vocab::Token
Copy(const yzw2v::vocab::Token& token, yzw2v::mem::Pool& pool) noexcept {
    auto* const begin = pool.Get<char>(token.length());
    const auto res = ::yzw2v::vocab::Token{begin, token.length()};
    std::memmove(begin, token.cbegin(), token.length());

    return res;
}

uint32_t yzw2v::vocab::Vocabulary::Add(const class Token& token) {
    auto hash = Hash(token);
    if (YZ_UNLIKELY(hash >= hash_table_size_)) {
        hash %= hash_table_size_;
    }

    while (INVALID_TOKEN_ID != hash_[hash]) {
        if (token == tokens_[hash_[hash]].token) {
            ++tokens_[hash_[hash]].count;
            return hash_[hash];
        }

        if (YZ_UNLIKELY(++hash > hash_table_size_)) {
            hash %= hash_table_size_;
        }
    }

    const auto index = static_cast<uint32_t>(tokens_.size());
    hash_[hash] = index;
    tokens_.emplace_back(Copy(token, pool_), 1);

    return index;
}

void yzw2v::vocab::Vocabulary::Sort() noexcept {
    const auto cmp_less = [](const TokenInfo& lhs, const TokenInfo& rhs) noexcept -> bool {
        if (lhs.count > rhs.count) {
            return true;
        } else if (lhs.count < rhs.count) {
            return false;
        }

        return lhs.token < rhs.token;
    };

    if (tokens_.size() >= 1) {
        // PARAGRAPH_TOKEN should go first
        std::sort(tokens_.begin() + 1, tokens_.end(), std::cref(cmp_less));
    }

    hash_.assign(hash_.size(), INVALID_TOKEN_ID);
    for (auto it = tokens_.cbegin(); tokens_.cend() != it; ++it) {
        auto hash = Hash(it->token) % hash_table_size_;
        for (; INVALID_TOKEN_ID != hash_[hash]; hash = (hash + 1) % hash_table_size_);
        hash_[hash] = static_cast<uint32_t>(it - tokens_.cbegin());
    }
}
