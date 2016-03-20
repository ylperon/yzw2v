#pragma once

#include "pool.h"

#include <iosfwd>
#include <limits>
#include <list>
#include <memory>
#include <vector>

#include <cstdint>


namespace yzw2v {
    namespace vocab {
        static constexpr size_t MAX_TOKEN_LENGTH = 256;
        static constexpr uint32_t INVALID_TOKEN_ID = std::numeric_limits<uint32_t>::max();

        class Token {
        public:
            Token() noexcept = default;
            Token(const Token& other) noexcept = default;
            Token(Token&& other) noexcept = default;
            explicit Token(const char* const begin) noexcept;
            Token(const char* const begin, const char* const end) noexcept;
            Token(const char* const begin, const uint8_t lenght) noexcept;
            ~Token() noexcept = default;

            Token& operator=(const Token& other) noexcept = default;
            Token& operator=(Token&& other) noexcept = default;

            bool operator ==(const Token& other) const noexcept;
            bool operator !=(const Token& other) const noexcept;
            bool operator <(const Token& other) const noexcept;
            bool operator >(const Token& other) const noexcept;
            bool operator <=(const Token& other) const noexcept;
            bool operator >=(const Token& other) const noexcept;

            const char* cbegin() const noexcept;
            const char* cend() const noexcept;
            const char* begin() const noexcept;
            const char* end() const noexcept;
            uint8_t length() const noexcept;

        private:
            const char* begin_{nullptr};
            uint8_t length_{0};
        };

        static const yzw2v::vocab::Token PARAGRAPH_TOKEN{"</s>"};
        static const uint32_t PARAGRAPH_TOKEN_ID = 0;

        struct TokenInfo {
            Token token;
            uint32_t count;

            TokenInfo() noexcept = default;
            TokenInfo(const TokenInfo& other) noexcept = default;
            TokenInfo(TokenInfo&& other) noexcept = default;
            TokenInfo(const Token& token_, const uint32_t count_) noexcept;
            ~TokenInfo() noexcept = default;
            TokenInfo& operator=(const TokenInfo& other) noexcept = default;
            TokenInfo& operator=(TokenInfo&& other) noexcept = default;
        };

        class Vocabulary {
        public:
            using const_iterator = std::vector<TokenInfo>::const_iterator;
            using const_reverse_iterator = std::vector<TokenInfo>::const_reverse_iterator;

            explicit Vocabulary(const uint32_t max_number_of_tokens);

            uint32_t Add(const Token& token);

            bool Has(const Token& token) const noexcept;
            uint32_t ID(const Token& token) const noexcept;

            bool Has(const uint32_t id) const noexcept;
            const TokenInfo& Token(const uint32_t id) const noexcept;
            uint32_t Count(const uint32_t id) const noexcept;

            uint32_t size() const noexcept;
            float LoadFactor() const noexcept;
            uint64_t TextWordCount() const noexcept;

            void Sort() noexcept;

            const_iterator cbegin() noexcept;
            const_iterator cend() noexcept;

            const_reverse_iterator crbegin() noexcept;
            const_reverse_iterator crend() noexcept;

        private:
            uint32_t max_number_of_tokens_;
            uint32_t hash_table_size_;
            mem::Pool pool_;
            std::vector<uint32_t> hash_;
            std::vector<TokenInfo> tokens_;

        public:
            static void WriteTSVWithFilter(const Vocabulary& vocab, const std::string& path,
                                           const uint32_t min_token_freq);
            static void WriteBinaryWithFilter(const Vocabulary& vocab, const std::string& path,
                                              const uint32_t min_token_freq);

            static void ReadBinaryWithFilter(const std::string& path, const uint32_t min_token_freq,
                                             Vocabulary& vocab);
        };

        void CollectIntoVocabulary(const std::string& path, const uint32_t min_token_freq,
                                   Vocabulary& vocab);
        Vocabulary CollectVocabulary(const std::string& path, const uint32_t min_token_freq,
                                     const uint32_t max_number_of_tokens);

        void WriteTSV(const Vocabulary& vocab, const std::string& path);
        void WriteTSVWithFilter(const Vocabulary& vocab, const std::string& path,
                                const uint32_t min_token_freq);

        void WriteBinary(const Vocabulary& vocab, const std::string& path);
        void WriteBinaryWithFilter(const Vocabulary& vocab, const std::string& path,
                                   const uint32_t min_token_freq);

        Vocabulary ReadBinary(const std::string& path);
        void ReadBinary(const std::string& path, Vocabulary& vocab);
        Vocabulary ReadBinaryWithFilter(const std::string& path, const uint32_t min_token_freq);
        void ReadBinaryWithFilter(const std::string& path, const uint32_t min_token_freq,
                                  Vocabulary& vocab);
    }  // namespace vocab
} // namespace yzw2v

std::ostream& operator<<(std::ostream& out, const yzw2v::vocab::Token& token);
