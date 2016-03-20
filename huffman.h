#pragma once

#include "pool.h"

#include <vector>
#include <memory>

#include <cstdint>

namespace yzw2v {
    namespace vocab {
        class Vocabulary;
    }
}

namespace yzw2v {
    namespace huff {
        struct Token {
            uint32_t* point{nullptr};
            uint8_t* code{nullptr};
            uint32_t length{0};
        };

        class HuffmanTree {
        public:
            HuffmanTree(const ::yzw2v::vocab::Vocabulary& vocab);

            const std::vector<Token>& Tokens() const noexcept;

        private:
            std::vector<Token> tokens_;
            mem::Pool points_pool_;
            mem::Pool code_pool_;
        };
    }  // namespace huff
}  // namespace yzw2v
