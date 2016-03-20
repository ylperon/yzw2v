#include "huffman.h"
#include "vocabulary.h"

#include <limits>

static constexpr size_t ELEMENTS_PER_BLOCK = 1024 * 1024 * 8;
static constexpr size_t POINTS_BLOCK_SIZE = sizeof(decltype(*yzw2v::huff::Token::point)) * ELEMENTS_PER_BLOCK;
static constexpr size_t CODES_BLOCK_SIZE = sizeof(decltype(*yzw2v::huff::Token::code)) * ELEMENTS_PER_BLOCK;
static constexpr size_t MAX_CODE_LENGTH = 100;

static void CreateHuffmanTree(const ::yzw2v::vocab::Vocabulary& vocab,
                              std::vector<::yzw2v::huff::Token>& tokens,
                              yzw2v::mem::Pool& point_pool,
                              yzw2v::mem::Pool& code_pool) {
    auto count = std::vector<uint32_t>(vocab.size() * 2 + 1,
                                       std::numeric_limits<uint32_t>::max() / 2);
    auto binary = std::vector<uint8_t>(vocab.size() * 2 + 1);
    auto parent_node = std::vector<uint32_t>(vocab.size() * 2 + 1);
    auto code = std::vector<uint8_t>(MAX_CODE_LENGTH);
    auto point = std::vector<uint32_t>(MAX_CODE_LENGTH);

    const auto vocab_size = vocab.size();
    for (auto index = uint32_t{}; index < vocab_size; ++index) {
        count[index] = vocab.Token(index).count;
    }

    auto pos1 = vocab_size - 1;
    auto pos2 = vocab_size;
    auto min1i = uint32_t{};
    auto min2i = uint32_t{};
    for (auto index = uint32_t{}; index < vocab_size - 1; ++index) {
        if (std::numeric_limits<uint32_t>::max() != pos1) {
            if (count[pos1] < count[pos2]) {
                min1i = pos1;
                --pos1;
            } else {
                min1i = pos2;
                ++pos2;
            }
        } else {
            min1i = pos2;
            ++pos2;
        }

        if (std::numeric_limits<uint32_t>::max() != pos1) {
            if (count[pos1] < count[pos2]) {
                min2i = pos1;
                --pos1;
            } else {
                min2i = pos2;
                ++pos2;
            }
        } else {
            min2i = pos2;
            ++pos2;
        }

        count[vocab_size + index] = count[min1i] + count[min2i];
        parent_node[min1i] = vocab_size + index;
        parent_node[min2i] = vocab_size + index;
        binary[min2i] = 1;
    }

    for (auto index = uint32_t{}; index < vocab_size; ++index){
        auto jindex = index;
        auto code_length = uint32_t{};
        while (true) {
            code[code_length] = binary[jindex];
            point[code_length] = jindex;
            ++code_length;
            jindex = parent_node[jindex];
            if (jindex == vocab_size * 2 - 2) {
                break;
            }
        }

        tokens[index].length = code_length;
        tokens[index].code = code_pool.Get<uint8_t>(code_length);
        tokens[index].point = point_pool.Get<uint32_t>(code_length);
        tokens[index].point[0] = vocab_size - 2;
        for (auto kindex = uint32_t{}; kindex < code_length; ++kindex) {
            tokens[index].code[code_length - kindex - 1] = code[kindex];
            tokens[index].point[code_length - kindex] = point[kindex] - vocab_size;
        }
    }
}

yzw2v::huff::HuffmanTree::HuffmanTree(const vocab::Vocabulary& vocab)
    : tokens_(vocab.size())
    , points_pool_{POINTS_BLOCK_SIZE}
    , code_pool_{CODES_BLOCK_SIZE} {
    CreateHuffmanTree(vocab, tokens_, points_pool_, code_pool_);
}

const std::vector<yzw2v::huff::Token>&
yzw2v::huff::HuffmanTree::Tokens() const noexcept {
    return tokens_;
}
