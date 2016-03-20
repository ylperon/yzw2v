#include "vocabulary.h"
#include "io.h"

#include <fstream>
#include <memory>

#include <cstring>

std::ostream& operator<<(std::ostream& out, const yzw2v::vocab::Token& token) {
    for (const auto c : token) {
        out << c;
    }

    return out;
}

void yzw2v::vocab::WriteTSV(const Vocabulary& vocab, const std::string& path) {
    Vocabulary::WriteTSVWithFilter(vocab, path, 0);
}

void yzw2v::vocab::WriteTSVWithFilter(const Vocabulary& vocab, const std::string& path,
                                      const uint32_t min_token_freq) {
    Vocabulary::WriteTSVWithFilter(vocab, path, min_token_freq);
}

void yzw2v::vocab::Vocabulary::WriteTSVWithFilter(const Vocabulary& vocab, const std::string& path,
                                                  const uint32_t min_token_freq) {
    std::ofstream out{path, std::ios::out};
    if (!out) {
        throw std::runtime_error{"failed to open file for writing"};
    }

    for (const auto& info : vocab.tokens_) {
        if (info.count < min_token_freq) {
            continue;
        }

        out << info.count
            << '\t' << info.token
            << '\n';
    }
}

static const char VOCABULARY_MAGIC[] = {"YZW2V_VOCABULARY (ﾉ◕ヮ◕)ﾉ*:･ﾟ✧"};
static constexpr size_t VOCABULARY_MAGIC_SIZE = sizeof(VOCABULARY_MAGIC)
                                                / sizeof(VOCABULARY_MAGIC[0]);

static uint32_t IntHash(const uint32_t value) noexcept {
    // Knuth's Multiplicative Method
    return value * uint32_t{2654435761};
}

void yzw2v::vocab::WriteBinary(const Vocabulary& vocab, const std::string& path) {
    Vocabulary::WriteBinaryWithFilter(vocab, path, 0);
}

void yzw2v::vocab::WriteBinaryWithFilter(const Vocabulary& vocab, const std::string& path,
                                         const uint32_t min_token_freq) {
    Vocabulary::WriteBinaryWithFilter(vocab, path, min_token_freq);
}

void yzw2v::vocab::Vocabulary::WriteBinaryWithFilter(
    const Vocabulary& vocab, const std::string& path, const uint32_t min_token_freq
){
    const auto number_of_tokens = [&vocab, min_token_freq]{
        if (!min_token_freq) {
            return vocab.size();
        }

        auto res = uint32_t{};
        for (const auto& info : vocab.tokens_) {
            if (info.count >= min_token_freq) {
                ++res;
            }
        }

        return res;
    }();

    std::ofstream out{path, std::ios::binary};
    if (!out) {
        throw std::runtime_error{"failed to open file for writing"};
    }

    static constexpr size_t BUFFER_SIZE = 1024 * 1024 * 32; // 32 Mb
    io::BinaryBufferedWriteProxy proxy{out, BUFFER_SIZE};

    // [VOCABULARY_MAGIC]
    proxy.Write(VOCABULARY_MAGIC, VOCABULARY_MAGIC_SIZE);

    // [vocab.max_number_of_tokens_, hash(vocab.max_number_of_tokens_)]
    proxy.Write(&vocab.max_number_of_tokens_, sizeof(vocab.max_number_of_tokens_));
    const auto max_number_of_tokens_hash = IntHash(vocab.max_number_of_tokens_);
    proxy.Write(&max_number_of_tokens_hash, sizeof(max_number_of_tokens_hash));

    // [vocab.hash_table_size_, hash(vocab.hash_table_size_)]
    proxy.Write(&vocab.hash_table_size_, sizeof(vocab.hash_table_size_));
    const auto hash_table_size_hash = IntHash(vocab.hash_table_size_);
    proxy.Write(&hash_table_size_hash, sizeof(hash_table_size_hash));

    // [vocab.tokens_.size(), hash(vocab.tokens_.size())]
    proxy.Write(&number_of_tokens, sizeof(number_of_tokens));
    const auto number_of_tokens_hash = IntHash(number_of_tokens);
    proxy.Write(&number_of_tokens_hash, sizeof(number_of_tokens_hash));

    // [count, length(token), token]
    for (const auto& info : vocab.tokens_) {
        if (info.count < min_token_freq) {
            continue;
        }

        proxy.Write(&info.count, sizeof(info.count));
        const auto size = info.token.length();
        proxy.Write(&size, sizeof(size));
        proxy.Write(info.token.cbegin(), info.token.length());
    }
}

yzw2v::vocab::Vocabulary yzw2v::vocab::ReadBinary(const std::string& path) {
    Vocabulary res{1};
    Vocabulary::ReadBinaryWithFilter(path, 0, res);
    return res;
}

void yzw2v::vocab::ReadBinary(const std::string& path, Vocabulary& vocab) {
    Vocabulary::ReadBinaryWithFilter(path, 0, vocab);
}

yzw2v::vocab::Vocabulary yzw2v::vocab::ReadBinaryWithFilter(const std::string& path,
                                                            const uint32_t min_token_freq) {
    Vocabulary res{1};
    Vocabulary::ReadBinaryWithFilter(path, min_token_freq, res);
    return res;
}

void yzw2v::vocab::ReadBinaryWithFilter(const std::string& path, const uint32_t min_token_freq,
                                        Vocabulary& vocab) {
    Vocabulary::ReadBinaryWithFilter(path, min_token_freq, vocab);
}

void yzw2v::vocab::Vocabulary::ReadBinaryWithFilter(const std::string& path,
                                                    const uint32_t min_token_freq,
                                                    Vocabulary& vocab) {
    std::ifstream in{path, std::ios::binary | std::ios::ate};
    if (!in) {
        throw std::runtime_error{"failed to open file for reading"};
    }

    const auto in_size = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);

    static constexpr size_t BUFFER_SIZE = 1024 * 1024 * 32; // 32 Mb
    io::BinaryBufferedReadProxy proxy{in, in_size, BUFFER_SIZE};

    {
        char magic[VOCABULARY_MAGIC_SIZE] = {};
        proxy.Read(magic, VOCABULARY_MAGIC_SIZE);
        if (std::strncmp(VOCABULARY_MAGIC, magic, VOCABULARY_MAGIC_SIZE)) {
            throw std::runtime_error{"magic doesn't match"};
        }
    }

    proxy.Read(&vocab.max_number_of_tokens_, sizeof(vocab.max_number_of_tokens_));
    {
        auto max_number_of_tokens_hash = uint32_t{};
        proxy.Read(&max_number_of_tokens_hash, sizeof(max_number_of_tokens_hash));
        if (IntHash(vocab.max_number_of_tokens_) != max_number_of_tokens_hash) {
            throw std::runtime_error{"hash(vocab.max_number_of_tokens_) doesn't match"};
        }
    }

    proxy.Read(&vocab.hash_table_size_, sizeof(vocab.hash_table_size_));
    {
        auto hash_table_size_hash = uint32_t{};
        proxy.Read(&hash_table_size_hash, sizeof(hash_table_size_hash));
        if (IntHash(vocab.hash_table_size_) != hash_table_size_hash) {
            throw std::runtime_error{"hash(vocab.hash_table_size_) does't match"};
        }
    }

    const auto number_of_tokens_to_read = [&proxy]{
        auto res = uint32_t{};
        proxy.Read(&res, sizeof(res));
        auto res_hash = uint32_t{};
        proxy.Read(&res_hash, sizeof(res_hash));
        if (IntHash(res) != res_hash) {
            throw std::runtime_error{"hash(number_of_tokens) doesn't match"};
        }

        return res;
    }();

    const std::unique_ptr<char[]> token_buffer{new char[MAX_TOKEN_LENGTH]};

    // it can be done in a more efficient manner
    vocab = Vocabulary{vocab.max_number_of_tokens_};
    for (auto i = uint32_t{}; i < number_of_tokens_to_read; ++i) {
        auto count = uint32_t{};
        proxy.Read(&count, sizeof(count));
        auto length = decltype(yzw2v::vocab::Token{}.length()){};
        proxy.Read(&length, sizeof(length));
        proxy.Read(token_buffer.get(), length);

        if (count >= min_token_freq ) {
            const auto id = vocab.Add({token_buffer.get(), length});
            vocab.tokens_[id].count = count;
        }
    }
}
