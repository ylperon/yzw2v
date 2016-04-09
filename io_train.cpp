#include "train.h"

#include "io.h"
#include "vocabulary.h"

#include <fstream>
#include <iomanip>

#include <cassert>

void yzw2v::train::WriteModelTXT(const std::string& path,
                                 const vocab::Vocabulary& vocab, const Model& model) {
    assert(vocab.size() == model.vocabulary_size);

    std::ofstream out{path, std::ios::binary};
    if (!out) {
        throw std::runtime_error{"failed to open file for write"};
    }

    const auto* const matrix = model.matrix_holder.get();

    out << model.vocabulary_size << ' ' << model.vector_size << '\n';
    for (auto i = uint32_t{}; i < model.vocabulary_size; ++i) {
        out << vocab.Token(i).token;
        const auto* const row = matrix->row(i);
        for (auto j = uint32_t{}; j < model.vector_size; ++j) {
            out << ' ' << row[j];
        }
        out << '\n';
    }
}

void yzw2v::train::WriteModelBinary(const std::string& path,
                                    const vocab::Vocabulary& vocab, const Model& model) {
    assert(vocab.size() == model.vocabulary_size);

    std::ofstream out{path, std::ios::binary};
    if (!out) {
        throw std::runtime_error{"failed to open file for write"};
    }

    static constexpr char SPACE[] = " ";
    static constexpr auto SPACE_LEN = sizeof(SPACE) - 1;
    static constexpr char NEW_LINE[] = "\n";
    static constexpr auto NEW_LINE_LEN = sizeof(NEW_LINE) - 1;
    static constexpr auto BUFFER_SIZE = size_t{1024} * 1024 * 128; // 128 Mb

    const auto* const matrix = model.matrix_holder.get();
    const auto vocabulary_size_str = std::to_string(model.vocabulary_size);
    const auto vector_size_str = std::to_string(model.vector_size);

    io::BinaryBufferedWriteProxy proxy{out, BUFFER_SIZE};
    proxy.Write(vocabulary_size_str.c_str(), vocabulary_size_str.size());
    proxy.Write(SPACE, SPACE_LEN);
    proxy.Write(vector_size_str.c_str(), vector_size_str.size());
    for (auto i = uint32_t{}; i < model.vocabulary_size; ++i) {
        proxy.Write(vocab.Token(i).token.cbegin(), vocab.Token(i).token.length());
        proxy.Write(SPACE, SPACE_LEN);
        proxy.Write(matrix->row(i), model.vector_size * sizeof(float));
        proxy.Write(NEW_LINE, NEW_LINE_LEN);
    }
}
