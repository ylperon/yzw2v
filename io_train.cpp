#include "train.h"
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
        const auto* const row = matrix + i * model.vector_size;
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

    const auto* const matrix = model.matrix_holder.get();

    out << model.vocabulary_size << ' ' << model.vector_size << '\n';
    for (auto i = uint32_t{}; i < model.vocabulary_size; ++i) {
        out << vocab.Token(i).token << ' ';
        const auto* const row = matrix + i * model.vector_size;
        for (auto j = uint32_t{}; j < model.vector_size; ++j) {
            out.write(reinterpret_cast<const char*>(row + j), sizeof(*(row + j)));
        }
        out << '\n';
    }
}
