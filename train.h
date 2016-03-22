#pragma once

#include "mem.h"
#include "matrix.h"

#include <memory>
#include <string>

#include <cstdint>

namespace yzw2v {
    namespace vocab {
        class Vocabulary;
    }

    namespace huff {
        class HuffmanTree;
    }
}

namespace yzw2v {
    namespace train {
        static constexpr uint32_t DEFAULT_ITERATIONS_COUNT = 5;
        static constexpr float DEFAULT_STARTING_ALPHA = 0.05f;
        static constexpr uint32_t DEFAULT_MAX_SENTENCE_LENGTH = 1000;
        static constexpr float DEFAULT_MIN_TOKEN_FREQ_THRESHOLD = 1e-3f;
        static constexpr uint32_t DEFAULT_NEGATIVE_SAMPLES_COUNT = 5;
        static constexpr bool DEFAULT_USE_HIERARCHICAL_SOFTMAX = false;
        static constexpr uint32_t DEFAULT_VECTOR_SIZE = 100;
        static constexpr uint32_t DEFAULT_WINDOW_SIZE = 5;
        static constexpr uint32_t DEFAULT_PRNG_SEED = 1;

        struct Params {
            uint32_t iterations_count = DEFAULT_ITERATIONS_COUNT;
            float starting_alpha = DEFAULT_STARTING_ALPHA;
            uint32_t max_sentence_length = DEFAULT_MAX_SENTENCE_LENGTH;
            float min_token_freq_threshold = DEFAULT_MIN_TOKEN_FREQ_THRESHOLD;
            uint32_t negative_samples_count = DEFAULT_NEGATIVE_SAMPLES_COUNT;
            bool use_hierarchical_softmax = DEFAULT_USE_HIERARCHICAL_SOFTMAX;
            uint32_t vector_size = DEFAULT_VECTOR_SIZE;
            uint32_t window_size = DEFAULT_WINDOW_SIZE;
            uint32_t prng_seed = DEFAULT_PRNG_SEED;
        };

        struct Model {
            uint32_t vocabulary_size;
            uint32_t vector_size;
            std::unique_ptr<num::Matrix> matrix_holder;
        };

        void WriteModelTXT(const std::string& path,
                           const vocab::Vocabulary& vocab,  const Model& model);

        void WriteModelBinary(const std::string& path,
                              const vocab::Vocabulary& vocab, const Model& model);

        Model TrainCBOWModel(const std::string& path,
                              const vocab::Vocabulary& vocab,
                              const huff::HuffmanTree& huffman_tree,
                              const Params& params, const uint32_t thread_count);
    }  // namespace train
}  // namespace yzw2v
