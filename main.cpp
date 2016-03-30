#include "huffman.h"
#include "train.h"
#include "vocabulary.h"

#include "third_party/cxxopts/src/cxxopts.hpp"

#include <chrono>
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <fenv.h>
#endif

static constexpr uint32_t MAX_NUMBER_OF_TOKENS = 21000000;

static void SetFloatinPointEnvironment() {
#if defined(__linux__)
    feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);
#endif
}

namespace {
    struct Args {
        std::string text_file;
        std::string model_file;
        uint32_t vector_size = 100;
        uint32_t max_window_size = 5;
        float sample_rate = 1e-3f;
        bool use_hierarchical_softmax = false;
        uint32_t number_of_negative_samples = 5;
        uint32_t thread_count = 12;
        uint32_t iterations = 5;
        uint32_t min_word_frequency = 5;
        float alpha = 0.05f;
        bool save_model_in_binary_format = true;
        std::string vocabulary_out_file;
        std::string vocabulary_in_file;

        bool fail_on_bad_floating_arithmetics = false;
    };
}

static uint32_t GetDefaultThreadCount() noexcept {
    return (std::thread::hardware_concurrency() + 1) / 2;
}

static Args ParseOptions(int argc, char* argv[]) {
    auto options = cxxopts::Options{argv[0]};
    auto args = Args{};
    options.add_options()(
        "size",
        "Set size of word vectors",
        cxxopts::value<>(args.vector_size)->default_value("100"),
        "INT"
    )(
        "train",
        "Use text data from FILE to train the model",
        cxxopts::value<>(args.text_file),
        "FILE"
    )(
        "save-vocab",
        "The vocabulary will be saved to FILE",
        cxxopts::value<>(args.vocabulary_out_file),
        "FILE"
    )(
        "read-vocab",
        "The vocabulary will be read from FILE, not constructed from the training data",
        cxxopts::value<>(args.vocabulary_in_file),
        "FILE"
    )(
        "binary",
        "Save the resulting vectors in binary format",
        cxxopts::value<>(args.save_model_in_binary_format)->default_value("1")
    )(
        "alpha",
        "Set the starting learning rate",
        cxxopts::value<>(args.alpha)->default_value("0.05"),
        "FLOAT"
    )(
        "output",
        "Use FILE to save the resulting word vectors",
        cxxopts::value<>(args.model_file),
        "FILE"
    )(
        "window",
        "Set max skip length between words",
        cxxopts::value<>(args.max_window_size)->default_value("5"),
        "INT"
    )(
        "sample",
        "Set threshold for occurrence of words. Those that appear with higher frequency in the training data will be randomly down-sampled",
        cxxopts::value<>(args.sample_rate)->default_value("0.001"),
        "FLOAT"
    )(
        "hs",
        "Use Hierarchical Softmax",
        cxxopts::value<>(args.use_hierarchical_softmax)
    )(
        "negative",
        "Number of negative examples",
        cxxopts::value<>(args.number_of_negative_samples)->default_value("5"),
        "INT"
    )(
        "threads",
        "Use <int> threads",
        cxxopts::value<>(args.thread_count)->default_value(std::to_string(GetDefaultThreadCount())),
        "INT"
    )(
        "iter",
        "Run more training iterations",
        cxxopts::value<>(args.iterations)->default_value("5"),
        "INT"
    )(
        "min-count",
        "This will discard words that appear less than INT times",
        cxxopts::value<>(args.min_word_frequency)->default_value("5"),
        "INT"
    )(
        "fail-on-bad-floating-arithmetics",
        "properly set floating point environment",
        cxxopts::value<>(args.fail_on_bad_floating_arithmetics)
    )(
        "h,help",
        "Print help"
    );

    options.parse(argc, argv);
    if (options.count("help")) {
        std::cout << options.help({""}) << std::endl;
        std::exit(EXIT_SUCCESS);
    }

    return args;
}

static yzw2v::train::Params MakeParamsFromArgs(const Args& args) noexcept {
    auto params = yzw2v::train::Params{};
    params.iterations_count = args.iterations;
    params.starting_alpha = args.alpha;
    params.min_token_freq_threshold = args.sample_rate;
    params.negative_samples_count = args.number_of_negative_samples;
    params.use_hierarchical_softmax = args.use_hierarchical_softmax;
    params.vector_size = args.vector_size;
    params.window_size = args.max_window_size;
    return params;
}

static int Main(const Args& args) {
    const auto vocab = [&args]{
        if (!args.vocabulary_in_file.empty()) {
            return yzw2v::vocab::ReadBinary(args.vocabulary_in_file);
        }

        return yzw2v::vocab::CollectVocabulary(args.text_file, args.min_word_frequency,
                                               MAX_NUMBER_OF_TOKENS);
    }();

    if (!args.vocabulary_out_file.empty()) {
        yzw2v::vocab::WriteBinary(vocab, args.vocabulary_out_file);
    }

    if (args.model_file.empty()) {
        return EXIT_SUCCESS;
    }

    std::clog << "Vocabulary size: " << vocab.size() << std::endl;
    const yzw2v::huff::HuffmanTree huffman_tree{vocab};
    const auto params = MakeParamsFromArgs(args);
    const auto start_time = std::chrono::high_resolution_clock::now();
    const auto model = yzw2v::train::TrainCBOWModel(args.text_file, vocab, huffman_tree, params,
                                                    args.thread_count);
    const auto stop_time = std::chrono::high_resolution_clock::now();
    std::clog << "Training done in "
              << std::chrono::duration_cast<std::chrono::seconds>(stop_time - start_time).count()
              << " seconds"
              << std::endl;
    WriteModelBinary(args.model_file, vocab, model);

    return EXIT_SUCCESS;
}

int main(int argc, char* argv[]) {
    const auto args = ParseOptions(argc, argv);
    if (args.fail_on_bad_floating_arithmetics) {
        SetFloatinPointEnvironment();
    }

    return Main(args);
}
