#include "train.h"

#include "huffman.h"
#include "io.h"
#include "mem.h"
#include "numeric.h"
#include "token_reader.h"
#include "unigram_distribution.h"
#include "vocabulary.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
#include <random>

#include <cmath>

static constexpr uint64_t PER_THREAD_WORD_COUNT_TO_UPDATE_PARAMS = 10000;
static constexpr size_t EXP_TABLE_SIZE = 1000;
static constexpr uint32_t MAX_EXP = 6;
static constexpr float MAX_EXP_FLT = static_cast<float>(MAX_EXP);
static constexpr uint32_t UNIGRAM_TABLE_SIZE = 100000000;

namespace {
    struct SharedData {
        const yzw2v::train::UnigramDistribution unigram_distribution_;

        const float* const exp_table;
        float* const syn0; // vocabulary_size * vector_size
        float* const syn1hs; // vocabulary_size * vector_size
        float* const syn1neg; // vocabulary_size * vector_size

        uint64_t processed_words_count;
        float alpha;
        decltype(std::chrono::high_resolution_clock::now()) start_time;

        template <typename PRNG>
        SharedData(const float alpha_,
                   float* const syn0_,
                   float* const syn1hs_,
                   float* const syn1neg_,
                   const float* const exp_table_,
                   const uint32_t unigram_table_size,
                   const yzw2v::vocab::Vocabulary& vocab,
                   PRNG&& prng)
            : unigram_distribution_{unigram_table_size, vocab, prng}
            , exp_table{exp_table_}
            , syn0{syn0_}
            , syn1hs{syn1hs_}
            , syn1neg{syn1neg_}
            , processed_words_count{}
            , alpha{alpha_}
            , start_time{std::chrono::high_resolution_clock::now()}
        {
        }
    };
} // namespace

static std::chrono::seconds GetTimePassed(const SharedData& data) {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::high_resolution_clock::now() - data.start_time
    );
}

static void Report(const float alpha, const uint64_t words_processed_count,
                   const uint64_t text_words_count, const uint32_t iterations_requested_for_model,
                   const std::chrono::seconds seconds_passed, std::ostream& out) {
    const auto progress = static_cast<double>(words_processed_count) // possible overflow
                          / (text_words_count * iterations_requested_for_model)
                          * 100;
    const auto words_per_sec = static_cast<double>(words_processed_count)
                               / (seconds_passed.count() + 1)
                               / 1000;
    out << "[trainer]"
        << std::fixed
        << ' ' << "progress=" << progress << '%'
        << ' ' << "alpha=" << alpha
        << ' ' << "words/sec=" << words_per_sec << 'K'
        << '\r';
}

namespace {
    class ModelTrainer {
    public:
        ModelTrainer(const std::string& text_file_path,
                     const uint64_t text_file_offset,
                     const uint64_t bytes_to_read_from_text_file,
                     const yzw2v::vocab::Vocabulary& vocab,
                     const yzw2v::huff::HuffmanTree& huffman_tree,
                     const yzw2v::train::Params& params,
                     SharedData& shared_data)
            : p_{params}
            , neu1_holder_{yzw2v::mem::AllocateFloatForSIMD(params.vector_size)}
            , neu1e_holder_{yzw2v::mem::AllocateFloatForSIMD(params.vector_size)}
            , shared_data_{shared_data}
            , neu1_{neu1_holder_.get()}
            , neu1e_{neu1e_holder_.get()}
            , vocab_{vocab}
            , huff_{huffman_tree}
            , prng_{params.prng_seed}
            , uniform_window_{0, params.window_size}
            , unigram_distribution_cur_index_{static_cast<uint32_t>(prng_() % shared_data_.unigram_distribution_.size())}
            , sentence_position_{0}
            , prev_word_count_{0}
            , word_count_{0}
            , text_words_count_{vocab.TextWordCount()}
            , token_reader_{text_file_path, bytes_to_read_from_text_file, text_file_offset}
            , iteration_{0}
        {
            sentence_.reserve(params.max_sentence_length);
        }

        void TrainCBOW();

    private:
        void ReportAndUpdateAlpha();
        void ReadSentence();
        void CBOWPropagateInputToHidden(const uint32_t window_begin, const uint32_t window_end);
        void CBOWPropagateHiddenToInput(const uint32_t window_begin, const uint32_t window_end);
        void CBOWApplyHierarchicalSoftmax();
        void CBOWApplyNegativeSampling();

        uint32_t WindowBegin(const uint32_t window_indent) const noexcept;
        uint32_t WindowEnd(const uint32_t window_indent) const noexcept;

        uint32_t SampleFromUnigramDistribution() noexcept;

    private:
        const yzw2v::train::Params p_;
        const std::unique_ptr<float, yzw2v::mem::detail::Deleter> neu1_holder_;
        const std::unique_ptr<float, yzw2v::mem::detail::Deleter> neu1e_holder_;

        SharedData& shared_data_;
        float* const neu1_;
        float* const neu1e_;

        const yzw2v::vocab::Vocabulary& vocab_;
        const yzw2v::huff::HuffmanTree& huff_;

        std::minstd_rand prng_;
        std::uniform_int_distribution<uint32_t> uniform_window_;
        std::uniform_real_distribution<float> uniform01_;

        uint32_t unigram_distribution_cur_index_;

        std::vector<uint32_t> sentence_;
        uint32_t sentence_position_;

        uint64_t prev_word_count_;
        uint64_t word_count_;

        const uint64_t text_words_count_;
        yzw2v::io::TokenReader token_reader_;
        uint32_t iteration_;
    };
}  // namespace

void ModelTrainer::TrainCBOW() {
    for (ReadSentence(); iteration_ < p_.iterations_count; ReadSentence()) {
        if (word_count_ - prev_word_count_ > PER_THREAD_WORD_COUNT_TO_UPDATE_PARAMS) {
            ReportAndUpdateAlpha();
        }

        if (sentence_.empty()) {
            // can be empty only when iteration is over
            continue;
        }

        for (sentence_position_ = 0; sentence_position_ < sentence_.size(); ++sentence_position_) {
            const auto window_indent = uniform_window_(prng_);
            const auto window_begin = WindowBegin(window_indent);
            const auto window_end = WindowEnd(window_indent);

            yzw2v::num::Zeroize(neu1_, p_.vector_size);
            yzw2v::num::Zeroize(neu1e_, p_.vector_size);

            CBOWPropagateInputToHidden(window_begin, window_end);
            if (p_.use_hierarchical_softmax) {
                CBOWApplyHierarchicalSoftmax();
            }

            if (p_.negative_samples_count) {
                CBOWApplyNegativeSampling();
            }

            CBOWPropagateHiddenToInput(window_begin, window_end);
        }
    }
}

uint32_t ModelTrainer::SampleFromUnigramDistribution() noexcept {
    if (YZ_UNLIKELY(unigram_distribution_cur_index_ == shared_data_.unigram_distribution_.size())) {
        unigram_distribution_cur_index_ = 0;
    }

    return shared_data_.unigram_distribution_[unigram_distribution_cur_index_++];
}

void ModelTrainer::ReportAndUpdateAlpha() {
    shared_data_.processed_words_count += word_count_ - prev_word_count_;
    prev_word_count_ = word_count_;
    Report(shared_data_.alpha, shared_data_.processed_words_count, text_words_count_,
            p_.iterations_count, GetTimePassed(shared_data_), std::clog);

    auto new_alpha = static_cast<float>(
            p_.starting_alpha
            * (1 - static_cast<double>(shared_data_.processed_words_count)
                   / (text_words_count_ * p_.iterations_count + 1)
              )
            );
    if (new_alpha < p_.starting_alpha * 0.0001f) {
        new_alpha = p_.starting_alpha * 0.0001f;
    }

    shared_data_.alpha = new_alpha;
}

void ModelTrainer::ReadSentence() {
    sentence_.clear();
    while (!token_reader_.Done()) {
        const auto token_id = vocab_.ID(token_reader_.Read());
        if (yzw2v::vocab::INVALID_TOKEN_ID == token_id) {
            continue;
        }

        ++word_count_;
        if (yzw2v::vocab::PARAGRAPH_TOKEN_ID == token_id) {
            if (sentence_.empty()) {
                continue;
            } else {
                break;
            }
        }

        if (p_.min_token_freq_threshold > 0) {
            // subsampling goes here
            const auto count = vocab_.Count(token_id);
            const auto prob =
                (std::sqrt(count / (p_.min_token_freq_threshold * text_words_count_)) + 1.f)
                * (p_.min_token_freq_threshold * text_words_count_)
                / count;
            if (prob < uniform01_(prng_)) {
                continue;
            }
        }

        sentence_.push_back(token_id);
        if (sentence_.size() >= p_.max_sentence_length) {
            break;
        }
    }

    sentence_position_ = 0;

    if (token_reader_.Done()) {
        shared_data_.processed_words_count += word_count_ - prev_word_count_;
        ++iteration_;

        word_count_ = 0;
        prev_word_count_ = 0;
        sentence_.clear();
        sentence_position_ = 0;
        token_reader_.Restart();
    }
}

void ModelTrainer::CBOWPropagateInputToHidden(const uint32_t window_begin,
                                              const uint32_t window_end)
{
    assert(window_begin < window_end);
    for (auto index = window_begin; index < window_end; ++index) {
        if (sentence_position_ == index) {
            continue;
        }

        const auto shift = p_.vector_size * sentence_[index];
        yzw2v::num::AddVector(neu1_, p_.vector_size, shared_data_.syn0 + shift);
    }

    yzw2v::num::MultiplyVector(neu1_, p_.vector_size, 1.0f / (window_end - window_begin));
}

void ModelTrainer::CBOWApplyHierarchicalSoftmax() {
    const auto token = huff_.Tokens()[sentence_[sentence_position_]];
    for (auto index = uint32_t{}; index < token.length; ++index) {
        const auto shift = token.point[index] * p_.vector_size;
        auto f = yzw2v::num::ScalarProduct(neu1_, p_.vector_size, shared_data_.syn1hs + shift);

        if (f <= -MAX_EXP_FLT || f >= MAX_EXP_FLT) {
            continue;
        } else {
            const auto exp_index = static_cast<uint32_t>(
                (f + MAX_EXP_FLT) * (EXP_TABLE_SIZE / MAX_EXP / 2)
            );
            f = shared_data_.exp_table[exp_index];
        }

        const auto g = (1.0f - token.code[index] - f) * shared_data_.alpha;
        yzw2v::num::AddVector(neu1e_, p_.vector_size, shared_data_.syn1hs + shift, g);
        yzw2v::num::AddVector(shared_data_.syn1hs + shift, p_.vector_size, neu1e_, g);
    }
}

void ModelTrainer::CBOWApplyNegativeSampling() {
    const auto cur_token = sentence_[sentence_position_];
    for (auto index = uint32_t{}; index < p_.negative_samples_count + 1; ++index) {
        auto target = uint32_t{};
        auto label = float{};
        if (0 == index) {
            target = cur_token;
            label = 1;
        } else {
            target = SampleFromUnigramDistribution();
            if (yzw2v::vocab::PARAGRAPH_TOKEN_ID == target) {
                target = prng_() % (vocab_.size() - 1) + 1;
            }

            if (cur_token == target) {
                continue;
            }

            label = 0;
        }

        const auto shift = target * p_.vector_size;
        auto f = yzw2v::num::ScalarProduct(neu1_, p_.vector_size, shared_data_.syn1neg + shift);

        auto g = float{};
        if (f > MAX_EXP_FLT) {
            g = (label - 1) * shared_data_.alpha;
        } else if (f < -MAX_EXP_FLT) {
            g = (label - 0) * shared_data_.alpha;
        } else {
            const auto exp_index = static_cast<uint32_t>(
                    (f + MAX_EXP_FLT) * (EXP_TABLE_SIZE / MAX_EXP / 2)
                    );
            g = (label - shared_data_.exp_table[exp_index]) * shared_data_.alpha;
        }

        yzw2v::num::AddVector(neu1e_, p_.vector_size, shared_data_.syn1neg + shift, g);
        yzw2v::num::AddVector(shared_data_.syn1neg + shift, p_.vector_size, neu1_, g);
    }
}

void ModelTrainer::CBOWPropagateHiddenToInput(const uint32_t window_begin,
                                              const uint32_t window_end)
{
    assert(window_begin < window_end);
    for (auto index = window_begin; index < window_end; ++index) {
        if (sentence_position_ == index) {
            continue;
        }

        const auto shift = sentence_[index] * p_.vector_size;
        yzw2v::num::AddVector(shared_data_.syn0 + shift, p_.vector_size, neu1e_);
    }
}

uint32_t ModelTrainer::WindowBegin(const uint32_t window_indent) const noexcept {
    if (sentence_position_ + window_indent < p_.window_size) {
        return uint32_t{};
    }

    return sentence_position_ + window_indent - p_.window_size;
}

uint32_t ModelTrainer::WindowEnd(const uint32_t window_indent) const noexcept {
    if (sentence_position_ + p_.window_size - window_indent + 1 > sentence_.size()) {
        return static_cast<uint32_t>(sentence_.size());
    }

    return sentence_position_ + p_.window_size - window_indent + 1;
}

template <typename PRNG>
static void InitializeMatrix(const uint32_t row_count, const uint32_t column_count,
                             float* const matrix, PRNG&& prng) {
    std::uniform_real_distribution<float> uniform01{-0.5f, 0.5f};
    for (auto row = uint32_t{}; row < row_count; ++row) {
        for (auto column = uint32_t{}; column < column_count; ++column) {
            matrix[row * column_count + column] = uniform01(prng) / column_count;
        }
    }
}

static std::unique_ptr<float[]> GenerateExpTable(const uint32_t size) {
    std::unique_ptr<float[]> res{new float[size]};
    float* const a = res.get();
    for (auto i = uint32_t{}; i < size; ++i) {
        a[i] = static_cast<float>(std::exp((static_cast<double>(i) / size * 2 - 1) * MAX_EXP));
        a[i] = a[i] / (a[i] + 1);
    }

    return res;
}

yzw2v::train::Model yzw2v::train::TrainCBOWModel(const std::string& path,
                                                 const vocab::Vocabulary& vocab,
                                                 const huff::HuffmanTree& huffman_tree,
                                                 const Params& params,
                                                 const uint32_t thread_count) {
    const auto matrix_size = vocab.size() * params.vector_size;
    const auto syn1hs_holder = [&params, matrix_size]() -> std::unique_ptr<float, yzw2v::mem::detail::Deleter> {
        if (params.use_hierarchical_softmax) {
            auto res = mem::AllocateFloatForSIMD(matrix_size);
            yzw2v::num::Zeroize(res.get(), matrix_size);
            return res;
        }

        return nullptr;
    }();
    const auto syn1neg_holder = [&params, matrix_size]() -> std::unique_ptr<float, yzw2v::mem::detail::Deleter> {
        if (params.negative_samples_count > 0) {
            auto res = mem::AllocateFloatForSIMD(matrix_size);
            yzw2v::num::Zeroize(res.get(), matrix_size);
            return res;
        }

        return nullptr;
    }();
    const auto exp_table_holder = std::unique_ptr<const float[]>(GenerateExpTable(EXP_TABLE_SIZE));

    auto res = Model{vocab.size(), params.vector_size, mem::AllocateFloatForSIMD(matrix_size)};
    std::minstd_rand prng{params.prng_seed};
    InitializeMatrix(vocab.size(), params.vector_size, res.matrix_holder.get(), prng);

    const auto file_size = io::FileSize(path);
    const auto bytes_per_thread = file_size / thread_count;
    const auto bytes_per_thread_remainder = file_size % thread_count;
    SharedData shared_data{params.starting_alpha,
                           res.matrix_holder.get(), syn1hs_holder.get(), syn1neg_holder.get(),
                           exp_table_holder.get(), UNIGRAM_TABLE_SIZE, vocab, prng};
    auto jobs = std::vector<std::future<void>>{};
    for (auto offset = uint64_t{}; offset < file_size; offset += bytes_per_thread) {
        auto bytes_per_this_thread = bytes_per_thread;
        if (offset + bytes_per_thread + bytes_per_thread_remainder == file_size) {
            bytes_per_this_thread += bytes_per_thread_remainder;
        }

        jobs.emplace_back(std::async(std::launch::async,
            [&path, &vocab, &huffman_tree, &params, &shared_data, offset, bytes_per_this_thread]{
                ModelTrainer trainer{path, offset, bytes_per_this_thread,
                                     vocab, huffman_tree, params, shared_data};
                trainer.TrainCBOW();
        }));
    }

    for (auto&& job : jobs) {
        job.wait();
    }

    return res;
}
