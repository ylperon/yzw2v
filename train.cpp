#include "train.h"

#include "huffman.h"
#include "io.h"
#include "matrix.h"
#include "mem.h"
#include "numeric.h"
#include "prefetch.h"
#include "prng.h"
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
#include <cstdio>

static constexpr uint64_t PER_THREAD_WORD_COUNT_TO_UPDATE_PARAMS = 10000;
static constexpr size_t EXP_TABLE_SIZE = 1000;
static constexpr uint32_t MAX_EXP = 6;
static constexpr float MAX_EXP_FLT = static_cast<float>(MAX_EXP);

namespace {
    struct SharedData {
        const yzw2v::sampling::UnigramDistribution unigram_distribution;

        const float* const exp_table;
        yzw2v::num::Matrix* const syn0;
        yzw2v::num::Matrix* const syn1hs;
        yzw2v::num::Matrix* const syn1neg;

        const uint64_t text_words_count_;
        uint64_t processed_words_count;
        float alpha;
        decltype(std::chrono::high_resolution_clock::now()) start_time;

        SharedData(const float alpha_,
                   yzw2v::num::Matrix* const syn0_,
                   yzw2v::num::Matrix* const syn1hs_,
                   yzw2v::num::Matrix* const syn1neg_,
                   const float* const exp_table_,
                   const yzw2v::vocab::Vocabulary& vocab)
            : unigram_distribution{vocab}
            , exp_table{exp_table_}
            , syn0{syn0_}
            , syn1hs{syn1hs_}
            , syn1neg{syn1neg_}
            , text_words_count_{vocab.TextWordCount()}
            , processed_words_count{}
            , alpha{alpha_}
            , start_time{std::chrono::high_resolution_clock::now()}
        {
        }
    };
} // namespace

static std::chrono::seconds GetTimePassed(const SharedData& data) noexcept {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::high_resolution_clock::now() - data.start_time
    );
}

static void Report(const float alpha, const uint64_t words_processed_count,
                   const uint64_t text_words_count, const uint32_t iterations_requested_for_model,
                   const std::chrono::seconds seconds_passed) {
    const auto progress = static_cast<double>(words_processed_count) // possible overflow
                          / (text_words_count * iterations_requested_for_model)
                          * 100;
    const auto words_per_sec = static_cast<double>(words_processed_count)
                               / (seconds_passed.count() + 1)
                               / 1000;

    fprintf(stdout, "%c[trainer] progress=%.6lf%% alpha=%.6f words/sec=%.2lfK  ",
           '\r', progress, static_cast<double>(alpha), words_per_sec);
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
                     const uint32_t seed,
                     SharedData& shared_data)
            : p_{params}
            , neu1_holder_{yzw2v::mem::AllocateFloatForSIMD(params.vector_size)}
            , neu1e_holder_{yzw2v::mem::AllocateFloatForSIMD(params.vector_size)}
            , shared_data_(shared_data)
            , neu1_{neu1_holder_.get()}
            , neu1e_{neu1e_holder_.get()}
            , vocab_{vocab}
            , huff_{huffman_tree}
            , prng_{seed}
            , sentence_position_{0}
            , prev_word_count_{0}
            , word_count_{0}
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

    private:
        const yzw2v::train::Params p_;
        const std::unique_ptr<float, yzw2v::mem::detail::Deleter> neu1_holder_;
        const std::unique_ptr<float, yzw2v::mem::detail::Deleter> neu1e_holder_;

        SharedData& shared_data_;
        float* const neu1_;
        float* const neu1e_;

        const yzw2v::vocab::Vocabulary& vocab_;
        const yzw2v::huff::HuffmanTree& huff_;

        yzw2v::sampling::PRNG prng_;

        std::vector<uint32_t> sentence_;
        uint32_t sentence_position_;

        uint64_t prev_word_count_;
        uint64_t word_count_;

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
            const auto window_indent = static_cast<uint32_t>(prng_() % p_.window_size);
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

void ModelTrainer::ReportAndUpdateAlpha() {
    shared_data_.processed_words_count += word_count_ - prev_word_count_;
    prev_word_count_ = word_count_;
    Report(shared_data_.alpha, shared_data_.processed_words_count, shared_data_.text_words_count_,
            p_.iterations_count, GetTimePassed(shared_data_));

    auto new_alpha = p_.starting_alpha
                     * (1 - static_cast<float>(shared_data_.processed_words_count)
                            / (shared_data_.text_words_count_ * p_.iterations_count + 1)
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
                (std::sqrt(count / (p_.min_token_freq_threshold * shared_data_.text_words_count_)) + 1.f)
                * (p_.min_token_freq_threshold * shared_data_.text_words_count_)
                / count;
            if (static_cast<double>(prob) < prng_.real_0_inc_1_inc()) {
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

        yzw2v::num::AddVector(neu1_, p_.vector_size, shared_data_.syn0->row(sentence_[index]));
    }

    yzw2v::num::MultiplyVector(neu1_, p_.vector_size, 1.0f / (window_end - window_begin));
}

void ModelTrainer::CBOWApplyHierarchicalSoftmax() {
    const auto token = huff_.Tokens()[sentence_[sentence_position_]];
    for (auto index = uint32_t{}; index < token.length; ++index) {
        const auto other_token = token.point[index];
        auto f = yzw2v::num::ScalarProduct(neu1_, p_.vector_size,
                                           shared_data_.syn1hs->row(other_token));

        if (f <= -MAX_EXP_FLT || f >= MAX_EXP_FLT) {
            continue;
        } else {
            const auto exp_index = static_cast<uint32_t>(
                (f + MAX_EXP_FLT) * (EXP_TABLE_SIZE / MAX_EXP / 2)
            );
            f = shared_data_.exp_table[exp_index];
        }

        const auto g = (1.0f - token.code[index] - f) * shared_data_.alpha;
        yzw2v::num::AddVector(neu1e_, p_.vector_size, shared_data_.syn1hs->row(other_token), g);
        yzw2v::num::AddVector(shared_data_.syn1hs->row(other_token), p_.vector_size, neu1e_, g);
    }
}

void ModelTrainer::CBOWApplyNegativeSampling() {
    shared_data_.unigram_distribution.prefetch(prng_);
    const auto cur_token = sentence_[sentence_position_];
    for (auto index = uint32_t{}; index < p_.negative_samples_count + 1; ++index) {
        auto target = uint32_t{};
        auto label = float{};
        if (0 == index) {
            target = cur_token;
            label = 1;
        } else {
            target = shared_data_.unigram_distribution(prng_);
            if (cur_token == target) {
                continue;
            }

            label = 0;
        }

        auto f = yzw2v::num::ScalarProduct(neu1_, p_.vector_size,
                                           shared_data_.syn1neg->row(target));

        auto g = float{};
        if (f > MAX_EXP_FLT) {
            g = (label - 1.0f) * shared_data_.alpha;
        } else if (f < -MAX_EXP_FLT) {
            g = (label - 0.0f) * shared_data_.alpha;
        } else {
            const auto exp_index = static_cast<uint32_t>(
                    (f + MAX_EXP_FLT) * (EXP_TABLE_SIZE / MAX_EXP / 2)
                    );
            g = (label - shared_data_.exp_table[exp_index]) * shared_data_.alpha;
        }

        yzw2v::num::AddVector(neu1e_, p_.vector_size, shared_data_.syn1neg->row(target), g);
        shared_data_.unigram_distribution.prefetch(prng_);
        yzw2v::num::AddVector(shared_data_.syn1neg->row(target), p_.vector_size, neu1_, g);
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

        yzw2v::num::AddVector(shared_data_.syn0->row(sentence_[index]), p_.vector_size, neu1e_);
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

static void InitializeMatrix(yzw2v::num::Matrix& matrix, yzw2v::sampling::PRNG& prng) noexcept {
    for (auto i = uint32_t{}; i < matrix.rows_count(); ++i) {
        auto* const row = matrix.row(i);
        for (auto j = uint32_t{}; j < matrix.columns_count(); ++j) {
            row[j] = static_cast<float>((prng.real_0_inc_1_inc() - 0.5) / matrix.columns_count());
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

static void Zeroize(yzw2v::num::Matrix& matrix) noexcept {
    for (uint32_t i = uint32_t{}; i < matrix.rows_count(); ++i) {
        yzw2v::num::Zeroize(matrix.row(i), matrix.columns_count());
    }
}

yzw2v::train::Model yzw2v::train::TrainCBOWModel(const std::string& path,
                                                 const vocab::Vocabulary& vocab,
                                                 const huff::HuffmanTree& huffman_tree,
                                                 const Params& params,
                                                 const uint32_t thread_count) {
    const auto syn1hs_holder = [&params, &vocab]() -> std::unique_ptr<num::Matrix> {
        if (params.use_hierarchical_softmax) {
            std::unique_ptr<num::Matrix> res{new num::Matrix{vocab.size(), params.vector_size}};
            Zeroize(*res);
            return res;
        }

        return nullptr;
    }();
    const auto syn1neg_holder = [&params, &vocab]() -> std::unique_ptr<num::Matrix> {
        if (params.negative_samples_count > 0) {
            std::unique_ptr<num::Matrix> res{new num::Matrix{vocab.size(), params.vector_size}};
            Zeroize(*res);
            return res;
        }

        return nullptr;
    }();
    const auto exp_table_holder = std::unique_ptr<const float[]>(GenerateExpTable(EXP_TABLE_SIZE));

    auto res = Model{
        vocab.size(), params.vector_size,
        std::unique_ptr<num::Matrix>{new num::Matrix{vocab.size(), params.vector_size}}
    };
    yzw2v::sampling::PRNG prng{params.prng_seed};
    InitializeMatrix(*res.matrix_holder, prng);

    const auto file_size = io::FileSize(path);
    const auto bytes_per_thread = file_size / thread_count;
    const auto bytes_per_thread_remainder = file_size % thread_count;
    SharedData shared_data{params.starting_alpha,
                           res.matrix_holder.get(), syn1hs_holder.get(), syn1neg_holder.get(),
                           exp_table_holder.get(), vocab};
    auto jobs = std::vector<std::future<void>>{};
    auto job_index = uint32_t{};
    for (auto offset = uint64_t{}; offset < file_size; offset += bytes_per_thread, ++job_index) {
        auto bytes_per_this_thread = bytes_per_thread;
        if (offset + bytes_per_thread + bytes_per_thread_remainder == file_size) {
            bytes_per_this_thread += bytes_per_thread_remainder;
        }

        jobs.emplace_back(std::async(std::launch::async,
            [&path, &vocab, &huffman_tree, &params, &shared_data, offset, bytes_per_this_thread,
             job_index]{
                ModelTrainer trainer{path, offset, bytes_per_this_thread,
                                     vocab, huffman_tree, params, job_index, shared_data};
                trainer.TrainCBOW();
        }));
    }

    for (auto&& job : jobs) {
        job.wait();
    }

    return res;
}
