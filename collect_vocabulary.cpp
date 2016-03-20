#include "vocabulary.h"
#include "token_reader.h"
#include "io.h"

#include <fstream>
#include <memory>

#include <cstdio>

/* You'll ask me: "Dude! Why do you write vocabulary to the disk?". The idea is following: To make
 * simple in-memory implementation of filtration we will need to allocate about the same amount of
 * memory we have already allocated (because we store tokens in memory pool) and because we deal
 * with "big data" (it's a homework on Large Scale Machine Learning, right?) we can't do it, but we
 * have disk! After all, we should be able to store vocabulary on disk, so we have some space on
 * disk and our vocabulary at this point must not be greater in size than vocabulary at the end of
 * input file processing.
 *
 * About performance. If we will create temprorary file in /tmp it will be located in tmpfs, which
 * means in-memory (see https://en.wikipedia.org/wiki/Tmpfs), so it should be fast.
 */
static void RemoveInfrequentTokens(yzw2v::vocab::Vocabulary& vocab, const uint32_t min_token_freq) {
    const std::unique_ptr<char[]> buf{new char[TMP_MAX + 1]};
    // btw, at least on Mac this function is deprecated
    const auto tmp_file_path = std::string{std::tmpnam(buf.get())};

    try {
        WriteBinaryWithFilter(vocab, tmp_file_path, min_token_freq);
        ReadBinary(tmp_file_path, vocab);
    } catch (std::exception&) {
        // not sure if we should check the return code
        std::remove(tmp_file_path.c_str());
        throw;
    }
}

void yzw2v::vocab::CollectIntoVocabulary(const std::string& path, const uint32_t min_token_freq,
                                         Vocabulary& vocab) {
    const auto file_size = io::FileSize(path);
    auto min_token_freq_during_collection = uint32_t{2};
    io::TokenReader reader{path, file_size};
    while (!reader.Done()) {
        for (auto index = 0; !reader.Done() && index < 10000; ++index) {
            vocab.Add(reader.Read());
        }

        if (vocab.LoadFactor() > 0.7) {
            RemoveInfrequentTokens(vocab, min_token_freq_during_collection);
            ++min_token_freq_during_collection;
        }
    }

    RemoveInfrequentTokens(vocab, min_token_freq);

    vocab.Sort();
}

yzw2v::vocab::Vocabulary yzw2v::vocab::CollectVocabulary(const std::string& path,
                                                         const uint32_t min_token_freq,
                                                         const uint32_t max_number_of_tokens) {
    Vocabulary vocab{max_number_of_tokens};
    CollectIntoVocabulary(path, min_token_freq, vocab);
    return vocab;
}
