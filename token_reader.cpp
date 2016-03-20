#include "token_reader.h"
#include "vocabulary.h"
#include "likely.h"

#include <cstring>

static const yzw2v::vocab::Token PARAGRAPH_TOKEN{"</s>"};
static constexpr uint32_t BUF_SIZE = 1024 * 1024 * 32 - 1; // 32 Mb

bool yzw2v::io::TokenReader::Done() const noexcept {
    return !buf_cur_;
}

void yzw2v::io::TokenReader::LoadToBuffer() {
    const auto bytes_to_read = static_cast<uint32_t>(
            std::min(bytes_left_,
                     static_cast<uint64_t>(BUF_SIZE) - unprocessed_size_));
    if (!input_.read(buf_.get() + unprocessed_size_, static_cast<std::streamsize>(bytes_to_read))) {
        throw std::runtime_error("read failed");
    }

    *(buf_.get() + unprocessed_size_ + bytes_to_read) = '\0';
    buf_cur_ = buf_.get() - 1;  // -1 to compensate `++buf_cur_` in while-for loop
    bytes_left_ -= bytes_to_read;
    token_begin_ = nullptr;
}

yzw2v::io::TokenReader::TokenReader(const std::string& path,
                                    const uint64_t bytes_to_read,
                                    const uint64_t offset)
    : return_paragraph_{true}  // first token we return must be a paragraph token
    , token_begin_{nullptr}
    , unprocessed_size_{0}
    , bytes_left_{bytes_to_read}
    , buf_{new char[BUF_SIZE + 1]} // +1 for zero-terminator
    , input_{path, std::ios::binary}
    , bytes_to_read_from_input_file_{bytes_to_read}
    , input_file_offset_{offset} {
    if (!input_.seekg(static_cast<std::streamoff>(input_file_offset_))) {
        throw std::runtime_error("seekg failed");
    }

    buf_cur_ = buf_.get() - 1; // -1 to compensate `++buf_cur_` in while-for loop
    LoadToBuffer();
}

void yzw2v::io::TokenReader::Restart() {
    return_paragraph_ = true;
    token_begin_ = nullptr;
    unprocessed_size_ = 0;
    bytes_left_ = bytes_to_read_from_input_file_;
    if (!input_.seekg(static_cast<std::streamoff>(input_file_offset_))) {
        throw std::runtime_error("seekg failed");
    }

    buf_cur_ = buf_.get() - 1;
    LoadToBuffer();
}

yzw2v::io::TokenReader::TokenReader(const std::string& path,
                                    const uint64_t bytes_to_read)
    : TokenReader{path, bytes_to_read, 0} {
}

yzw2v::vocab::Token yzw2v::io::TokenReader::Read() {
    if (return_paragraph_) {
        return_paragraph_ = false;
        return PARAGRAPH_TOKEN;
    }

    while (!Done()) {
        for (++buf_cur_; *buf_cur_; ++buf_cur_) {
            if (YZ_UNLIKELY(' ' == *buf_cur_ || '\t' == *buf_cur_ || '\n' == *buf_cur_)) {
                if ('\n' == *buf_cur_) {
                    return_paragraph_ = true;
                }

                if (YZ_LIKELY(!!token_begin_)) {
                    // previous symbol was last symbol of the token
                    if (buf_cur_ - token_begin_ <= static_cast<std::ptrdiff_t>(yzw2v::vocab::MAX_TOKEN_LENGTH)) {
                        const vocab::Token token{token_begin_, buf_cur_};
                        token_begin_ = nullptr;
                        return token;
                    }

                    token_begin_ = nullptr;
                }

                if (return_paragraph_) {
                    return_paragraph_ = false;
                    return PARAGRAPH_TOKEN;
                }

                continue;
            }

            if (!token_begin_) {
                token_begin_ = buf_cur_;
            }
        }

        if (token_begin_) {
            // token at the end of the buffer, we have to copy it to the begining of the buffer, so on
            // the next iteration we could finish process this token
            unprocessed_size_ = static_cast<uint32_t>(buf_cur_ - token_begin_);
            std::memmove(buf_.get(), token_begin_, unprocessed_size_);
            token_begin_ = nullptr;
        }

        if (bytes_left_) {
            LoadToBuffer();
        } else {
            buf_cur_ = nullptr;
        }
    }

    return PARAGRAPH_TOKEN;
}
