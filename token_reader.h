#pragma once

#include <fstream>
#include <string>
#include <memory>

namespace yzw2v {
    namespace vocab {
        class Token;
    }
}

namespace yzw2v {
    namespace io {
        class TokenReader {
        public:
            TokenReader(const std::string& path,
                        const uint64_t bytes_to_read,
                        const uint64_t offset);

            TokenReader(const std::string& path,
                        const uint64_t bytes_to_read);

            vocab::Token Read();
            bool Done() const noexcept;

            void Restart();

        private:
            void LoadToBuffer();

        private:
            bool return_paragraph_;
            char* buf_cur_;
            const char* token_begin_;
            uint32_t unprocessed_size_;
            uint64_t bytes_left_;
            std::unique_ptr<char[]> buf_;
            std::ifstream input_;
            uint64_t bytes_to_read_from_input_file_;
            uint64_t input_file_offset_;
        };
    }
}
