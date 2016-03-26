#pragma once

#include <iosfwd>
#include <memory>
#include <string>

#include <cstdint>

namespace yzw2v {
    namespace io {
        uint64_t FileSize(const std::string& path);

        class BinaryBufferedWriteProxy {
        public:
            BinaryBufferedWriteProxy(std::ostream& slave, const size_t buffer_size);
            ~BinaryBufferedWriteProxy();

            void Write(const void* const data, const size_t data_size);
            void Flush();

        private:
            uint8_t* buf_cur_;
            size_t buf_left_;
            std::ostream& slave_;
            const size_t buf_size_;
            const std::unique_ptr<uint8_t[]> buf_;
        };

        /* If you began reading via `BinaryBufferedReadProxy` instance you should keep reading only
         * via this proxy instance as some of input data may still be in its internal buffer and we
         * can't put it back into `std::istream` in general case.
         */
        class BinaryBufferedReadProxy {
        public:
            BinaryBufferedReadProxy(std::istream& slave, const size_t slave_size,
                                    const size_t buffer_size);

            void Read(void* const data, const size_t data_size);

        private:
            uint8_t* buf_cur_;
            size_t buf_left_;
            size_t slave_size_left_;
            std::istream& slave_;
            const size_t buf_size_;
            const std::unique_ptr<uint8_t[]> buf_;
        };
    }  // namespace io
}  // namespace yzw2v
