#include "io.h"
#include "likely.h"

#include <fstream>
#include <istream>
#include <ostream>

#include <cstring>

uint64_t yzw2v::io::FileSize(const std::string& path) {
    std::ifstream in{path, std::ios::binary | std::ios::ate};
    if (!in) {
        throw std::runtime_error("failed to open file");
    }

    return static_cast<uint64_t>(in.tellg());
}

yzw2v::io::BinaryBufferedWriteProxy::BinaryBufferedWriteProxy(std::ostream& slave,
                                                              const size_t buffer_size)
    : slave_(slave)
    , buf_size_{buffer_size}
    , buf_{new uint8_t[buffer_size]}
{
    buf_cur_ = buf_.get();
    buf_left_ = buffer_size;
}

void yzw2v::io::BinaryBufferedWriteProxy::Write(const void* const data, const size_t data_size) {
    const auto* data_cur = static_cast<const uint32_t*>(data);
    auto data_left = data_size;

    while (data_left > buf_left_) {
        std::memmove(buf_cur_, data_cur, buf_left_);
        slave_.write(reinterpret_cast<const char*>(buf_.get()),
                     static_cast<std::streamsize>(buf_size_));

        data_cur += buf_left_;
        data_left -= buf_left_;
        buf_cur_ = buf_.get();
        buf_left_ = buf_size_;
    }

    std::memmove(buf_cur_, data_cur, data_left);
    buf_cur_ += data_left;
    buf_left_ -= data_left;
}

void yzw2v::io::BinaryBufferedWriteProxy::Flush() {
    slave_.write(reinterpret_cast<const char*>(buf_.get()),
                 static_cast<std::streamsize>(buf_size_ - buf_left_));
    buf_left_ = buf_size_;
    buf_cur_ = buf_.get();
    slave_.flush();
}

yzw2v::io::BinaryBufferedWriteProxy::~BinaryBufferedWriteProxy() {
    Flush();
}

yzw2v::io::BinaryBufferedReadProxy::BinaryBufferedReadProxy(std::istream& slave,
                                                            const size_t slave_size,
                                                            const size_t buffer_size)
    : slave_size_left_{slave_size}
    , slave_(slave)
    , buf_size_{buffer_size}
    , buf_{new uint8_t[buffer_size]}
{
    buf_cur_ = buf_.get();
    buf_left_ = 0;
}

void yzw2v::io::BinaryBufferedReadProxy::Read(void* const data, const size_t data_size) {
    if (YZ_UNLIKELY(data_size > slave_size_left_ + buf_left_)) {
        throw std::runtime_error{"can't read that many from slave stream"};
    }

    auto* data_cur = static_cast<uint32_t*>(data);
    auto data_left = data_size;
    while (data_left > buf_left_) {
        std::memmove(data_cur, buf_cur_, buf_left_);
        data_left -= buf_left_;
        data_cur += buf_left_;

        buf_left_ = std::min(buf_size_, slave_size_left_);
        buf_cur_ = buf_.get();
        slave_.read(reinterpret_cast<char*>(buf_cur_), static_cast<std::streamsize>(buf_left_));
        slave_size_left_ -= buf_left_;
    }

    std::memmove(data_cur, buf_cur_, data_left);
    buf_cur_ += data_left;
    buf_left_ -= data_left;
}
