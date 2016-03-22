#pragma once

#include "mem.h"

#include <memory>

#include <cstdint>

namespace yzw2v {
    namespace num {
        class Matrix {
        public:
            Matrix(const uint32_t rows_count, const uint32_t columns_count);

            float* row(const uint32_t index) noexcept;
            const float* row(const uint32_t index) const noexcept;

            uint32_t rows_count() const noexcept;
            uint32_t columns_count() const noexcept;

        private:
            uint32_t padded_columns_count_;
            float* matrix_;
            uint32_t rows_count_;
            uint32_t columns_count_;
            std::unique_ptr<float, mem::detail::Deleter> matrix_holder_;
        };
    }
}
