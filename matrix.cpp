#include "matrix.h"

yzw2v::num::Matrix::Matrix(const uint32_t rows_count, const uint32_t columns_count)
    : padded_columns_count_{mem::RoundSizeUpByVecSize(columns_count)}
    , rows_count_{rows_count}
    , columns_count_{columns_count}
    , matrix_holder_{mem::AllocateFloatForSIMD(rows_count * padded_columns_count_)}
{
    matrix_ = matrix_holder_.get();
}

float* yzw2v::num::Matrix::row(const uint32_t index) noexcept {
    return matrix_ + padded_columns_count_ * index;
}

const float* yzw2v::num::Matrix::row(const uint32_t index) const noexcept {
    return matrix_ + padded_columns_count_ * index;
}

uint32_t yzw2v::num::Matrix::rows_count() const noexcept {
    return rows_count_;
}

uint32_t yzw2v::num::Matrix::columns_count() const noexcept {
    return columns_count_;
}
