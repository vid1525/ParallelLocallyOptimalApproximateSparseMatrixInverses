#include "methods_common.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace methods::common {

ColumnBatch::ColumnBatch(const int64_t rows, const int64_t first, const int64_t width)
  : first_column(first),
    m(rows, width),
    r(rows, width),
    z(rows, width),
    p(rows, width),
    ap(rows, width),
    az(rows, width),
    tmp(rows, width) {}

SparseMatrix& ColumnBatch::GetMatrix(const MatrixType matrix_type) {
  switch (matrix_type) {
    case MatrixType::M: return m;
    case MatrixType::R: return r;
    case MatrixType::Z: return z;
    case MatrixType::P: return p;
    case MatrixType::AP: return ap;
    case MatrixType::AZ: return az;
    case MatrixType::Tmp: return tmp;
  }
  throw std::invalid_argument("unknown column-batch matrix");
}

const SparseMatrix& ColumnBatch::GetMatrix(const MatrixType matrix_type) const {
  switch (matrix_type) {
    case MatrixType::M: return m;
    case MatrixType::R: return r;
    case MatrixType::Z: return z;
    case MatrixType::P: return p;
    case MatrixType::AP: return ap;
    case MatrixType::AZ: return az;
    case MatrixType::Tmp: return tmp;
  }
  throw std::invalid_argument("unknown column-batch matrix");
}

void CurrentStateManager::MultiplyBatches(
  const SparseMatrix& matrix, const MatrixType input_type, const MatrixType output_type
) {
  if (input_type == output_type) {
    throw std::invalid_argument("Batched multiplication needs distinct input and output matrices");
  }
  RunBatchesParallel(batches.size(), thread_count, [&](const int64_t batch_idx) {
    ColumnBatch& batch = batches[batch_idx];
    if (batch.GetMatrix(input_type).rows() != matrix.cols()) {
      throw std::invalid_argument("Sparse matrix dimensions do not match");
    }
    const SparseMatrix& input_matrix = batch.GetMatrix(input_type);
    SparseMatrix& output_matrix = batch.GetMatrix(output_type);
    output_matrix = matrix * input_matrix;
    output_matrix.makeCompressed();
  });
}

int64_t CurrentStateManager::GetMaxNonZeros() const {
  const double max_non_zeros = input_params.max_density * static_cast<double>(n) * n;
  return std::max(n, static_cast<int64_t>(std::floor(max_non_zeros)));
}

int64_t CurrentStateManager::CountNonZeros(const MatrixType matrix_type) const {
  return std::accumulate(batches.begin(), batches.end(), int64_t{0},
    [matrix_type](const int64_t total, const ColumnBatch& batch) {
      return total + batch.GetMatrix(matrix_type).nonZeros();
    }
  );
}

void CurrentStateManager::StabilizeResiduals() {
  std::vector<double> norms_squared(batches.size(), 0.0);
  RunBatchesParallel(batches.size(), thread_count, [&](const int64_t index) {
    norms_squared[index] = batches[index].r.squaredNorm();
  });
  const double residual_norm = std::sqrt(std::accumulate(norms_squared.begin(), norms_squared.end(), 0.0));
  const double roundoff_floor = 64.0 * std::numeric_limits<double>::epsilon() * std::sqrt(static_cast<double>(n));
  const double refresh_threshold = std::max(10.0 * input_params.tolerance, roundoff_floor);
  if (std::isfinite(residual_norm) && residual_norm > refresh_threshold) {
    return;
  }
  RunBatchesParallel(batches.size(), thread_count, [&](const int64_t index) {
    CalculateResidualForBatch(batches[index]);
  });
}

SparseMatrix CurrentStateManager::AssembleM() {
  // prepare offsets
  std::vector<int64_t> offsets(batches.size() + 1, 0);
  RunBatchesParallel(batches.size(), thread_count, [&](const int64_t index) {
    batches[index].m.makeCompressed(); // if it's already compressted, then O(1) operation
  });
  for (int64_t i = 0; i < static_cast<int64_t>(batches.size()); ++i) {
    offsets[i + 1] = offsets[i] + batches[i].m.nonZeros();
  }

  // copy data to the result matrix with proper offsets
  SparseMatrix result(n, n);
  result.resizeNonZeros(offsets.back());
  RunBatchesParallel(batches.size(), thread_count, [&](const int64_t index) {
    const ColumnBatch& batch = batches[index];
    const int64_t offset = offsets[index];
    for (int64_t i = 0; i < batch.Width(); ++i) {
      result.outerIndexPtr()[batch.first_column + i] = offset + batch.m.outerIndexPtr()[i];
    }
    if (batch.m.nonZeros() != 0) {
      std::copy_n(batch.m.innerIndexPtr(), batch.m.nonZeros(), result.innerIndexPtr() + offset);
      std::copy_n(batch.m.valuePtr(), batch.m.nonZeros(), result.valuePtr() + offset);
    }
  });
  result.outerIndexPtr()[n] = offsets.back();
  return result;
}

}  // namespace methods::common
