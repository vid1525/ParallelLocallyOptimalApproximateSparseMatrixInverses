#include "methods_common.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace methods::common {
namespace {

struct DropCandidate {
  double score = 0.0;
  int64_t offset = 0;
};

double GetApproximationDropScore(
  const ColumnBatch& batch,
  const int64_t local_column,
  const int64_t row,
  const double value,
  const double col_norm_squared
) {
  // m_{k,l}^2 ||A e_k||_2^2 + 2 m_{k,l} (A R)_{k,l}.
  const double result = value * value * col_norm_squared + 2.0 * value * batch.tmp.coeff(row, local_column);
  return std::isfinite(result) ? result : -std::numeric_limits<double>::infinity();
}

bool IsLess(const DropCandidate& lhs, const DropCandidate& rhs) {
  return lhs.score < rhs.score || (lhs.score == rhs.score && lhs.offset < rhs.offset);
}

bool ShouldDrop(const DropCandidate& candidate, const DropCandidate& cutoff) {
  return !IsLess(cutoff, candidate);
}

void GetCandidateOffsets(
  std::vector<int64_t>& offsets,
  const std::vector<ColumnBatch>& batches,
  const int64_t thread_count,
  const MatrixType matrix_type,
  const bool preserve_diagonal
) {
  assert(offsets.size() == batches.size() + 1);
  std::vector<int64_t> counts(batches.size(), 0);
  RunBatchesParallel(batches.size(), thread_count, [&](const int64_t batch_idx) {
    const auto& batch = batches[batch_idx];
    const auto& matrix = batch.GetMatrix(matrix_type);

    for (int64_t j = 0; j < matrix.cols(); ++j) {
      const auto global_idx = batch.first_column + j;
      for (SparseMatrix::InnerIterator entry(matrix, j); entry; ++entry) {
        counts[batch_idx] += static_cast<int64_t>(!preserve_diagonal || entry.row() != global_idx);
      }
    }
  });
  for (int64_t i = 0; i < static_cast<int64_t>(batches.size()); ++i) {
    offsets[i + 1] = offsets[i] + counts[i];
  }
}

void FindDropCandidatesForBatch(
  std::vector<DropCandidate>& candidates,
  const ColumnBatch& batch,
  int64_t offset,
  const std::vector<double>& a_column_norms_squared
) {
  for (int64_t i = 0; i < batch.Width(); ++i) {
    const int64_t global_col_idx = batch.first_column + i;
    for (SparseMatrix::InnerIterator entry(batch.m, i); entry; ++entry) {
      const auto row_idx = entry.row();
      if (row_idx == global_col_idx) {
        continue;
      }
      candidates[offset] = DropCandidate{
        .score = GetApproximationDropScore(batch, i, row_idx, entry.value(), a_column_norms_squared[row_idx]),
        .offset = offset,
      };
      ++offset;
    }
  }
}

DropCandidate GetDropCutoff(std::vector<DropCandidate>& candidates, const int64_t drop_count) {
  if (drop_count <= 0 || drop_count > static_cast<int64_t>(candidates.size())) {
    throw std::invalid_argument("Invalid global sparse dropping count");
  }
  auto cutoff = candidates.begin() + drop_count - 1;
  std::nth_element(candidates.begin(), cutoff, candidates.end(), IsLess);
  return *cutoff;
}

}  // namespace

void CurrentStateManager::CalculateResidualForBatch(ColumnBatch& batch) {
  batch.r = -1.0 * input_params.A * batch.m;
  for (int64_t i = 0; i < batch.Width(); ++i) {
    batch.r.coeffRef(batch.first_column + i, i) += 1.0;  // add identity correction
  }
  // recompute I - A*M without dropping residual entries
  batch.r.prune(0.0);
}

void CurrentStateManager::EnsureSymmetry() {
  if (!preserve_symmetry) {
    return;
  }
  // M := (M + M^T) / 2
  SparseMatrix M_approx = AssembleM();
  SparseMatrix M_T_approx = M_approx.transpose();
  RunBatchesParallel(batches.size(), thread_count, [&](const int64_t index) {
    ColumnBatch& batch = batches[index];
    batch.m = (
      0.5 * (M_approx.middleCols(batch.first_column, batch.Width()) + M_T_approx.middleCols(batch.first_column, batch.Width()))
    );
    batch.m.makeCompressed();
  });
}

void CurrentStateManager::ApplyDroppingStrategy() {
  EnsureSymmetry();
  RunBatchesParallel(batches.size(), thread_count, [&](const int64_t index) {
    auto& batch = batches[index];
    // protects diagonal entries from both dropping stages
    batch.m.prune([&](const int64_t row, const int64_t col, const double value) {
      return value != 0.0 && (row == batch.first_column + col || std::abs(value) >= kPruneValueThreshold);
    });
    CalculateResidualForBatch(batch);
  });

  // no density truncation if Nnz(M) <= max
  const auto max_nonzeros = GetMaxNonZeros();
  const auto nonzeros = CountNonZeros(MatrixType::M);
  if (nonzeros <= max_nonzeros) {
    return;
  }

  // prepare for dropping
  MultiplyBatches(input_params.A, MatrixType::R, MatrixType::Tmp);
  std::vector<int64_t> offsets(batches.size() + 1, 0);
  GetCandidateOffsets(offsets, batches, thread_count, MatrixType::M, true);
  std::vector<DropCandidate> candidates(offsets.back());
  RunBatchesParallel(batches.size(), thread_count, [&](const int64_t index) {
    FindDropCandidatesForBatch(candidates, batches[index], offsets[index], a_column_norms_squared);
  });

  // get values to be dropped
  const auto cutoff = GetDropCutoff(candidates, nonzeros - max_nonzeros);
  std::vector<DropCandidate>().swap(candidates); // free mem
  RunBatchesParallel(batches.size(), thread_count, [&](const int64_t index) {
    auto& batch = batches[index];
    auto offset = offsets[index];
    batch.m.prune(
      [&](const int64_t row, const int64_t col, const double value) {
        if (row == batch.first_column + col) {
          return true;
        }
        const DropCandidate candidate{
          .score = GetApproximationDropScore(batch, col, row, value, a_column_norms_squared[row]),
          .offset = offset,
        };
        ++offset;
        return !ShouldDrop(candidate, cutoff);
      }
    );
    assert(offset == offsets[index + 1]);
    CalculateResidualForBatch(batch);
  });
}

bool CurrentStateManager::DropForMatrixType(const MatrixType matrix_type) {
  if (!input_params.enable_dropping) {
    return false;
  }
  const auto original_nonzeros = CountNonZeros(matrix_type);
  RunBatchesParallel(batches.size(), thread_count, [&](const int64_t index) {
    batches[index].GetMatrix(matrix_type).prune(1.0, kPruneValueThreshold);
  });
  const auto max_nonzeros = GetMaxNonZeros();
  const auto nonzeros = CountNonZeros(matrix_type);
  if (nonzeros <= max_nonzeros) {
    return nonzeros != original_nonzeros;
  }

  // prepare for dropping
  std::vector<int64_t> offsets(batches.size() + 1, 0);
  GetCandidateOffsets(offsets, batches, thread_count, matrix_type, false);
  std::vector<DropCandidate> candidates(offsets.back());
  RunBatchesParallel(batches.size(), thread_count, [&](const int64_t index) {
    const SparseMatrix& matrix = batches[index].GetMatrix(matrix_type);
    int64_t offset = offsets[index];
    for (int64_t i = 0; i < matrix.cols(); ++i) {
      for (SparseMatrix::InnerIterator entry(matrix, i); entry; ++entry) {
        candidates[offset] = {std::abs(entry.value()), offset};
        ++offset;
      }
    }
    assert(offset == offsets[index + 1]);
  });

  // dropping smallest entries compared by abs value
  const DropCandidate cutoff = GetDropCutoff(candidates, nonzeros - max_nonzeros);
  std::vector<DropCandidate>().swap(candidates); // free mem
  RunBatchesParallel(batches.size(), thread_count, [&](const int64_t index) {
    auto& matrix = batches[index].GetMatrix(matrix_type);
    auto offset = offsets[index];
    matrix.prune(
      [&](const int64_t, const int64_t, const double value) {
        const DropCandidate candidate{std::abs(value), offset};
        ++offset;
        return !ShouldDrop(candidate, cutoff);
      }
    );
    assert(offset == offsets[index + 1]);
  });
  return true;
}

}  // namespace methods::common
