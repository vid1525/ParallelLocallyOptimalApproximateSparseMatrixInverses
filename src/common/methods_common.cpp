#include "methods_common.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace methods::common {
namespace {

void ValidateProblem(const InputParams& input_params) {
  const SparseMatrix& A = input_params.A;
  if (A.rows() != A.cols() || A.rows() == 0) {
    throw std::invalid_argument("A must be a non-empty square matrix");
  }
  if (input_params.M0.size() != 0 && (input_params.M0.rows() != A.rows() || input_params.M0.cols() != A.cols())) {
    throw std::invalid_argument("Initial preconditioner M0 must have the same shape as A");
  }
  if (input_params.Pr.size() != 0 && (input_params.Pr.rows() != A.rows() || input_params.Pr.cols() != A.cols())) {
    throw std::invalid_argument("Pr must have the same shape as A");
  }
  if (input_params.max_iterations <= 0) {
    throw std::invalid_argument("max_iterations must be positive");
  }
  if (!std::isfinite(input_params.tolerance) || input_params.tolerance < 0.0) {
    throw std::invalid_argument("tolerance must be finite and non-negative");
  }
  if (!std::isfinite(input_params.max_density) || input_params.max_density <= EPS || input_params.max_density > 1.0) {
    throw std::invalid_argument("max_density must be finite and in the interval (0, 1]");
  }
  const auto max_entires_count = input_params.max_density * A.rows() * A.rows();
  if (input_params.enable_dropping && max_entires_count < static_cast<double>(A.rows())) {
    throw std::invalid_argument("max_density is too small to preserve the diagonal of M");
  }
}

int64_t GetThreadCount(const int64_t requested, const int64_t columns) {
  if (requested <= 0) {
    throw std::invalid_argument("Invalid threads count value");
  }
  const auto thread_count = std::min(requested, columns);
  if (thread_count > std::numeric_limits<int64_t>::max()) {
    throw std::invalid_argument("Thread count is more than the Eigen/OpenMP limit");
  }
  Eigen::setNbThreads(thread_count);
  return thread_count;
}

SparseMatrix GetDiagonalInversePreconditioner(const SparseMatrix& A) {
  if (A.rows() != A.cols()) {
    throw std::invalid_argument("Matrix A must be square to build a diagonal preconditioner");
  }
  const Eigen::VectorXd diagonal = A.diagonal();
  if (!diagonal.allFinite() || (diagonal.array() < std::numeric_limits<double>::epsilon()).any()) {
    throw std::invalid_argument("A has an invalid diagonal entry; provide Pr explicitly");
  }
  const Eigen::VectorXd inverse = diagonal.cwiseInverse();
  if (!inverse.allFinite()) {
    throw std::invalid_argument("A diagonal cannot be inverted safely; provide Pr explicitly");
  }
  return SparseMatrix(inverse.asDiagonal());
}

std::vector<double> GetColumnNormsSquared(
  const SparseMatrix& A, const int64_t thread_count, const bool enable_dropping
) {
  if (!enable_dropping) {
    return std::vector<double>(A.cols(), 0.0);
  }
  std::vector<double> a_column_norms_squared(A.cols(), 0.0);
  RunBatchesParallel(A.cols(), thread_count, [&](const int64_t col_idx) {
    a_column_norms_squared[col_idx] = A.col(col_idx).squaredNorm();
  });
  return a_column_norms_squared;
}

std::vector<ColumnBatch> InitializeBatches(
  const InputParams& input_params,
  const int64_t batch_count,
  const int64_t n,
  const int64_t thread_count
) {
  std::vector<ColumnBatch> batches;
  batches.reserve(batch_count);
  for (int64_t i = 0; i < batch_count; ++i) {
    const auto first_col = n * i / batch_count;
    const auto last_col = n * (i + 1) / batch_count;
    batches.emplace_back(n, first_col, last_col - first_col);
  }

  const auto has_inital_preconditioner = (input_params.M0.size() != 0);
  RunBatchesParallel(batch_count, thread_count, [&](const int64_t batch_idx) {
    auto& batch = batches[batch_idx]; 
    if (has_inital_preconditioner) {
      batch.m = input_params.M0.innerVectors(batch.first_column, batch.Width());
      return;
    }

    batch.m.reserve(batch.Width());
    for (int64_t i = 0; i < batch.Width(); ++i) {
      batch.m.startVec(i);
      batch.m.insertBack(batch.first_column + i, i) = 1.0;
    }
    batch.m.finalize();
  });
  return batches;
}

SparseMatrix GetPreconditioner(const InputParams& input_params) {
  if (input_params.Pr.size() != 0) {
    return input_params.Pr;
  }
  return GetDiagonalInversePreconditioner(input_params.A);
}

double GetResidualNorm(const CurrentStateManager& p) {
  std::vector<double> squared_norms(p.batches.size(), 0.0);
  RunBatchesParallel(p.batches.size(), p.thread_count, [&](const int64_t index) {
    squared_norms[index] = p.batches[index].r.squaredNorm();
  });
  double squared_norm = 0.0;
  for (const double value : squared_norms) {
    squared_norm += value;
  }
  return std::sqrt(squared_norm);
}

}  // namespace

std::pair<CurrentStateManager, OutputResult> PrepareStateManager(
  const InputParams& input_params,
  const int64_t num_threads,
  const std::string& method_name,
  const bool preserve_symmetry
) {
  ValidateProblem(input_params);

  const int64_t n = input_params.A.cols();
  const auto thread_count = GetThreadCount(num_threads, n);
  const auto batch_count = std::min(n, thread_count);
  auto p = CurrentStateManager{
    .input_params = input_params,
    .batches = InitializeBatches(input_params, batch_count, n, thread_count),
    .a_column_norms_squared = GetColumnNormsSquared(input_params.A, thread_count, input_params.enable_dropping),
    .n = n,
    .thread_count = thread_count,
    .preserve_symmetry = preserve_symmetry,
    .apply_pr = GetPreconditioner(input_params),
    .apply_a = input_params.A
  };

  if (input_params.enable_dropping) {
    p.ApplyDroppingStrategy();
  } else {
    RunBatchesParallel(batch_count, thread_count, [&](const int64_t index) {
      p.CalculateResidualForBatch(p.batches[index]);
    });
  }
  const auto initial_residual_norm = GetResidualNorm(p);
  return {std::move(p), OutputResult(method_name, initial_residual_norm)};
}

}  // namespace methods::common
