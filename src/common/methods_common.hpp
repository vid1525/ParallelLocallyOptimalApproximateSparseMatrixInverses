#pragma once

#include "methods.hpp"

#include <Eigen/Core>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace methods::common {

namespace {

template <typename TCallback>
void RunBatchesParallelCore(const int64_t batches_count, const int64_t thread_count, TCallback&& callback) {
  if (thread_count == 1) {
    for (int64_t batch_idx = 0; batch_idx < batches_count; ++batch_idx) {
      callback(batch_idx);
    }
    return;
  }
#pragma omp parallel for schedule(dynamic, 1) num_threads(thread_count)
  for (int64_t batch_idx = 0; batch_idx < batches_count; ++batch_idx) {
    callback(batch_idx);
  }
}

} // namespace

const double kPruneValueThreshold = std::numeric_limits<double>::epsilon();

enum class MatrixType {
  M,     // approximate inverse
  R,     // residual
  Z,     // preconditioned residual
  P,     // search direction
  AP,    // A * search direction
  AZ,    // A * preconditioned residual
  Tmp,   // current temporary storage
};

struct ColumnBatch {
  int64_t first_column = 0;
  SparseMatrix m;
  SparseMatrix r;
  SparseMatrix z;
  SparseMatrix p;
  SparseMatrix ap;
  SparseMatrix az;
  SparseMatrix tmp;

  ColumnBatch(const int64_t rows, const int64_t first, const int64_t width);

  int64_t Width() const { return m.cols(); }

  SparseMatrix& GetMatrix(const MatrixType matrix_type);

  const SparseMatrix& GetMatrix(const MatrixType matrix_type) const;
};

struct CurrentStateManager {
  const InputParams& input_params;
  std::vector<ColumnBatch> batches;
  std::vector<double> a_column_norms_squared;
  int64_t n = 0;
  int64_t thread_count = 1;
  bool preserve_symmetry = true;
  SparseMatrix apply_pr;
  const SparseMatrix& apply_a;

  void MultiplyBatches(const SparseMatrix& matrix, const MatrixType input_type, const MatrixType output_type);

  SparseMatrix AssembleM();

  void ApplyDroppingStrategy();

  bool DropForMatrixType(const MatrixType direction);

  void CalculateResidualForBatch(ColumnBatch& batch);

  template <typename TCallback>
  void UpdateResidualsAfterDropping(TCallback&& callback, const bool stabilize = true) {
    if (input_params.enable_dropping) {
      ApplyDroppingStrategy();
      return;
    }
    RunBatchesParallel(batches.size(), thread_count, [&](const int64_t batch_idx) {
      callback(batches[batch_idx]);
    });
    if (stabilize) { StabilizeResiduals(); }
  }

private:
  int64_t GetMaxNonZeros() const;

  void EnsureSymmetry();

  int64_t CountNonZeros(const MatrixType matrix_type) const;

  void StabilizeResiduals();
};

std::pair<CurrentStateManager, OutputResult> PrepareStateManager(
  const InputParams& input_params,
  const int64_t num_threads,
  const std::string& method_name,
  const bool preserve_symmetry = true
);

template <typename TCallback>
void RunBatchesParallel(const int64_t batches_count, const int64_t thread_count, TCallback&& callback) {
  // disable multiple threads in Eigen to avoid threads allocation overlapping
  Eigen::setNbThreads(1);
  RunBatchesParallelCore(
    batches_count, thread_count, std::forward<TCallback>(callback)
  );
  Eigen::setNbThreads(thread_count);
}

double GetSafeQuotient(double numerator, double denominator);

void LinearCombination(const SparseMatrix& x, const double a, SparseMatrix& y, const double b, const bool prune = false);

void MulScalarColumnwise(SparseMatrix& x, const std::vector<double>& a);

void AddScaled(SparseMatrix& target, const SparseMatrix& direction, const double coefficient, const bool prune = false);

double FrobeniusDot(const SparseMatrix& lhs, const SparseMatrix& rhs);

Eigen::Vector2d SolveLeastSquares2x2(const Eigen::Matrix2d& matrix, const Eigen::Vector2d& x);

bool IsFiniteNonZero(const double value, const double threshold = kPruneValueThreshold);


}  // namespace methods::common
