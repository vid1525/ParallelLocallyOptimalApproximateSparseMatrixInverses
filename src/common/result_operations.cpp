#include "methods_common.hpp"

#include <cmath>

namespace methods::common {

bool OutputResult::IsStopCriterionReached(const InputParams& input_params, const SingleIterationData& data) {
  return !input_params.enable_dropping && data.density_m >= input_params.max_density - EPS;
}

bool OutputResult::IsConverged(const InputParams& input_params, const SingleIterationData& data) {
  return data.residual_norm < input_params.tolerance * initial_residual_norm;
}

void OutputResult::AppendIterationData(
  const int64_t iteration,
  const std::vector<ColumnBatch>& batches,
  const int64_t n,
  const int64_t thread_count,
  const MatrixType direction
) {
  const bool is_initial_iteration = history.empty();
  struct BatchMetrics {
    double residual_squared = 0.0;
    int64_t m_nonzeros = 0;
    int64_t direction_nonzeros = 0;
  };

  std::vector<BatchMetrics> metrics(batches.size());
  RunBatchesParallel(batches.size(), thread_count, [&](const int64_t index) {
    metrics[index] = {
      is_initial_iteration ? 0.0 : batches[index].r.squaredNorm(),
      batches[index].m.nonZeros(),
      batches[index].GetMatrix(direction).nonZeros(),
    };
  });

  BatchMetrics totals;
  for (const BatchMetrics& batch : metrics) {
    totals.residual_squared += batch.residual_squared;
    totals.m_nonzeros += batch.m_nonzeros;
    totals.direction_nonzeros += batch.direction_nonzeros;
  }
  const double entries = static_cast<double>(n) * static_cast<double>(n);
  history.emplace_back(SingleIterationData(
    iteration,
    is_initial_iteration ? initial_residual_norm : std::sqrt(totals.residual_squared),
    static_cast<double>(totals.m_nonzeros) / entries,
    static_cast<double>(totals.direction_nonzeros) / entries
  ));
}

bool OutputResult::AppendIteration(
  const InputParams& input_params,
  const std::vector<ColumnBatch>& batches,
  const int64_t n,
  const int64_t thread_count,
  const MatrixType direction
) {
  const auto iter = static_cast<int64_t>(history.size());
  iterations = iter;
  AppendIterationData(iter, batches, n, thread_count, direction);
  if (IsConverged(input_params, history.back())) {
    return converged = true;
  }
  return IsStopCriterionReached(input_params, history.back());
}

}  // namespace methods::common
