#pragma once

#include "methods_common.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace methods::inner_outer {

namespace {

namespace mc = methods::common;

const int64_t kInnerIterationsPerOuter = 2;

inline std::pair<
  std::vector<std::vector<double>>,
  std::vector<std::vector<double>>
> GetCoeffs(const mc::CurrentStateManager& p, const bool locally_optimal) {
  std::vector<std::vector<double>> coefs(p.batches.size());
  std::vector<std::vector<double>> lomr_coefs(p.batches.size());
  for (int64_t i = 0; i < static_cast<int64_t>(p.batches.size()); ++i) {
    const auto width = p.batches[i].Width();
    coefs[i].resize(width);
    if (locally_optimal) {
      lomr_coefs[i].resize(width);
    }
  }
  return {std::move(coefs), std::move(lomr_coefs)};
}

inline void RunInnerIterationForBatch(
  mc::ColumnBatch& batch, std::vector<double>& first, std::vector<double>* second
) {
  if (second != nullptr) {
    std::fill(second->begin(), second->end(), 0.0);
  }

  for (int64_t i = 0; i < batch.Width(); ++i) {
    const auto residual = batch.r.col(i);
    const auto image = batch.az.col(i);
    if (second == nullptr || batch.p.col(i).nonZeros() == 0) {
      first[i] = mc::GetSafeQuotient(residual.dot(image), image.squaredNorm());
      continue;
    }

    const auto previous_image = batch.ap.col(i);
    const double coupling = image.dot(previous_image);
    Eigen::Matrix2d normal;
    normal << image.squaredNorm(), coupling,
              coupling, previous_image.squaredNorm();
    const auto& coefficients = mc::SolveLeastSquares2x2(
      normal, Eigen::Vector2d(residual.dot(image), residual.dot(previous_image))
    );
    first[i] = coefficients[0];
    (*second)[i] = coefficients[1];
  }

  if (second == nullptr) {
    batch.p = batch.z;
    mc::MulScalarColumnwise(batch.p, first);
    batch.p.prune(1.0, mc::kPruneValueThreshold);

    batch.ap = batch.az;
    mc::MulScalarColumnwise(batch.ap, first);
    batch.ap.prune(1.0, mc::kPruneValueThreshold);
    return;
  }
  
  auto z = batch.z;
  mc::MulScalarColumnwise(z, first);
  mc::MulScalarColumnwise(batch.p, *second);
  batch.p += z;
  batch.p.prune(1.0, mc::kPruneValueThreshold);

  auto az = batch.az;
  mc::MulScalarColumnwise(az, first);
  mc::MulScalarColumnwise(batch.ap, *second);
  batch.ap += az;
  batch.ap.prune(1.0, mc::kPruneValueThreshold);
}

// returns true if algorithm is finished
inline bool RunInnerIteration(
  mc::CurrentStateManager& p,
  mc::OutputResult& result,
  const mc::SparseMatrix& outer_preconditioner,
  std::vector<std::vector<double>>& coefs,
  std::vector<std::vector<double>>& lomr_coefs,
  const mc::InputParams& input_params,
  const bool locally_optimal
) {
  p.MultiplyBatches(outer_preconditioner, mc::MatrixType::R, mc::MatrixType::Z);
  p.MultiplyBatches(p.apply_a, mc::MatrixType::Z, mc::MatrixType::AZ);
  mc::RunBatchesParallel(p.batches.size(), p.thread_count, [&](const int64_t index) {
    RunInnerIterationForBatch(p.batches[index], coefs[index], locally_optimal ? &lomr_coefs[index] : nullptr);
  });

  if (p.DropForMatrixType(mc::MatrixType::P)) {
    p.MultiplyBatches(p.apply_a, mc::MatrixType::P, mc::MatrixType::AP);
  }
  mc::RunBatchesParallel(p.batches.size(), p.thread_count, [&](const int64_t index) {
    mc::AddScaled(p.batches[index].m, p.batches[index].p, 1.0);
  });
  p.UpdateResidualsAfterDropping([](mc::ColumnBatch& batch) {
    mc::AddScaled(batch.r, batch.ap, -1.0);
  });

  if (result.AppendIteration(input_params, p.batches, p.n, p.thread_count, mc::MatrixType::P)) {
    result.M = p.AssembleM();
    return true;
  }
  return false;
}

inline mc::OutputResult RunMinimalResidual(
  const mc::InputParams& input_params, const int64_t num_threads, const std::string& method_name, const bool locally_optimal
) {
  // R_0 = I_n - A * M_0 and precompute
  auto [p, result] = mc::PrepareStateManager(input_params, num_threads, method_name, false);
  if (result.AppendIteration(input_params, p.batches, p.n, p.thread_count, mc::MatrixType::P)) {
    result.M = p.AssembleM();
    return result;
  }

  auto [coefs, lomr_coefs] = GetCoeffs(p, locally_optimal);

  for (int64_t i = 0; i < input_params.max_iterations; ) {
    const auto& outer_preconditioner = p.AssembleM();
    if (locally_optimal) {
      mc::RunBatchesParallel(p.batches.size(), p.thread_count, [&](const int64_t index) {
        p.batches[index].p.setZero();
        p.batches[index].ap.setZero();
      });
    }

    const auto inner_iterations = std::min(kInnerIterationsPerOuter, input_params.max_iterations - i);
    for (int64_t j = 0; j < inner_iterations; ++j, ++i) {
      if (RunInnerIteration(p, result, outer_preconditioner, coefs, lomr_coefs, input_params, locally_optimal)) {
        return result;
      }
    }
  }

  result.M = p.AssembleM();
  return result;
}

} // namespace

common::OutputResult RunLocallyOptimalMinimalResidual(const common::InputParams& input_params, const int64_t num_threads) {
  return RunMinimalResidual(input_params, num_threads, "inner_outer_lomr", true);
}

common::OutputResult RunMinimalResidual(const common::InputParams& input_params, const int64_t num_threads) {
  return RunMinimalResidual(input_params, num_threads, "inner_outer_mr", false);
}

common::OutputResult RunMinres(const common::InputParams& input_params, const int64_t num_threads);

}  // namespace methods::inner_outer
