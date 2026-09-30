#pragma once

#include "methods_common.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace methods::inner_outer {

namespace mc = methods::common;

inline constexpr int64_t kInnerIterationsPerOuter = 2;

inline mc::OutputResult Run(const mc::InputParams& params, const int64_t num_threads, const bool locally_optimal) {
  auto [state, result] = mc::PrepareStateManager(
    params, num_threads, locally_optimal ? "inner_outer_lomr" : "inner_outer_mr", false
  );
  if (result.AppendIteration(params, state.batches, state.n, state.thread_count, mc::MatrixType::P)) {
    result.M = state.AssembleM();
    return result;
  }

  std::vector<mc::SparseMatrix> q;
  std::vector<std::vector<double>> ratios;
  for (const auto& batch : state.batches) {
    q.emplace_back(state.n, batch.Width());
    ratios.emplace_back(batch.Width(), 0.0);
  }

  for (int64_t iteration = 0; iteration < params.max_iterations;) {
    const mc::SparseMatrix outer_m = state.AssembleM();
    mc::RunBatchesParallel(state.batches.size(), state.thread_count, [&](const int64_t index) {
      state.CalculateResidualForBatch(state.batches[index]);
      q[index].setZero();
    });
    const auto inner_count = std::min(kInnerIterationsPerOuter, params.max_iterations - iteration);
    for (int64_t inner = 0; inner < inner_count; ++inner, ++iteration) {
      state.MultiplyBatches(outer_m, mc::MatrixType::R, mc::MatrixType::Z);
      state.MultiplyBatches(state.apply_a, mc::MatrixType::Z, mc::MatrixType::AZ);
      mc::RunBatchesParallel(state.batches.size(), state.thread_count, [&](const int64_t index) {
        auto& batch = state.batches[index];
        std::vector<double> delta(batch.Width());
        std::vector<double> gamma(batch.Width(), 0.0);
        if (locally_optimal) {
          batch.ap = state.apply_a * q[index];
        }
        for (int64_t col = 0; col < batch.Width(); ++col) {
          const auto az = batch.az.col(col);
          const auto r = batch.r.col(col);
          if (!locally_optimal || inner == 0) {
            delta[col] = mc::GetSafeQuotient(r.dot(az), az.squaredNorm());
            gamma[col] = locally_optimal ? 1.0 : 0.0;
          } else {
            const auto aq = batch.ap.col(col);
            const double coupling = az.dot(aq);
            Eigen::Matrix2d gram;
            gram << az.squaredNorm(), coupling, coupling, aq.squaredNorm();
            const Eigen::Vector2d coefficients = mc::SolveLeastSquares2x2(
              gram, Eigen::Vector2d(r.dot(az), r.dot(aq))
            );
            delta[col] = coefficients[0];
            gamma[col] = coefficients[1];
          }
          ratios[index][col] = mc::GetSafeQuotient(gamma[col], delta[col]);
        }
        batch.p = batch.z;
        mc::MulScalarColumnwise(batch.p, delta);
        if (locally_optimal) {
          auto previous_q = q[index];
          mc::MulScalarColumnwise(previous_q, gamma);
          mc::AddScaled(batch.p, previous_q, 1.0);
        }
      });

      state.DropForMatrixType(mc::MatrixType::P);
      state.MultiplyBatches(state.apply_a, mc::MatrixType::P, mc::MatrixType::AP);
      mc::RunBatchesParallel(state.batches.size(), state.thread_count, [&](const int64_t index) {
        mc::AddScaled(state.batches[index].m, state.batches[index].p, 1.0);
      });
      state.UpdateResidualsAfterDropping([](mc::ColumnBatch& batch) {
        mc::AddScaled(batch.r, batch.ap, -1.0);
      });
      if (locally_optimal) {
        mc::RunBatchesParallel(state.batches.size(), state.thread_count, [&](const int64_t index) {
          mc::MulScalarColumnwise(q[index], ratios[index]);
          mc::AddScaled(q[index], state.batches[index].r, 1.0);
        });
      }
      if (result.AppendIteration(params, state.batches, state.n, state.thread_count, mc::MatrixType::P)) {
        result.M = state.AssembleM();
        return result;
      }
    }
  }
  result.M = state.AssembleM();
  return result;
}

}  // namespace methods::inner_outer
