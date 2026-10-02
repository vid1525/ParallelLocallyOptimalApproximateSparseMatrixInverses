#include "methods_common.hpp"

#include <cmath>
#include <numeric>
#include <vector>

namespace methods::global_spai {

namespace {

namespace mc = methods::common;

} // namespace

mc::OutputResult RunConjugateGradient(const mc::InputParams& input_params, const int64_t num_threads) {
  // R_0 = I_n - A * M_0 and precompute
  auto [p, result] = mc::PrepareStateManager(input_params, num_threads, "cg");  

  std::vector<double> rz_dot(p.batches.size(), 0.0);
  p.MultiplyBatches(p.apply_pr, mc::MatrixType::R, mc::MatrixType::Z);
  mc::RunBatchesParallel(p.batches.size(), p.thread_count, [&](const int64_t index) {
    auto& batch = p.batches[index];
    batch.p = batch.z;
    rz_dot[index] = mc::FrobeniusDot(batch.r, batch.z);
  });
  p.DropForMatrixType(mc::MatrixType::P);

  double r_z = std::accumulate(rz_dot.begin(), rz_dot.end(), 0.0);
  if (result.AppendIteration(input_params, p.batches, p.n, p.thread_count, mc::MatrixType::P)) {
    result.M = p.AssembleM();
    return result;
  }
  p.MultiplyBatches(p.apply_a, mc::MatrixType::P, mc::MatrixType::AP);

  for (int64_t i = 1; i <= input_params.max_iterations; ++i) {
    mc::RunBatchesParallel(p.batches.size(), p.thread_count, [&](const int64_t index) {
      rz_dot[index] = mc::FrobeniusDot(p.batches[index].p, p.batches[index].ap);
    });
    const double denominator = std::accumulate(rz_dot.begin(), rz_dot.end(), 0.0);
    if (!mc::IsFiniteNonZero(r_z, 0.0) || !mc::IsFiniteNonZero(denominator, 0.0)) {
      break;
    }

    // alpha_i
    const auto alpha = r_z / denominator;
    if (!std::isfinite(alpha)) {
      break;
    }
    const double previous_r_z = r_z;

    mc::RunBatchesParallel(p.batches.size(), p.thread_count, [&](const int64_t index) {
      mc::AddScaled(p.batches[index].m, p.batches[index].p, alpha, false);
    });
    p.UpdateResidualsAfterDropping([&](mc::ColumnBatch& batch) {
      mc::AddScaled(batch.r, batch.ap, -alpha, false);
    }, false);
    if (result.AppendIteration(input_params, p.batches, p.n, p.thread_count, mc::MatrixType::P)) {
      break;
    }

    p.MultiplyBatches(p.apply_pr, mc::MatrixType::R, mc::MatrixType::Z);
    mc::RunBatchesParallel(p.batches.size(), p.thread_count, [&](const int64_t index) {
      rz_dot[index] = mc::FrobeniusDot(p.batches[index].r, p.batches[index].z);
    });

    // beta_i+1
    r_z = std::accumulate(rz_dot.begin(), rz_dot.end(), 0.0);
    const double beta = r_z / previous_r_z;
    if (!std::isfinite(beta)) {
      break;
    }

    mc::RunBatchesParallel(p.batches.size(), p.thread_count, [&](const int64_t index) {
      auto& batch = p.batches[index];
      mc::LinearCombination(batch.z, 1.0, batch.p, beta, false);
    });
    p.DropForMatrixType(mc::MatrixType::P);
    p.MultiplyBatches(p.apply_a, mc::MatrixType::P, mc::MatrixType::AP);
  }

  result.M = p.AssembleM();
  return result;
}

}  // namespace methods::global_spai
