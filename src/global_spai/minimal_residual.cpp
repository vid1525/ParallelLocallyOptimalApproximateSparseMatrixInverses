#include "methods_common.hpp"

#include <cmath>
#include <numeric>
#include <vector>

namespace methods::global_spai {

namespace {

namespace mc = methods::common;

struct MrProjection {
  double numerator = 0.0;
  double denominator = 0.0;
};

MrProjection Sum(const std::vector<MrProjection>& values) {
  return std::accumulate(values.begin(), values.end(), MrProjection{},
    [](const MrProjection& total, const MrProjection& value) {
      return MrProjection{
        total.numerator + value.numerator,
        total.denominator + value.denominator,
      };
    }
  );
}

}  // namespace

mc::OutputResult RunMinimalResidual(const mc::InputParams& input_params, const int64_t num_threads) {
  // R_0 = I_n - A * M_0 and precompute
  auto [p, result] = mc::PrepareStateManager(input_params, num_threads, "mr");

  // calc Z_0
  std::vector<MrProjection> rz_dot(p.batches.size());
  p.MultiplyBatches(p.apply_pr, mc::MatrixType::R, mc::MatrixType::Z);
  p.DropForMatrixType(mc::MatrixType::Z);
  if (result.AppendIteration(input_params, p.batches, p.n, p.thread_count, mc::MatrixType::Z)) {
    result.M = p.AssembleM();
    return result;
  }

  // numerator (Z_i, Pi * A * Z_i)_F and denominator ||Pi * A * Z_i||_F^2. 
  p.MultiplyBatches(p.apply_a, mc::MatrixType::Z, mc::MatrixType::AZ);
  p.MultiplyBatches(p.apply_pr, mc::MatrixType::AZ, mc::MatrixType::Tmp);
  mc::RunBatchesParallel(p.batches.size(), p.thread_count, [&](const int64_t index) {
    const auto& batch = p.batches[index];
    rz_dot[index] = MrProjection{
      .numerator = mc::FrobeniusDot(batch.z, batch.tmp),
      .denominator = batch.tmp.squaredNorm(),
    };
  });

  MrProjection projection = Sum(rz_dot);
  for (int64_t i = 1; i <= input_params.max_iterations; ++i) {
    if (!mc::IsFiniteNonZero(projection.denominator)) {
      break;
    }
    // alpha_i
    const double alpha = projection.numerator / projection.denominator;
    if (!std::isfinite(alpha)) {
      break;
    }

    mc::RunBatchesParallel(p.batches.size(), p.thread_count, [&](const int64_t index) {
      auto& batch = p.batches[index];
      mc::AddScaled(batch.m, batch.z, alpha);
    });
    p.UpdateResidualsAfterDropping([&](mc::ColumnBatch& batch) {
      mc::AddScaled(batch.r, batch.az, -alpha);
    });

    if (result.AppendIteration(input_params, p.batches, p.n, p.thread_count, mc::MatrixType::Z)) {
      break;
    }

    p.MultiplyBatches(p.apply_pr, mc::MatrixType::R, mc::MatrixType::Z);
    p.DropForMatrixType(mc::MatrixType::Z);
    p.MultiplyBatches(p.apply_a, mc::MatrixType::Z, mc::MatrixType::AZ);
    p.MultiplyBatches(p.apply_pr, mc::MatrixType::AZ, mc::MatrixType::Tmp);
    mc::RunBatchesParallel(p.batches.size(), p.thread_count, [&](const int64_t index) {
      const auto& batch = p.batches[index];
      rz_dot[index] = MrProjection{
        .numerator = mc::FrobeniusDot(batch.z, batch.tmp),
        .denominator = batch.tmp.squaredNorm(),
      };
    });
    projection = Sum(rz_dot);
  }

  result.M = p.AssembleM();
  return result;
}

}  // namespace methods::global_spai
