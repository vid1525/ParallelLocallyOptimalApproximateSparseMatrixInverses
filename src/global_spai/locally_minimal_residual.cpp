#include "methods_common.hpp"

#include <cmath>
#include <numeric>
#include <vector>

namespace methods::global_spai {

namespace mc = methods::common;

namespace {

struct LomrProjection {
  double az_praz = 0.0;
  double ap_prap = 0.0;
  double az_prap = 0.0;
  double z_az = 0.0;
  double z_ap = 0.0;
};

LomrProjection Sum(const std::vector<LomrProjection>& values) {
  return std::accumulate(values.begin(), values.end(), LomrProjection{},
    [](const LomrProjection& total, const LomrProjection& value) {
      return LomrProjection{
        total.az_praz + value.az_praz,
        total.ap_prap + value.ap_prap,
        total.az_prap + value.az_prap,
        total.z_az + value.z_az,
        total.z_ap + value.z_ap,
      };
    }
  );
}

// [delta, gamma, is_break]
std::tuple<double, double, bool> GetDeltaGamma(const std::vector<LomrProjection>& projections, const int64_t i) {
  const auto& p = Sum(projections);
  if (i == 1) {
    if (!mc::IsFiniteNonZero(p.az_praz, 0.0)) {
      return {-1, -1, true};
    }
    return {p.z_az / p.az_praz, 1.0, false};
  }

  Eigen::Matrix2d n;
  n << p.az_praz, p.az_prap,
        p.az_prap, p.ap_prap;
  const auto coef = mc::SolveLeastSquares2x2(n, Eigen::Vector2d(p.z_az, p.z_ap));
  return {coef[0], coef[1], false};
}

}  // namespace

mc::OutputResult RunLocallyOptimalMinimalResidual(const mc::InputParams& input_params, const int64_t num_threads) {
  // R_0 = I_n - A * M_0 and precompute
  auto [p, result] = mc::PrepareStateManager(input_params, num_threads, "lomr");

  std::vector<LomrProjection> projections(p.batches.size());
  p.MultiplyBatches(p.apply_pr, mc::MatrixType::R, mc::MatrixType::Z);
  if (result.AppendIteration(input_params, p.batches, p.n, p.thread_count, mc::MatrixType::Z)) {
    result.M = p.AssembleM();
    return result;
  }

  for (int64_t i = 1; i <= input_params.max_iterations; ++i) {
    p.MultiplyBatches(p.apply_a, mc::MatrixType::P, mc::MatrixType::AP);
    p.MultiplyBatches(p.apply_a, mc::MatrixType::Z, mc::MatrixType::AZ);
    p.MultiplyBatches(p.apply_pr, mc::MatrixType::AZ, mc::MatrixType::Tmp);
    mc::RunBatchesParallel(p.batches.size(), p.thread_count, [&](const int64_t index) {
      const auto& batch = p.batches[index];
      projections[index] = LomrProjection{
        .az_praz = mc::FrobeniusDot(batch.az, batch.tmp),
        .ap_prap = 0.0,
        .az_prap = 0.0,
        .z_az = mc::FrobeniusDot(batch.z, batch.az),
        .z_ap = 0.0,
      };
    });
    if (i > 1) {
      p.MultiplyBatches(p.apply_pr, mc::MatrixType::AP, mc::MatrixType::Tmp);
      mc::RunBatchesParallel(p.batches.size(), p.thread_count, [&](const int64_t index) {
        const auto& batch = p.batches[index];
        projections[index].ap_prap = mc::FrobeniusDot(batch.ap, batch.tmp);
        projections[index].az_prap = mc::FrobeniusDot(batch.az, batch.tmp);
        projections[index].z_ap = mc::FrobeniusDot(batch.z, batch.ap);
      });
    }

    const auto [delta, gamma, do_break] = GetDeltaGamma(projections, i);
    if (do_break || !std::isfinite(delta) || !std::isfinite(gamma)) { break; }

    mc::RunBatchesParallel(p.batches.size(), p.thread_count, [&](const int64_t index) {
      auto& batch = p.batches[index];
      // update M with the old P after - store the normalized direction P_i = Z_i + (gamma / delta) P_{i-1}
      mc::AddScaled(batch.m, batch.z, delta, false);
      mc::AddScaled(batch.m, batch.p, gamma, false);
    });
    p.UpdateResidualsAfterDropping([&](mc::ColumnBatch& batch) {
      mc::AddScaled(batch.r, batch.az, -delta, false);
      mc::AddScaled(batch.r, batch.ap, -gamma, false);
    }, false);
    const double ratio = delta != 0.0 ? gamma / delta : 0.0;
    mc::RunBatchesParallel(p.batches.size(), p.thread_count, [&](const int64_t index) {
      auto& batch = p.batches[index];
      // Restart the direction if normalization would be undefined.
      mc::LinearCombination(batch.z, 1.0, batch.p, std::isfinite(ratio) ? ratio : 0.0, false);
    });
    p.DropForMatrixType(mc::MatrixType::P);

    if (result.AppendIteration(input_params, p.batches, p.n, p.thread_count, mc::MatrixType::P)) {
      break;
    }
    p.MultiplyBatches(p.apply_pr, mc::MatrixType::R, mc::MatrixType::Z);
  }

  result.M = p.AssembleM();
  return result;
}

}  // namespace methods::global_spai
