#include "inner_outer_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace methods::inner_outer {

namespace {

namespace mc = methods::common;

const int64_t kMinresInnerIterationsPerOuter = 2;

struct MinresColumnState {
  double previous_beta = 0.0;
  double dbar = 0.0;
  double epsilon = 0.0;
  double phi_bar = 0.0;
  double cosine = -1.0;
  double sine = 0.0;
};

struct CoefHolder {
  std::vector<std::vector<MinresColumnState>> states;
  std::vector<mc::SparseMatrix> previous_lanczos_vectors;
  std::vector<std::vector<double>> alphas;
  std::vector<std::vector<double>> inverse_next_betas;
  std::vector<std::vector<double>> phis;
};

inline void InitIteration(
  std::vector<mc::ColumnBatch>& batches,
  std::vector<mc::SparseMatrix>& previous_lanczos_vectors,
  std::vector<std::vector<MinresColumnState>>& states,
  const int64_t thread_count
) {
  mc::RunBatchesParallel(batches.size(), thread_count, [&](const int64_t index) {
    auto& batch = batches[index];
    auto& batch_states = states[index];
    std::vector<double> inverse_norms(batch.Width(), 0.0);

    batch.p.setZero();
    batch.ap.setZero();
    batch.tmp.setZero();
    previous_lanczos_vectors[index].setZero();
    for (int64_t i = 0; i < batch.Width(); ++i) {
      const double beta = batch.r.col(i).norm();
      batch_states[i] = MinresColumnState{.phi_bar = beta};
      inverse_norms[i] = mc::GetSafeQuotient(1.0, beta);
    }
    batch.z = batch.r;
    mc::MulScalarColumnwise(batch.z, inverse_norms);
  });
}

inline CoefHolder PrepareCoefs(const mc::CurrentStateManager& p) {
  const int64_t batches_size = static_cast<int64_t>(p.batches.size());
  CoefHolder h;
  h.states.resize(batches_size);
  h.previous_lanczos_vectors.reserve(batches_size);
  h.alphas.resize(batches_size);
  h.inverse_next_betas.resize(batches_size);
  h.phis.resize(batches_size);
  for (int64_t index = 0; index < batches_size; ++index) {
    const auto width = p.batches[index].Width();
    h.previous_lanczos_vectors.emplace_back(p.n, width);
    h.states[index].resize(width);
    h.alphas[index].resize(width);
    h.inverse_next_betas[index].resize(width);
    h.phis[index].resize(width);
  }
  return h;
}

inline void FillBatchTmp(mc::ColumnBatch& batch, CoefHolder& h, const int64_t index) {
  auto& batch_states = h.states[index];
  auto& batch_alphas = h.alphas[index];
  for (int64_t i = 0; i < batch.Width(); ++i) {
    batch_alphas[i] = batch.z.col(i).dot(batch.az.col(i));
  }
  std::vector<double> negative_alphas(batch.Width(), 0.0);
  std::vector<double> negative_previous_betas(batch.Width(), 0.0);
  for (int64_t i = 0; i < batch.Width(); ++i) {
    negative_alphas[i] = -batch_alphas[i];
    negative_previous_betas[i] = -batch_states[i].previous_beta;
  }

  auto tmp_matrix = batch.z;
  mc::MulScalarColumnwise(tmp_matrix, negative_alphas);
  batch.tmp = h.previous_lanczos_vectors[index];
  mc::MulScalarColumnwise(batch.tmp, negative_previous_betas);
  batch.tmp += tmp_matrix + batch.az;
  batch.tmp.prune(1.0, mc::kPruneValueThreshold);
}

inline std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> GetCoefForBatch(
  mc::ColumnBatch& batch, CoefHolder& h, const int64_t index
) {
  auto& batch_states = h.states[index];
  auto& batch_inverse_betas = h.inverse_next_betas[index];
  auto& batch_phis = h.phis[index];
  std::vector<double> inverse_gammas(batch.Width(), 0.0);
  std::vector<double> previous_direction_coefficients(batch.Width(), 0.0);
  std::vector<double> older_direction_coefficients(batch.Width(), 0.0);

  for (int64_t i = 0; i < batch.Width(); ++i) {
    auto& state = batch_states[i];
    const auto next_beta = batch.tmp.col(i).norm();
    batch_inverse_betas[i] = mc::GetSafeQuotient(1.0, next_beta);

    const auto old_epsilon = state.epsilon;
    const auto delta = state.cosine * state.dbar + state.sine * h.alphas[index][i];
    const auto gbar = state.sine * state.dbar - state.cosine * h.alphas[index][i];
    state.epsilon = state.sine * next_beta;
    state.dbar = -state.cosine * next_beta;

    const auto gamma = std::hypot(gbar, next_beta);
    const auto inverse_gamma = mc::GetSafeQuotient(1.0, gamma);
    state.cosine = gbar * inverse_gamma;
    state.sine = next_beta * inverse_gamma;
    batch_phis[i] = state.cosine * state.phi_bar;
    state.phi_bar *= state.sine;

    inverse_gammas[i] = inverse_gamma;
    previous_direction_coefficients[i] = -delta * inverse_gamma;
    older_direction_coefficients[i] = -old_epsilon * inverse_gamma;
    state.previous_beta = next_beta;
  }
  
  return {std::move(inverse_gammas), std::move(previous_direction_coefficients), std::move(older_direction_coefficients)};
}

inline bool RunInnerIteration(
  mc::CurrentStateManager& p, mc::OutputResult& result, CoefHolder& h, const mc::InputParams& input_params
) {
  p.MultiplyBatches(p.apply_a, mc::MatrixType::Z, mc::MatrixType::AZ);
  mc::RunBatchesParallel(p.batches.size(), p.thread_count, [&](const int64_t index) {
    auto& batch = p.batches[index];    
    FillBatchTmp(batch, h, index);
    const auto& [inv_gammas, prev_dir_coefs, old_dir_coefs] = GetCoefForBatch(batch, h, index);

    auto direction = batch.z;
    mc::MulScalarColumnwise(direction, inv_gammas);
    auto tmp = batch.p;
    mc::MulScalarColumnwise(tmp, prev_dir_coefs);
    direction += tmp;

    tmp = batch.ap;
    mc::MulScalarColumnwise(tmp, old_dir_coefs);
    direction += tmp;
    direction.prune(1.0, mc::kPruneValueThreshold);

    batch.ap = std::move(batch.p);
    batch.p = std::move(direction);
  });

  p.DropForMatrixType(mc::MatrixType::P);
  p.MultiplyBatches(p.apply_a, mc::MatrixType::P, mc::MatrixType::AZ);
  mc::RunBatchesParallel(p.batches.size(), p.thread_count, [&](const int64_t index) {
    auto& batch = p.batches[index];
    auto corr = batch.p;
    mc::MulScalarColumnwise(corr, h.phis[index]);
    mc::AddScaled(batch.m, corr, 1.0);
  });
  p.UpdateResidualsAfterDropping([&](mc::ColumnBatch& batch) {
      const int64_t index = &batch - p.batches.data();
      auto r_corr = batch.az;
      mc::MulScalarColumnwise(r_corr, h.phis[index]);
      mc::AddScaled(batch.r, r_corr, -1.0);
    }
  );

  if (result.AppendIteration(input_params, p.batches, p.n, p.thread_count, mc::MatrixType::P)) {
    result.M = p.AssembleM();
    return true;
  }

  mc::RunBatchesParallel(p.batches.size(), p.thread_count, [&](const int64_t index) {
    auto& batch = p.batches[index];
    h.previous_lanczos_vectors[index] = std::move(batch.z);
    batch.z = batch.tmp;
    mc::MulScalarColumnwise(batch.z, h.inverse_next_betas[index]);
  });
  return false;
}

}  // namespace

mc::OutputResult RunMinres(const mc::InputParams& input_params, const int64_t num_threads) {
  // R_0 = I_n - A * M_0 and precompute
  auto [p, result] = mc::PrepareStateManager(input_params, num_threads, "inner_outer_minres", true);
  auto h = PrepareCoefs(p);

  if (result.AppendIteration(input_params, p.batches, p.n, p.thread_count, mc::MatrixType::P)) {
    result.M = p.AssembleM();
    return result;
  }

  for (int64_t i = 0; i < input_params.max_iterations; ) {
    InitIteration(p.batches, h.previous_lanczos_vectors, h.states, p.thread_count);
    const auto inner_terations = std::min(kMinresInnerIterationsPerOuter, input_params.max_iterations - i);
    for (int64_t j = 0; j < inner_terations; ++j, ++i) {
      if (RunInnerIteration(p, result, h, input_params)) {
        return result;
      }
    }
  }

  result.M = p.AssembleM();
  return result;
}

}  // namespace methods::inner_outer
