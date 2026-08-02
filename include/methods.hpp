#pragma once

#include <Eigen/Sparse>

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace methods {

namespace common {

  inline constexpr double EPS = 1e-9;

  using SparseMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor, int64_t>;

  struct ColumnBatch;

  enum class MatrixType;

  inline SparseMatrix EnsureCompressedSparseMatrix(SparseMatrix matrix = SparseMatrix()) {
    matrix.makeCompressed();
    return matrix;
  }

  struct InputParams {
    SparseMatrix A;
    SparseMatrix M0;
    SparseMatrix Pr;

    int64_t max_iterations = 100;
    double tolerance = 1e-9;
    double max_density = 0.03;
    bool enable_dropping = true;
    int64_t max_non_zeros = -1;

    InputParams(
      SparseMatrix A = SparseMatrix(),
      SparseMatrix M0 = SparseMatrix(),
      SparseMatrix Pr = SparseMatrix(),
      const int64_t max_iterations = 100,
      const double tolerance = 1e-9,
      const double max_density = 0.03,
      const bool enable_dropping = true
    )
      : A(EnsureCompressedSparseMatrix(std::move(A))),
        M0(EnsureCompressedSparseMatrix(std::move(M0))),
        Pr(EnsureCompressedSparseMatrix(std::move(Pr))),
        max_iterations(max_iterations),
        tolerance(tolerance),
        max_density(max_density),
        enable_dropping(enable_dropping),
        max_non_zeros(
          std::max(
            static_cast<int64_t>(A.rows()),
            static_cast<int64_t>(std::floor(max_density * A.rows() * A.rows()))
          )
        ) {}
  };

  struct SingleIterationData {
    int64_t iteration = 0;
    double residual_norm = 0.0;
    double density_m = 0.0;
    double density_direction = 0.0;

    SingleIterationData(
      int64_t iteration = 0,
      double residual_norm = 0.0,
      double density_m = 0.0,
      double density_direction = 0.0
    ) :
        iteration(iteration),
        residual_norm(residual_norm),
        density_m(density_m),
        density_direction(density_direction) {}
  };

  using IterationHistory = std::vector<SingleIterationData>;

  struct OutputResult {
  private:
    double initial_residual_norm = 0.0;

    bool IsStopCriterionReached(const InputParams& input_params, const SingleIterationData& data);

    bool IsConverged(const InputParams& input_params, const SingleIterationData& data);

    void AppendIterationData(
      const int64_t iteration,
      const std::vector<ColumnBatch>& batches,
      const int64_t n,
      const int64_t thread_count,
      const MatrixType direction
    );

  public:
    std::string method;
    SparseMatrix M;
    int64_t iterations = 0;
    bool converged = false;
    IterationHistory history;

    OutputResult() = default;

    OutputResult(std::string method, const double initial_residual_norm = 0.0)
      : initial_residual_norm(initial_residual_norm), method(std::move(method)) {}

    OutputResult(
      SparseMatrix M,
      const int64_t iterations = 0,
      const bool converged = false,
      std::string method = "",
      IterationHistory history = IterationHistory()
    ) : initial_residual_norm(history.empty() ? 0.0 : history.front().residual_norm),
        method(std::move(method)),
        M(EnsureCompressedSparseMatrix(std::move(M))),
        iterations(iterations),
        converged(converged),
        history(std::move(history)) {}

    // returns true if stop is reached
    bool AppendIteration(
      const InputParams& input_params,
      const std::vector<ColumnBatch>& batches,
      const int64_t n,
      const int64_t thread_count,
      const MatrixType direction
    );
  };

} // namespace methods::common

namespace global_spai {

  common::OutputResult RunConjugateGradient(const common::InputParams& params, const int64_t num_threads = 1);

  common::OutputResult RunMinimalResidual(const common::InputParams& params, const int64_t num_threads = 1);

  common::OutputResult RunLocallyOptimalMinimalResidual(const common::InputParams& params, const int64_t num_threads = 1);

} // namespace methods::global_spai

namespace inner_outer {

  common::OutputResult RunMinres(const common::InputParams& params, const int64_t num_threads = 1);

  common::OutputResult RunMinimalResidual(const common::InputParams& params, const int64_t num_threads = 1);

  common::OutputResult RunLocallyOptimalMinimalResidual(const common::InputParams& params, const int64_t num_threads = 1);

} // namespace methods::inner_outer

}  // namespace methods
