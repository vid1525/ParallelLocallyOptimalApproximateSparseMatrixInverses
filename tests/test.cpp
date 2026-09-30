#include "methods.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

namespace {

namespace mc = methods::common;
namespace mgs = methods::global_spai;
namespace mio = methods::inner_outer;

mc::SparseMatrix GetTestMatrixA(const int64_t n) {
  std::vector<Eigen::Triplet<double, int64_t>> entries;
  entries.reserve(3 * n);
  for (int64_t i = 0; i < n; ++i) {
    entries.emplace_back(i, i, 2.0);
    if (i > 0) {
      entries.emplace_back(i, i - 1, -1.0);
    }
    if (i + 1 < n) {
      entries.emplace_back(i, i + 1, -1.0);
    }
  }
  mc::SparseMatrix A(n, n);
  A.setFromTriplets(entries.begin(), entries.end());
  return A;
}

void CheckMethodCommon(const std::string& name, const mc::OutputResult& result, const double initial_residual) {
  assert(!result.history.empty());
  assert(static_cast<int64_t>(result.history.size()) == result.iterations + 1);
  assert(result.history.front().iteration == 0);
  assert(result.history.back().iteration == result.iterations);
  const auto final_residual = result.history.back().residual_norm;
  std::cout << name << " initial = " << initial_residual << " final = " << final_residual << " iterations=" << result.iterations << " ";
  assert(final_residual < initial_residual);
  std::cout << "[ok]\n";
}

void CheckNnzCommon(const std::string& name, const mc::OutputResult& result, const double max_density) {
  const int64_t n = result.M.rows();
  const int64_t max_nonzeros = std::max(n, static_cast<int64_t>(std::floor(max_density * n * n)));
  std::cout << name << " nnz(M) = " << result.M.nonZeros() << " max_nonzeros = " << max_nonzeros << " ";
  assert(result.M.nonZeros() <= max_nonzeros);
  assert(result.iterations == 3);
  std::cout << "[ok]\n";
}

void CheckDensityStop(const std::string& name, const mc::OutputResult& result, const double max_density) {
  assert(!result.history.empty());
  const auto density = result.history.back().density_m;
  std::cout << name << " density = " << density << " converged = " << result.converged << " iterations = " << result.iterations << " ";
  assert(!result.converged);
  assert(result.iterations > 0);
  assert(result.iterations < 8);
  assert(density >= max_density - mc::EPS);
  std::cout << "[ok]\n";
}

void CheckEquivalentResults(const std::string& name, const mc::OutputResult& one_thread, const mc::OutputResult& two_threads) {
  assert(!one_thread.history.empty());
  assert(!two_threads.history.empty());
  const auto diff = one_thread.M - two_threads.M;
  const auto matrix_diff = diff.norm();
  const auto residual_diff = std::fabs(one_thread.history.back().residual_norm - two_threads.history.back().residual_norm);
  std::cout << name << " matrix_diff = " << matrix_diff << " residual_diff = " << residual_diff << " ";
  assert(one_thread.iterations == two_threads.iterations);
  assert(matrix_diff < 1e-10);
  assert(residual_diff < 1e-10);
  std::cout << "[ok]\n";
}

template <typename T>
void CommonParallelizationTest(const std::string& name, const mc::InputParams& params, const T& method, const double initial_residual) {
  const auto method_p1 = method(params, 1);
  const auto method_p2 = method(params, 2);
  CheckMethodCommon(name + std::string("_p1"), method_p1, initial_residual);
  CheckMethodCommon(name + std::string("_p2"), method_p2, initial_residual);
  CheckEquivalentResults(name + std::string("_check"), method_p1, method_p2);
}

template <typename T>
void DroppingParallelizationTest(const std::string& name, const mc::InputParams& params, const T& method) {
  const auto method_p1 = method(params, 1);
  const auto method_p4 = method(params, 4);
  CheckNnzCommon(name + std::string("_p1"), method_p1, params.max_density);
  CheckNnzCommon(name + std::string("_p4"), method_p4, params.max_density);
  CheckEquivalentResults(name + std::string("_check"), method_p1, method_p4);
}

mc::SparseMatrix GetSparseIdentity(const int64_t n) {
  mc::SparseMatrix I(n, n);
  I.setIdentity();
  return I;
}

double GetResidualFrobeniusNorm(const mc::SparseMatrix& A, const mc::SparseMatrix& M) {
  if (A.rows() != A.cols() || M.rows() != A.cols() || M.cols() != A.rows()) {
    throw std::invalid_argument("A and M must be square matrices of the same shape");
  }
  return (GetSparseIdentity(A.rows()) - A * M).norm();
}

}  // namespace

int main() {
  const auto A = GetTestMatrixA(12);
  assert(A.outerSize() == A.cols());
  assert(A.isCompressed());
  mc::InputParams params(A, GetSparseIdentity(12), GetSparseIdentity(12), 8, 1e-14, 1.0);
  const auto initial_residual = GetResidualFrobeniusNorm(params.A, params.M0);

  // parallelization test
  std::cout << "Parallelization test:\n";
  CommonParallelizationTest("global_cg", params, mgs::RunConjugateGradient, initial_residual);
  CommonParallelizationTest("global_mr", params, mgs::RunMinimalResidual, initial_residual);
  CommonParallelizationTest("global_lomr", params, mgs::RunLocallyOptimalMinimalResidual, initial_residual);
  CommonParallelizationTest("inner_outer_mr", params, mio::RunMinimalResidual, initial_residual);
  CommonParallelizationTest("inner_outer_lomr", params, mio::RunLocallyOptimalMinimalResidual, initial_residual);
  std::cout << "-------------------------------" << std::endl;

  // relative stopping criterion test
  std::cout << "Relative stopping test:\n";
  mc::InputParams relative_params = params;
  relative_params.tolerance = 0.9;
  const auto relative_result = mgs::RunMinimalResidual(relative_params, 2);
  assert(relative_result.converged);
  assert(relative_result.iterations == 1);
  assert(relative_result.history.size() == 2);
  assert(relative_result.history.back().residual_norm < relative_params.tolerance * initial_residual);
  std::cout << "global_mr_relative iterations = " << relative_result.iterations << " [ok]\n";

  // dropping test
  std::cout << "Dropping test:\n";
  mc::InputParams sparse_params = params;
  sparse_params.max_iterations = 3;
  sparse_params.max_density = 0.12;
  sparse_params.tolerance = 0.0;
  DroppingParallelizationTest("global_cg_drop", sparse_params, mgs::RunConjugateGradient);
  DroppingParallelizationTest("global_mr_drop", sparse_params, mgs::RunMinimalResidual);
  DroppingParallelizationTest("global_lomr_drop", sparse_params, mgs::RunLocallyOptimalMinimalResidual);
  DroppingParallelizationTest("inner_outer_mr_drop", sparse_params, mio::RunMinimalResidual);
  DroppingParallelizationTest("inner_outer_lomr_drop", sparse_params, mio::RunLocallyOptimalMinimalResidual);
  std::cout << "-------------------------------" << std::endl;

  // no dropping test
  std::cout << "No dropping test:\n";
  mc::InputParams no_drop_params = params;
  no_drop_params.max_iterations = 8;
  no_drop_params.tolerance = 0.0;
  no_drop_params.max_density = 0.2;
  no_drop_params.enable_dropping = false;
  CheckDensityStop("global_cg_no_drop", mgs::RunConjugateGradient(no_drop_params, 1), no_drop_params.max_density);
  CheckDensityStop("global_mr_no_drop", mgs::RunMinimalResidual(no_drop_params, 1), no_drop_params.max_density);
  CheckDensityStop("global_lomr_no_drop", mgs::RunLocallyOptimalMinimalResidual(no_drop_params, 1), no_drop_params.max_density);
  CheckDensityStop("inner_outer_mr_no_drop", mio::RunMinimalResidual(no_drop_params, 1), no_drop_params.max_density);
  CheckDensityStop("inner_outer_lomr_no_drop", mio::RunLocallyOptimalMinimalResidual(no_drop_params, 1), no_drop_params.max_density);
  std::cout << "-------------------------------" << std::endl;

  std::cout << "Residual test:\n";
  mc::InputParams reliable_params = params;
  reliable_params.max_iterations = 20;
  reliable_params.tolerance = 1e-20;
  const auto reliable_result = mio::RunLocallyOptimalMinimalResidual(reliable_params, 2);
  const double actual_residual = GetResidualFrobeniusNorm(A, reliable_result.M);
  std::cout << "reliable_residual = " << reliable_result.history.back().residual_norm << " actual = " << actual_residual << " ";
  assert(reliable_result.iterations == reliable_params.max_iterations);
  assert(!reliable_result.converged);
  assert(std::fabs(reliable_result.history.back().residual_norm - actual_residual) < 1e-9);
  std::cout << "[ok]\n";
  return 0;
}
