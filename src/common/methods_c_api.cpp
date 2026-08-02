#include "methods_c_api.h"

#include "methods.hpp"

#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>

namespace {

namespace mc = methods::common;
namespace mgs = methods::global_spai;
namespace mio = methods::inner_outer;

mc::SparseMatrix GetCompressedSparseMatrix(const MatrixData& data) {
  if (data.rows <= 0 || data.cols <= 0 || data.nnz <= 0) {
    throw std::invalid_argument("Sparse matrix dimensions and nnz must be positive");
  }
  for (int64_t i = 0; i < data.cols; ++i) {
    if (data.col_offsets[i] > data.col_offsets[i + 1]) {
      throw std::invalid_argument("Sparse matrix column offsets must be non-decreasing");
    }
  }
  for (int64_t i = 0; i < data.nnz; ++i) {
    if (data.row_indices[i] < 0 || data.row_indices[i] >= data.rows) {
      throw std::invalid_argument("Sparse matrix row idx is out of range");
    }
  }
  mc::SparseMatrix matrix = Eigen::Map<const mc::SparseMatrix, Eigen::Unaligned, Eigen::Stride<0, 0>>(
    data.rows,
    data.cols,
    data.nnz,
    data.col_offsets,
    data.row_indices,
    data.values
  );
  matrix.makeCompressed();
  return matrix;
}

mc::InputParams GetInputParams(
  const MatrixData& A,
  const MatrixData& M0,
  const MatrixData& Pr,
  const MethodParams& params
) {
  return mc::InputParams(
    GetCompressedSparseMatrix(A),
    params.use_initial_m ? GetCompressedSparseMatrix(M0) : mc::SparseMatrix(),
    params.use_preconditioner ? GetCompressedSparseMatrix(Pr) : mc::SparseMatrix(),
    params.max_iterations,
    params.tolerance,
    params.max_density,
    params.enable_dropping != 0
  );
}

Result GetErrorResult(const std::string& message) {
  Result out{};
  out.error_message = new char[message.size() + 1];
  std::memcpy(out.error_message, message.c_str(), message.size() + 1);
  return out;
}

Result GetSuccessResult(mc::OutputResult&& cpp_result) {
  auto& matrix = cpp_result.M;
  auto& history = cpp_result.history;
  matrix.makeCompressed();
  const auto rows = static_cast<int64_t>(matrix.rows());
  const auto cols = static_cast<int64_t>(matrix.cols());
  const auto nnz = static_cast<int64_t>(matrix.nonZeros());
  const auto history_size = static_cast<int64_t>(history.size());
  auto c_result = Result{
    .rows = rows,
    .cols = cols,
    .nnz = nnz,
    .iterations = cpp_result.iterations,
    .converged = cpp_result.converged,
    .history_size = history_size,
    .col_offsets = nullptr,
    .row_indices = nullptr,
    .values = nullptr,
    .history = nullptr,
    .error_message = nullptr
  };

  try {
    c_result.col_offsets = new int64_t[cols + 1];
    c_result.row_indices = new int64_t[nnz];
    c_result.values = new double[nnz];
    c_result.history = new SingleIterationData[history_size];

    std::memcpy(c_result.col_offsets, matrix.outerIndexPtr(), sizeof(int64_t) * (c_result.cols + 1));
    std::memcpy(c_result.row_indices, matrix.innerIndexPtr(), sizeof(int64_t) * c_result.nnz);
    std::memcpy(c_result.values, matrix.valuePtr(), sizeof(double) * c_result.nnz);
    for (int64_t i = 0; i < c_result.history_size; ++i) {
      const auto& cur = history[i];
      c_result.history[i] = SingleIterationData{
        .iteration = cur.iteration,
        .residual_norm = cur.residual_norm,
        .density_m = cur.density_m,
        .density_direction = cur.density_direction
      };
    }
  } catch (...) {
    delete [] c_result.col_offsets;
    delete [] c_result.row_indices;
    delete [] c_result.values;
    delete [] c_result.history;
    throw;
  }
  return c_result;
}

template <typename TMethod>
Result RunMethod(MatrixData A, MatrixData M0, MatrixData Pr, MethodParams cparams, TMethod method) {
  try {
    const auto input_params = GetInputParams(A, M0, Pr, cparams);
    return GetSuccessResult(method(input_params, cparams.num_threads));
  } catch (const std::exception& ex) {
    return GetErrorResult(ex.what());
  } catch (...) {
    return GetErrorResult("Unknown failure.");
  }
}

}  // namespace

extern "C" {

Result spai_global_cg(MatrixData A, MatrixData M0, MatrixData Pr, MethodParams params) {
  return RunMethod(A, M0, Pr, params, mgs::RunConjugateGradient);
}

Result spai_global_mr(MatrixData A, MatrixData M0, MatrixData Pr, MethodParams params) {
  return RunMethod(A, M0, Pr, params, mgs::RunMinimalResidual);
}

Result spai_global_lomr(MatrixData A, MatrixData M0, MatrixData Pr, MethodParams params) {
  return RunMethod(A, M0, Pr, params, mgs::RunLocallyOptimalMinimalResidual);
}

Result inner_outer_mr(MatrixData A, MatrixData M0, MatrixData Pr, MethodParams params) {
  return RunMethod(A, M0, Pr, params, mio::RunMinimalResidual);
}

Result inner_outer_lomr(MatrixData A, MatrixData M0, MatrixData Pr, MethodParams params) {
  return RunMethod(A, M0, Pr, params, mio::RunLocallyOptimalMinimalResidual);
}

Result inner_outer_minres(MatrixData A, MatrixData M0, MatrixData Pr, MethodParams params) {
  return RunMethod(A, M0, Pr, params, mio::RunMinres);
}

void free_mem(Result* result) {
  if (result == nullptr) {
    return;
  }
  delete [] result->col_offsets;
  delete [] result->row_indices;
  delete [] result->values;
  delete [] result->history;
  delete [] result->error_message;
  result->col_offsets = nullptr;
  result->row_indices = nullptr;
  result->values = nullptr;
  result->history = nullptr;
  result->error_message = nullptr;
}

}  // extern "C"
