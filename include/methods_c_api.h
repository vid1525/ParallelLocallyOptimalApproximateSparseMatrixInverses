#pragma once

#include <stdint.h>

extern "C" {

struct MatrixData {
  int64_t rows;
  int64_t cols;
  int64_t nnz;
  const int64_t* col_offsets;
  const int64_t* row_indices;
  const double* values;
};

struct MethodParams {
  int64_t max_iterations;
  double tolerance;
  double max_density;
  int64_t enable_dropping;
  int64_t num_threads;
  int64_t use_initial_m;
  int64_t use_preconditioner;
  int64_t inner_iterations;
};

struct SingleIterationData {
  int64_t iteration;
  double residual_norm;
  double density_m;
  double density_direction;
};

struct Result {
  int64_t rows;
  int64_t cols;
  int64_t nnz;
  int64_t iterations;
  int64_t converged;
  int64_t history_size;
  int64_t* col_offsets;
  int64_t* row_indices;
  double* values;
  SingleIterationData* history;
  char* error_message;
};

Result spai_global_cg(MatrixData A, MatrixData M0, MatrixData Pr, MethodParams params);

Result spai_global_mr(MatrixData A, MatrixData M0, MatrixData Pr, MethodParams params);

Result spai_global_lomr(MatrixData A, MatrixData M0, MatrixData Pr, MethodParams params);

Result inner_outer_mr(MatrixData A, MatrixData M0, MatrixData Pr, MethodParams params);

Result inner_outer_lomr(MatrixData A, MatrixData M0, MatrixData Pr, MethodParams params);

void free_mem(Result* result);

}
