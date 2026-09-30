import numpy as np
from scipy import sparse

from libc.stdint cimport uintptr_t, int64_t


cdef extern from "methods_c_api.h":
    ctypedef struct MatrixData:
        int64_t rows
        int64_t cols
        int64_t nnz
        const int64_t* col_offsets
        const int64_t* row_indices
        const double* values

    ctypedef struct MethodParams:
        int64_t max_iterations
        double tolerance
        double max_density
        int64_t enable_dropping
        int64_t num_threads
        int64_t use_initial_m
        int64_t use_preconditioner

    ctypedef struct SingleIterationData:
        int64_t iteration
        double residual_norm
        double density_m
        double density_direction

    ctypedef struct Result:
        int64_t rows
        int64_t cols
        int64_t nnz
        int64_t iterations
        int64_t converged
        int64_t history_size
        int64_t* col_offsets
        int64_t* row_indices
        double* values
        SingleIterationData* history
        char* error_message

    Result spai_global_cg(MatrixData A, MatrixData M0, MatrixData Pr, MethodParams params) noexcept nogil

    Result spai_global_mr(MatrixData A, MatrixData M0, MatrixData Pr, MethodParams params) noexcept nogil

    Result spai_global_lomr(MatrixData A, MatrixData M0, MatrixData Pr, MethodParams params) noexcept nogil

    Result inner_outer_mr(MatrixData A, MatrixData M0, MatrixData Pr, MethodParams params) noexcept nogil

    Result inner_outer_lomr(MatrixData A, MatrixData M0, MatrixData Pr, MethodParams params) noexcept nogil

    void free_mem(Result* result)


ctypedef Result (*MethodFunc)(MatrixData, MatrixData, MatrixData, MethodParams) noexcept nogil


class OutputResult:
    def __init__(self, M, iterations, converged, history):
        self.M = M
        self.iterations = iterations
        self.converged = converged
        self.history = history
        self.residual_norms = history["residual_norm"]


def _empty_csc():
    return sparse.csc_matrix((0, 0), dtype=np.float64)


def _as_csc(matrix, shape=None):
    if matrix is None:
        return _empty_csc()

    csc = sparse.csc_matrix(matrix, dtype=np.float64, copy=False)
    if shape is not None and csc.shape != tuple(shape):
        raise ValueError("matrix shape does not match A")

    csc.sum_duplicates()
    csc.sort_indices()
    csc.indptr = np.ascontiguousarray(csc.indptr, dtype=np.int64)
    csc.indices = np.ascontiguousarray(csc.indices, dtype=np.int64)
    csc.data = np.ascontiguousarray(csc.data, dtype=np.float64)
    return csc


cdef MatrixData _matrix_data(object csc):
    cdef MatrixData data
    data.rows = <int64_t>csc.shape[0]
    data.cols = <int64_t>csc.shape[1]
    data.nnz = <int64_t>csc.nnz
    data.col_offsets = <const int64_t*><uintptr_t>csc.indptr.ctypes.data
    data.row_indices = <const int64_t*><uintptr_t>csc.indices.ctypes.data
    data.values = <const double*><uintptr_t>csc.data.ctypes.data
    return data


cdef object _history_from_result(Result* result):
    cdef int64_t i
    history = np.empty(result.history_size, dtype=np.dtype([
        ("iteration", np.int64),
        ("residual_norm", np.float64),
        ("density_m", np.float64),
        ("density_direction", np.float64),
    ]))
    for i in range(result.history_size):
        history[i] = (
            result.history[i].iteration,
            result.history[i].residual_norm,
            result.history[i].density_m,
            result.history[i].density_direction,
        )
    return history


cdef object _result_to_python(Result result):
    cdef object message
    cdef object indptr
    cdef object indices
    cdef object values
    cdef object M
    cdef object history

    try:
        if result.error_message != NULL:
            message = (<bytes>result.error_message).decode("utf-8", "replace")
            raise RuntimeError(message)

        indptr = np.asarray(<int64_t[:result.cols + 1]>result.col_offsets).copy()
        if result.nnz:
            indices = np.asarray(<int64_t[:result.nnz]>result.row_indices).copy()
            values = np.asarray(<double[:result.nnz]>result.values).copy()
        else:
            indices = np.empty(0, dtype=np.int64)
            values = np.empty(0, dtype=np.float64)

        M = sparse.csc_matrix((values, indices, indptr), shape=(result.rows, result.cols))
        history = _history_from_result(&result)
        return OutputResult(M, result.iterations, bool(result.converged), history)
    finally:
        free_mem(&result)


cdef object _call_method(
    MethodFunc method,
    object A,
    object M0,
    object Pr,
    int64_t max_iterations,
    double tolerance,
    double max_density,
    object enable_dropping,
    int64_t num_threads,
):
    cdef object A_csc = _as_csc(A)
    cdef object M0_csc = _as_csc(M0, A_csc.shape) if M0 is not None else _empty_csc()
    cdef object Pr_csc = _as_csc(Pr, A_csc.shape) if Pr is not None else _empty_csc()
    cdef MatrixData A_data = _matrix_data(A_csc)
    cdef MatrixData M0_data = _matrix_data(M0_csc)
    cdef MatrixData Pr_data = _matrix_data(Pr_csc)

    cdef MethodParams params
    params.max_iterations = max_iterations
    params.tolerance = tolerance
    params.max_density = max_density
    params.enable_dropping = 1 if enable_dropping else 0
    params.num_threads = num_threads
    params.use_initial_m = 1 if M0 is not None else 0
    params.use_preconditioner = 1 if Pr is not None else 0

    cdef Result result
    with nogil:
        result = method(A_data, M0_data, Pr_data, params)
    return _result_to_python(result)


class GlobalSpaiMethods:
    available_methods = ("cg", "mr", "lomr")

    def cg(self, A, M0=None, Pr=None, max_iterations=100, tolerance=1e-9, max_density=0.03, enable_dropping=True, num_threads=1):
        return _call_method(
            spai_global_cg,
            A,
            M0,
            Pr,
            max_iterations,
            tolerance,
            max_density,
            enable_dropping,
            num_threads,
        )

    def mr(self, A, M0=None, Pr=None, max_iterations=100, tolerance=1e-9, max_density=0.03, enable_dropping=True, num_threads=1):
        return _call_method(
            spai_global_mr,
            A,
            M0,
            Pr,
            max_iterations,
            tolerance,
            max_density,
            enable_dropping,
            num_threads,
        )

    def lomr(self, A, M0=None, Pr=None, max_iterations=100, tolerance=1e-9, max_density=0.03, enable_dropping=True, num_threads=1):
        return _call_method(
            spai_global_lomr,
            A,
            M0,
            Pr,
            max_iterations,
            tolerance,
            max_density,
            enable_dropping,
            num_threads,
        )


class InnerOuterMethods:
    available_methods = ("mr", "lomr")

    def mr(self, A, M0=None, Pr=None, max_iterations=100, tolerance=1e-9, max_density=0.03, enable_dropping=True, num_threads=1):
        return _call_method(
            inner_outer_mr,
            A,
            M0,
            Pr,
            max_iterations,
            tolerance,
            max_density,
            enable_dropping,
            num_threads,
        )

    def lomr(self, A, M0=None, Pr=None, max_iterations=100, tolerance=1e-9, max_density=0.03, enable_dropping=True, num_threads=1):
        return _call_method(
            inner_outer_lomr,
            A,
            M0,
            Pr,
            max_iterations,
            tolerance,
            max_density,
            enable_dropping,
            num_threads,
        )


global_spai = GlobalSpaiMethods()
inner_outer = InnerOuterMethods()
