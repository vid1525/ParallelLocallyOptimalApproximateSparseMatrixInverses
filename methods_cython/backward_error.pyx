import numpy as np
from cython.parallel cimport prange
from libc.math cimport sqrt, isfinite
from libc.stdint cimport int32_t, int64_t
from openmp cimport omp_get_num_threads

ctypedef fused index_t:
    int32_t
    int64_t

ctypedef fused m_index_t:
    int32_t
    int64_t


cdef double _row_dot(
    const double[::1] values, const index_t[::1] indices,
    const index_t[::1] offsets, const double[::1] x, Py_ssize_t row,
) noexcept nogil:
    cdef Py_ssize_t k
    cdef double total = 0.0
    for k in range(offsets[row], offsets[row + 1]):
        total += values[k] * x[indices[k]]
    return total


cdef void _multiply(
    const double[::1] values, const index_t[::1] indices,
    const index_t[::1] offsets, const double[::1] x,
    double[::1] out, int threads,
) noexcept nogil:
    cdef Py_ssize_t row
    for row in prange(out.shape[0], num_threads=threads, schedule='static',
                      use_threads_if=out.shape[0] >= 8192):
        out[row] = _row_dot(values, indices, offsets, x, row)


cdef double _dot(const double[::1] x, const double[::1] y, int threads) noexcept nogil:
    cdef Py_ssize_t i
    cdef double total = 0.0
    for i in prange(x.shape[0], num_threads=threads, schedule='static',
                    use_threads_if=x.shape[0] >= 8192):
        total += x[i] * y[i]
    return total


def parallel_threads(Py_ssize_t n, int requested):
    if n <= 0 or requested <= 0:
        raise ValueError('n and requested must be positive')
    cdef int count = 1
    cdef Py_ssize_t i
    cdef int threads = min(n, requested)
    for i in prange(1, nogil=True, num_threads=threads, use_threads_if=n >= 8192):
        count = omp_get_num_threads()
    return count


def pcg(
    const double[::1] a_values, const index_t[::1] a_indices, const index_t[::1] a_offsets,
    const double[::1] m_values, const m_index_t[::1] m_indices, const m_index_t[::1] m_offsets,
    const double[::1] b, Py_ssize_t max_iterations, int threads, double tolerance,
):
    cdef Py_ssize_t n = b.shape[0]
    if n <= 0 or max_iterations <= 0 or threads <= 0:
        raise ValueError('n, max_iterations and threads must be positive')
    cdef double[::1] r = np.empty(n, dtype=np.float64)
    cdef double[::1] z = np.empty(n, dtype=np.float64)
    cdef double[::1] p = np.empty(n, dtype=np.float64)
    cdef double[::1] Ap = np.empty(n, dtype=np.float64)
    cdef object history_array = np.empty(max_iterations, dtype=np.float64)
    cdef double[::1] history = history_array
    cdef Py_ssize_t i, count = 1
    cdef double b_norm, r_z, denominator, alpha, beta, error, r_r
    cdef int status = 0
    threads = min(threads, n)

    with nogil:
        for i in prange(n, num_threads=threads, schedule='static', use_threads_if=n >= 8192):
            r[i] = b[i]
        _multiply(m_values, m_indices, m_offsets, r, z, threads)
        for i in prange(n, num_threads=threads, schedule='static', use_threads_if=n >= 8192):
            p[i] = z[i]
        r_r = _dot(r, r, threads)
        b_norm = sqrt(r_r)
        r_z = _dot(r, z, threads)
        history[0] = 1.0
        while count < max_iterations and history[count - 1] > tolerance:
            _multiply(a_values, a_indices, a_offsets, p, Ap, threads)
            denominator = _dot(p, Ap, threads)
            if not isfinite(r_z) or not isfinite(denominator) or r_z == 0 or denominator == 0:
                status = 2
                break
            alpha = r_z / denominator
            beta = 1.0 / r_z
            for i in prange(n, num_threads=threads, schedule='static', use_threads_if=n >= 8192):
                r[i] -= alpha * Ap[i]
            r_r = _dot(r, r, threads)
            _multiply(m_values, m_indices, m_offsets, r, z, threads)
            r_z = _dot(r, z, threads)
            beta = beta * r_z
            for i in prange(n, num_threads=threads, schedule='static', use_threads_if=n >= 8192):
                p[i] = z[i] + beta * p[i]
            error = sqrt(r_r) / b_norm
            history[count] = error
            count += 1
            if not isfinite(error):
                status = 3
                break
        if isfinite(history[count - 1]) and history[count - 1] <= tolerance:
            status = 1
    return history_array[:count].copy(), status
