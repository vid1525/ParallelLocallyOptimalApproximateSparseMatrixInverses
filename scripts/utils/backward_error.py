from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import mmread
from methods_cython.backward_error import pcg, parallel_threads


DEFAULT_MAX_ITERATIONS = 50
DEFAULT_REALIZATIONS = 20
TOLERANCE = 1e-6


def random_rhs(n: int, seed: int = 42) -> np.ndarray:
    return next(random_rhs_realizations(n, seed=seed, realizations=1))


def random_rhs_realizations(n: int, seed: int = 42, realizations: int = DEFAULT_REALIZATIONS):
    if n <= 0:
        raise ValueError('The system must be nonempty')
    if realizations <= 0:
        raise ValueError('realizations must be positive')
    rng = np.random.default_rng(seed)
    for _ in range(realizations):
        b = rng.random(n)
        while np.linalg.norm(b) == 0:
            b = rng.random(n)
        yield b


@dataclass
class BackwardErrorResult:
    history: np.ndarray
    converged: bool
    status: str


def _prepare_matrix(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    matrix = sparse.csr_matrix(matrix, dtype=np.float64, copy=False)
    if (not matrix.has_canonical_format or not matrix.data.flags.c_contiguous
            or not matrix.indices.flags.c_contiguous or not matrix.indptr.flags.c_contiguous):
        matrix = matrix.copy()
        matrix.sum_duplicates()
    matrix.check_format(full_check=True)
    for start in range(0, matrix.nnz, 1 << 20):
        if not np.isfinite(matrix.data[start:start + (1 << 20)]).all():
            raise ValueError('A and M must contain finite values')
    return matrix


class PreparedPCGSystem:
    def __init__(self, A: sparse.spmatrix, num_threads: int = 1):
        if A.shape[0] == 0 or A.shape[0] != A.shape[1]:
            raise ValueError('A must be a nonempty square matrix')
        if num_threads <= 0 or num_threads > np.iinfo(np.int32).max:
            raise ValueError('num_threads must be a positive OpenMP thread count')
        self.A = _prepare_matrix(A)
        self.num_threads = min(num_threads, A.shape[0])
        self.active_threads = parallel_threads(A.shape[0], self.num_threads)

    def prepare_preconditioner(self, M: sparse.spmatrix) -> PreparedPCGSolver:
        return PreparedPCGSolver(self, M)


class PreparedPCGSolver:
    def __init__(self, system: PreparedPCGSystem, M: sparse.spmatrix):
        if M.shape != system.A.shape:
            raise ValueError('A and M must have the same shape')
        self.system = system
        self.M = _prepare_matrix(M)

    def calculate(self, max_iterations: int = DEFAULT_MAX_ITERATIONS,
                  *, b: np.ndarray | None = None) -> BackwardErrorResult:
        if max_iterations < 0:
            raise ValueError('max_iterations must be nonnegative')
        A, M = self.system.A, self.M
        b = random_rhs(A.shape[0]) if b is None else np.ascontiguousarray(b, dtype=np.float64)
        b_norm = np.linalg.norm(b)
        if b.shape != (A.shape[0],) or not np.isfinite(b_norm) or b_norm <= 0:
            raise ValueError('b must be a finite vector with ||b|| > 0')
        history, status = pcg(
            A.data, A.indices, A.indptr, M.data, M.indices, M.indptr, b,
            max_iterations or A.shape[0], self.system.num_threads, TOLERANCE)
        statuses = {
            0: 'iteration limit reached',
            1: 'converged',
            2: 'PCG breakdown (invalid recurrence denominator)',
            3: 'PCG breakdown (nonfinite residual)',
        }
        return BackwardErrorResult(history, status == 1, statuses[status])


def calculate_backward_error(
    A: sparse.spmatrix,
    M: sparse.spmatrix,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    *,
    b: np.ndarray | None = None,
    num_threads: int = 1,
) -> BackwardErrorResult:
    system = PreparedPCGSystem(A, num_threads=num_threads)
    return system.prepare_preconditioner(M).calculate(max_iterations, b=b)


def load_preconditioner(matrix_path: Path) -> sparse.csr_matrix:
    M = sparse.csr_matrix(mmread(matrix_path), dtype=np.float64)
    M = M + M.T
    M.data *= 0.5
    return M


def evaluate_saved_preconditioner(
    A: sparse.spmatrix, matrix_path: Path, max_iterations: int,
    *, b: np.ndarray | None = None,
    num_threads: int = 1,
) -> BackwardErrorResult:
    M = load_preconditioner(matrix_path)
    return calculate_backward_error(A, M, max_iterations, b=b, num_threads=num_threads)
