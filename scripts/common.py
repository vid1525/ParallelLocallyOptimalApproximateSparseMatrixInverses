from __future__ import annotations

import argparse
import collections
import json
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from scipy.io import mmread, mmwrite
import sys
import typing
from pathlib import Path
import matplotlib.pyplot as plt


# default paths
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPTS = ROOT / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import scripts.utils.methods as methods
import scripts.utils.typst as typst

DEFAULT_SAMPLE_DATASETS_DIR = SCRIPTS / 'sample_datasets'
DEFAULT_LARGE_DATASET_DIR = SCRIPTS / 'datasets'
DEFAULT_LARGE_DATASET_CONFIG = DEFAULT_LARGE_DATASET_DIR / 'config.json'


# math
EPS = 1e-9


# figure
DEFAULT_FIGURE_WIDTH = 10.0
DEFAULT_FIGURE_HEIGHT = 6.0
DEFAULT_FIGURE_DPI = 300
DEFAULT_FONT_SIZE = 16

# sample datasets
SAMPLE_DATASET_TRI_100_EIGS_4K = 'tri100eigs4k'
SAMPLE_DATASET_4_BW_100_EIGS_20K = '4bw100eigs20k'
SAMPLE_DATASET_4_BW_100_EIGS_20K_2 = '4bw100eigs20k2'
SAMPLE_DATASET_RAND_20K = 'rand20k'
SAMPLE_DATASET_RAND_20K_2 = 'rand20k2'
SAMPLE_DATASET_POISSON_32K = 'Poisson32k'
SAMPLE_DATASET_MSC_04515 = 'msc04515'
SAMPLE_DATASET_BUNDLE_1 = 'bundle1'
SAMPLE_DATASET_BCSSTK_21 = 'bcsstk21'
SAMPLE_DATASET_WATHEN_100 = 'wathen100'
SAMPLE_DATASETS = [
    SAMPLE_DATASET_TRI_100_EIGS_4K,
    SAMPLE_DATASET_4_BW_100_EIGS_20K,
    SAMPLE_DATASET_4_BW_100_EIGS_20K_2,
    SAMPLE_DATASET_RAND_20K,
    SAMPLE_DATASET_RAND_20K_2,
    SAMPLE_DATASET_POISSON_32K,
    SAMPLE_DATASET_MSC_04515,
    SAMPLE_DATASET_BUNDLE_1,
    SAMPLE_DATASET_BCSSTK_21,
    SAMPLE_DATASET_WATHEN_100,
]


# public functions
def run_methods(matrix_data: MatrixData, args: argparse.Namespace) -> None:
    plt.figure(figsize=(args.figure_width, args.figure_height))
    families = methods.get_method_families(args.method_family)
    matrix_data.load_data()
    convergence_table = typst.TypstTable(_get_convergence_table_row_columns())

    for family in families:
        for method_name, method in methods.generate_methods_by_family(family):
            convergence_table.add_row(
                _run_single_method(
                    family=family,
                    matrix_data=matrix_data,
                    method_name=method_name,
                    method=method,
                    args=args,
                )
            )

    chart_name = args.method_family
    _plot_convergence_chart(
        ds_name=matrix_data.name,
        chart_name=chart_name,
        dataset_dir=matrix_data.dataset_dir,
        args=args,
    )
    convergence_table.dump(matrix_data.dataset_dir / f'{chart_name}_summary.typ')
    # print(
    #     f'{matrix_data.name}: lambda_max = {matrix_data.lambda_max:.12g}, '
    #     f'lambda_min = {matrix_data.lambda_min:.12g}, kappa = {matrix_data.kappa:.12g}'
    # )


def update_output_folder(args: argparse.Namespace):
    if args.enable_dropping:
        args.output_dir = args.output_dir / 'dropping'
    else:
        args.output_dir = args.output_dir / 'no_dropping'

    args.output_dir = args.output_dir / f'max_iter_{args.max_iterations}' / f'max_density_{args.max_density:.3f}'


def _run_single_method(
    family: dict[str, object],
    matrix_data: MatrixData,
    method_name: str,
    method: typing.Callable,
    args: argparse.Namespace,
) -> _ConvergenceTableRow:
    result = method(
        matrix_data.A,
        M0=matrix_data.M0,
        Pr=matrix_data.Pr,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        max_density=args.max_density,
        enable_dropping=args.enable_dropping,
        num_threads=args.num_threads,
    )

    label = methods.get_method_label(family, method_name)
    result_folder = methods.get_preconditioner_result_folder(family, method_name)
    graph = result.history['residual_norm']

    _save_preconditioner_matrix(matrix_data.dataset_dir, result_folder, result.M)
    plt.semilogy(np.arange(graph.size), graph, linewidth=2, label=label)
    final_iteration = result.history[-1]
    return _ConvergenceTableRow(
        method=label,
        n=matrix_data.n,
        nnz=matrix_data.nnz,
        # lambda_min=matrix_data.lambda_min,
        # lambda_max=matrix_data.lambda_max,
        last_iteration=int(final_iteration['iteration']),
        residual_norm=final_iteration['residual_norm'],
        density=result.M.nnz / (matrix_data.n ** 2),
    )


def select_sample_datasets(args: argparse.Namespace) -> list[MatrixData]:
    if args is None:
        raise ValueError('Expected args to be not None')

    def _get_sample_datasets(args: argparse.Namespace) -> list[MatrixData]:
        f = lambda x: f'{x}-dataset.mtx'
        return list(map(lambda name: MatrixData(name, f(name), args), SAMPLE_DATASETS))

    datasets = _get_sample_datasets(args)
    if args.datasets is None:
        return datasets

    requested = set(args.datasets)
    selected = [dataset for dataset in datasets if dataset.name in requested]
    missing = requested.difference(dataset.name for dataset in selected)
    if missing:
        raise ValueError(f'Unknown sample dataset(s): {", ".join(sorted(missing))}')
    return selected


def select_large_datasets(args: argparse.Namespace) -> list[MatrixData]:
    if args is None:
        raise ValueError('Expected args to be not None')

    matrix_data_list_from_config = _load_matrix_data_from_config(args.config)
    datasets = list(
        map(
            lambda x: MatrixData(x.name, x.filename, args, group=x.group, source_url=x.source_url),
            matrix_data_list_from_config
        )
    )

    if args.datasets is None:
        return datasets

    requested = set(args.datasets)
    selected = [dataset for dataset in datasets if dataset.name in requested]
    missing = requested.difference(dataset.name for dataset in selected)
    if missing:
        raise ValueError(f'Unknown large dataset(s): {", ".join(sorted(missing))}')
    return selected


MatrixDataFromConfig = collections.namedtuple(
    'MatrixDataFromConfig', [
        'name',
        'group',
        'rows',
        'cols',
        'nnz',
        'source_page',
        'source_url',
        'filename',
    ]
)


def large_dataset_matrix_path(dataset: MatrixDataFromConfig, dataset_dir: Path) -> Path:
    return dataset_dir / dataset.filename


class MatrixData:
    def __init__(self, name: str, filename: str, args: argparse.Namespace, group: str = None, source_url: str = None):
        if args is None:
            raise ValueError('Expected args to be not None')

        # matrix data metadata
        self.name = name
        self.filename = filename
        self.group = group
        self.source_url = source_url
        self.output_dir = args.output_dir
        self.dataset_dir = args.output_dir / name
        self.input_dataset_dir = args.datasets_dir

    def load_data(self):
        self.A = MatrixData._load_sample_matrix(self.input_dataset_dir, self.filename)
        self.n = self.A.shape[0]
        self.nnz = self.A.nnz
        self.density = self.nnz / (self.n ** 2)
        # self.lambda_max, self.lambda_min, self.kappa = MatrixData._get_max_min_eigenvalues(self.A)
        self.Pr = MatrixData._get_diagonal_inverse_preconditioner(self.A)
        self.M0 = sparse.eye(self.A.shape[0], format='csc', dtype=np.float64)

    @staticmethod
    def _load_sample_matrix(dataset_dir: Path, filename: str) -> sparse.csc_matrix:
        matrix_path = dataset_dir / filename
        if not matrix_path.exists():
            raise FileNotFoundError(
                f'Missing {matrix_path}, generate it first with generate_sample_datasets.py --output-dir {dataset_dir}'
            )
        return MatrixData._load_sparse_matrix(matrix_path)

    @staticmethod
    def _load_sparse_matrix(matrix_path: Path) -> sparse.csc_matrix:
        A = mmread(matrix_path).tocsc().astype(np.float64)
        A.sum_duplicates()
        A.eliminate_zeros()
        A.sort_indices()
        return A

    @staticmethod
    def _get_max_min_eigenvalues(A: sparse.spmatrix) -> tuple[float, float, float]:
        def _get_eigenvalue(A: sparse.spmatrix, which: str, maxiter) -> float:
            return float(sparse_linalg.eigsh(A, k=1, return_eigenvectors=False, which=which, maxiter=maxiter)[0])

        maxiter = max(10 * A.shape[0], 1000)
        lambda_max = _get_eigenvalue(A, which='LA', maxiter=maxiter)
        lambda_min = _get_eigenvalue(A, which='SA', maxiter=maxiter)
        return lambda_max, lambda_min, (lambda_max / lambda_min if np.abs(lambda_min) > EPS else np.nan)

    @staticmethod
    def _get_diagonal_inverse_preconditioner(A: sparse.spmatrix) -> sparse.csc_matrix:
        diag = A.diagonal()
        if np.any(abs(diag) < EPS):
            raise ValueError('A has a zero diagonal entry.')
        return sparse.diags(1.0 / diag, format='csc')


def _get_datastring_in_title(args: argparse.Namespace) -> str:
    return f'{"dropping" if args.enable_dropping else "no dropping"}, max_iter - {args.max_iterations}, max_density - {args.max_density:.3f}'


def _get_datastring_in_filename(args: argparse.Namespace) -> str:
    return f'{"dropping" if args.enable_dropping else "no_dropping"}-max_iter_{args.max_iterations}-max_density_{args.max_density:.3f}'


# private functions
def _plot_convergence_chart(ds_name: str, chart_name: str, dataset_dir: Path, args: argparse.Namespace) -> Path:
    plt.xlabel('Iteration, $i$', fontsize=DEFAULT_FONT_SIZE)
    plt.ylabel(r'Residual norm, $||R_i||_F$', labelpad=5, fontsize=DEFAULT_FONT_SIZE)
    plt.title(f'{ds_name}', fontsize=DEFAULT_FONT_SIZE)  # : {_get_datastring_in_title(args)}
    plt.grid(True, alpha=0.6)
    plt.legend(fontsize=DEFAULT_FONT_SIZE)
    plt.tick_params(axis='both', which='major', labelsize=DEFAULT_FONT_SIZE)
    
    dataset_dir.mkdir(parents=True, exist_ok=True)
    chart_filename = dataset_dir / f'{chart_name}-{_get_datastring_in_filename(args)}-convergence.png'
    plt.tight_layout()
    plt.savefig(chart_filename, dpi=DEFAULT_FIGURE_DPI)
    plt.close()
    print(f'saved {chart_filename}')


def _load_matrix_data_from_config(path: Path) -> list[MatrixData]:
    with path.open() as config_file:
        datasets = json.load(config_file)

    if not isinstance(datasets, list):
        raise ValueError(f'{path} invalid configuration file')

    return [MatrixDataFromConfig(**dataset) for dataset in datasets]


def _save_preconditioner_matrix(dataset_dir: Path, method_name: str, M: sparse.spmatrix) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    method_dir = dataset_dir / method_name
    method_dir.mkdir(parents=True, exist_ok=True)
    filename = method_dir / 'preconditioner.mtx'
    mmwrite(filename, M)
    return filename


def _get_convergence_table_row_columns() -> list[tuple[str, str]]:
    return [
        ('method', 'Method'),
        ('n', '$n$'),
        ('nnz', '$cal(n n z)$'),
        # ('lambda_min', '$lambda_min$'),
        # ('lambda_max', '$lambda_max$'),
        ('last_iteration', '\\#iter'),
        ('residual_norm', '$||I_n - A M||_F$'),
        ('density', '$cal(n n z) \\/ n^2$'),
    ]


_ConvergenceTableRow = collections.namedtuple(
    '_ConvergenceTableRow', typst.get_params_from_generator(_get_convergence_table_row_columns)
)
