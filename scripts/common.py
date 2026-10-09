from __future__ import annotations

import argparse
import collections
import csv
from contextlib import nullcontext
import json
import numpy as np
from scipy import sparse
from scipy.io import mmread, mmwrite
from scipy.sparse.linalg import norm
import sys
import typing
from pathlib import Path
from tempfile import TemporaryDirectory
import matplotlib
matplotlib.use('Agg')
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
import scripts.utils.backward_error as backward_error

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
    matrix_data.load_data()
    matrix_data.dataset_dir.mkdir(parents=True, exist_ok=True)
    residual_figure, residual_axis = plt.subplots(
        figsize=(args.figure_width, args.figure_height)
    )
    families = methods.get_method_families(args.method_family)
    print(f'{matrix_data.name}: preconditioner construction limit = {args.max_iterations}')
    convergence_table = typst.TypstTable(_get_convergence_table_row_columns())
    residual_histories = []
    construction_evaluations = []
    saved_preconditioners = []

    storage = (
        TemporaryDirectory(prefix='backward-error-', dir=matrix_data.dataset_dir)
        if args.backward_error and not args.write_preconditioners
        else nullcontext(matrix_data.dataset_dir)
    )

    with storage as storage_dir:
        try:
            for family in families:
                for method_name, method in methods.generate_methods_by_family(family):
                    convergence_table.add_row(
                        _run_single_method(
                            family=family,
                            matrix_data=matrix_data,
                            method_name=method_name,
                            method=method,
                            args=args,
                            residual_axis=residual_axis,
                            residual_histories=residual_histories,
                            preconditioner_dir=Path(storage_dir),
                            saved_preconditioners=saved_preconditioners,
                            construction_evaluations=construction_evaluations,
                        )
                    )

            chart_name = args.method_family
            _plot_chart(
                ds_name=matrix_data.name,
                chart_name=chart_name,
                dataset_dir=matrix_data.dataset_dir,
                args=args,
                figure=residual_figure,
                axis=residual_axis,
                y_label=r'Residual norm, $||R_i||_F$',
                filename_suffix='convergence',
            )
            _dump_iteration_tables(
                matrix_data.dataset_dir, chart_name, args, 'convergence', residual_histories)
            convergence_table.dump(matrix_data.dataset_dir / f'{chart_name}_summary.typ')
            construction_path = matrix_data.dataset_dir / (
                f'{chart_name}-{_get_datastring_in_filename(args)}-construction_status.csv')
            with construction_path.open('w', newline='', encoding='utf-8') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(construction_evaluations[0]))
                writer.writeheader()
                writer.writerows(construction_evaluations)
        finally:
            plt.close(residual_figure)

        if not args.backward_error:
            return

        history_limit = args.max_backward_error_iterations or matrix_data.n
        print(f'{matrix_data.name}: backward-error history limit = {history_limit} '
              '(includes initial residual)')
        backward_error_figure, backward_error_axis = plt.subplots(
            figsize=(args.figure_width, args.figure_height)
        )
        backward_error_histories = [[] for _ in range(args.backward_error_realizations)]
        evaluations = []
        try:
            system = backward_error.PreparedPCGSystem(matrix_data.A, num_threads=args.num_threads)
            for label, style, path in saved_preconditioners:
                M = backward_error.load_preconditioner(path)
                solver = system.prepare_preconditioner(M)
                failures = []
                rhs = backward_error.random_rhs_realizations(
                    system.A, seed=args.backward_error_seed, realizations=args.backward_error_realizations, num_threads=system.num_threads
                )
                for realization, b in enumerate(rhs):
                    evaluation = solver.calculate(args.max_backward_error_iterations, b=b)
                    evaluations.append({
                        'method': label,
                        'realization': realization + 1,
                        'solver': evaluation.solver,
                        'active_threads': system.active_threads,
                        'history_limit': history_limit,
                        'solver_updates': evaluation.history.size - 1,
                        'final_relative_residual': evaluation.history[-1],
                        'converged': evaluation.converged,
                        'status': evaluation.status,
                    })
                    if not evaluation.converged:
                        failures.append(f'{realization + 1}: {evaluation.status}')
                    backward_error_histories[realization].append((label, evaluation.history))
                    _plot_backward_error_history(backward_error_axis, evaluation.history, label, style, realization)
                if failures:
                    print(f'{label}: backward-error evaluation failed: {"; ".join(failures)}')
                else:
                    print(f'{label}: backward-error evaluation converged')
                del solver, M
            _plot_chart(
                ds_name=matrix_data.name,
                chart_name=chart_name,
                dataset_dir=matrix_data.dataset_dir,
                args=args,
                figure=backward_error_figure,
                axis=backward_error_axis,
                y_label=r'Backward error, $||r_i||_2 / ||b||_2$',
                filename_suffix='backward_error',
            )
            for realization, histories in enumerate(backward_error_histories, start=1):
                suffix = 'backward_error' if realization == 1 else f'backward_error_realization_{realization:02d}'
                _dump_iteration_tables(matrix_data.dataset_dir, chart_name, args, suffix, histories)
            status_path = matrix_data.dataset_dir / f'{chart_name}-{_get_datastring_in_filename(args, "backward_error")}-solver_status.csv'
            with status_path.open('w', newline='', encoding='utf-8') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(evaluations[0]))
                writer.writeheader()
                writer.writerows(evaluations)
        finally:
            plt.close(backward_error_figure)


def _plot_backward_error_history(
    axis: plt.Axes, history: np.ndarray, label: str,
    style: dict[str, str], realization: int,
) -> None:
    prominent = realization == 0
    curve_style = dict(style, marker=style['marker'] if prominent else None)
    axis.semilogy(
        history,
        linewidth=3.0 if prominent else 1.0,
        alpha=1.0 if prominent else 0.2,
        zorder=3 if prominent else 1,
        markersize=5,
        markerfacecolor='white',
        markevery=max(1, history.size // 12),
        label=label if prominent else '_nolegend_',
        **curve_style,
    )


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
    residual_axis: plt.Axes,
    residual_histories: list[tuple[str, np.ndarray]],
    preconditioner_dir: Path,
    saved_preconditioners: list[tuple[str, dict[str, str], Path]],
    construction_evaluations: list[dict[str, object]],
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
    actual_residual = norm(sparse.eye(matrix_data.n, format='csc') - matrix_data.A @ result.M)
    construction_evaluations.append({
        'method': label,
        'requested_threads': args.num_threads,
        'iterations': result.iterations,
        'reported_residual_norm': result.history[-1]['residual_norm'],
        'actual_residual_norm': actual_residual,
        'construction_backward_error': actual_residual / (
            norm(matrix_data.A) * norm(result.M) + np.sqrt(matrix_data.n)),
        'density': result.M.nnz / matrix_data.n ** 2,
        'converged': actual_residual < args.tolerance,
    })
    result_folder = methods.get_preconditioner_result_folder(family, method_name)
    style = methods.get_plot_style(family, method_name)
    iterations = result.history['iteration']
    if int(iterations[-1]) != result.iterations:
        raise RuntimeError('Method iteration count does not match its history')

    if args.write_preconditioners or args.backward_error:
        path = _save_preconditioner_matrix(preconditioner_dir, result_folder, result.M)
        saved_preconditioners.append((label, style, path))
    residual_axis.semilogy(
        iterations,
        result.history['residual_norm'],
        linewidth=2,
        markevery=max(1, iterations.size // 12),
        markersize=4,
        markerfacecolor='white',
        label=label,
        **style,
    )
    residual_histories.append((label, result.history['residual_norm']))
    print(f'{label}: completed {result.iterations} preconditioner iterations '
          f'(limit {args.max_iterations})')
    if result.converged and actual_residual >= args.tolerance:
        print(f'{label}: construction did not converge (true residual above tolerance)')
    elif not result.converged:
        stop_reason = ('density limit reached' if not args.enable_dropping
                       and result.history[-1]['density_m'] >= args.max_density
                       else 'iteration limit reached' if result.iterations == args.max_iterations
                       else 'recurrence breakdown')
        print(f'{label}: construction did not converge ({stop_reason})')
    return _ConvergenceTableRow(
        method=label,
        n=matrix_data.n,
        nnz=matrix_data.nnz,
        # lambda_min=matrix_data.lambda_min,
        # lambda_max=matrix_data.lambda_max,
        last_iteration=int(iterations[-1]),
        residual_norm=result.history[-1]['residual_norm'],
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
        self.Pr = MatrixData._get_diagonal_inverse_preconditioner(self.A)
        self.M0 = sparse.eye(self.n, format='csc')

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
    def _get_diagonal_inverse_preconditioner(A: sparse.spmatrix) -> sparse.csc_matrix:
        diag = A.diagonal()
        if not np.isfinite(diag).all() or np.any(diag <= 0.0):
            raise ValueError('A must have a finite positive diagonal.')
        return sparse.diags(1.0 / diag, format='csc')


def _get_datastring_in_filename(args: argparse.Namespace, filename_suffix: str = '') -> str:
    data = f'{"dropping" if args.enable_dropping else "no_dropping"}-max_iter_{args.max_iterations}-max_density_{args.max_density:.3f}'
    if filename_suffix.startswith('backward_error'):
        data += f'-max_backward_iter_{args.max_backward_error_iterations}'
    return data


# private functions
def _plot_chart(
    ds_name: str,
    chart_name: str,
    dataset_dir: Path,
    args: argparse.Namespace,
    figure: plt.Figure,
    axis: plt.Axes,
    y_label: str,
    filename_suffix: str,
) -> Path:
    iteration_label = 'PCG iteration, $i$' if filename_suffix.startswith('backward_error') else 'Iteration, $i$'
    axis.set_xlabel(iteration_label, fontsize=DEFAULT_FONT_SIZE)
    axis.set_ylabel(y_label, labelpad=5, fontsize=DEFAULT_FONT_SIZE)
    axis.set_title(ds_name, fontsize=DEFAULT_FONT_SIZE)
    axis.grid(True, alpha=0.6)
    axis.legend(fontsize=DEFAULT_FONT_SIZE)
    axis.tick_params(axis='both', which='major', labelsize=DEFAULT_FONT_SIZE)
    
    dataset_dir.mkdir(parents=True, exist_ok=True)
    chart_filename = dataset_dir / f'{chart_name}-{_get_datastring_in_filename(args, filename_suffix)}-{filename_suffix}.png'
    figure.tight_layout()
    figure.savefig(chart_filename, dpi=DEFAULT_FIGURE_DPI)
    plt.close(figure)
    print(f'saved {chart_filename}')


def _dump_iteration_tables(
    dataset_dir: Path,
    chart_name: str,
    args: argparse.Namespace,
    filename_suffix: str,
    histories: list[tuple[str, np.ndarray]],
) -> None:
    if not histories:
        return
    max_iterations = max(history.size for _, history in histories)
    filename = f'{chart_name}-{_get_datastring_in_filename(args, filename_suffix)}-{filename_suffix}'
    headers = ['Iteration', *(label for label, _ in histories)]
    rows = []
    for iteration in range(max_iterations):
        rows.append([
            iteration,
            *(history[iteration] if iteration < history.size else '' for _, history in histories),
        ])

    csv_path = dataset_dir / f'{filename}.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f'saved {csv_path}')

    columns = [('iteration', 'Iteration')] + [
        (f'method_{index}', label) for index, (label, _) in enumerate(histories)
    ]
    row_type = collections.namedtuple('IterationRow', [key for key, _ in columns])
    table = typst.TypstTable(columns)
    for row in rows:
        table.add_row(row_type(*row))
    typst_path = dataset_dir / f'{filename}.typ'
    table.dump(typst_path)
    print(f'saved {typst_path}')


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
        ('residual_norm', '$||R||_F$ (reported)'),
        ('density', '$cal(n n z) \\/ n^2$'),
    ]


_ConvergenceTableRow = collections.namedtuple(
    '_ConvergenceTableRow', typst.get_params_from_generator(_get_convergence_table_row_columns)
)
