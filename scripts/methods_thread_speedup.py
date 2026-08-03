import argparse
import collections
import time
import typing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sys

import common
import scripts.utils.methods as methods
import scripts.utils.typst as typst


THREAD_COUNTS = (1, 2, 4, 8, 16, 32)

_FIELD_METHOD = 'method'


def _get_average_running_time(
    method: typing.Callable,
    matrix_data: common.MatrixData,
    num_threads: int,
    args: argparse.Namespace,
) -> float:
    timing_sum = 0.0

    for try_idx in range(args.tries):
        start = time.perf_counter()
        method(
            matrix_data.A,
            M0=matrix_data.M0,
            Pr=matrix_data.Pr,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            max_density=args.max_density,
            enable_dropping=args.enable_dropping,
            num_threads=args.num_threads,
        )
        elapsed = time.perf_counter() - start
        timing_sum += elapsed
        print(f'p = {num_threads}, repetition = {try_idx + 1} / {args.tries}: {elapsed:.6g}s')

    return timing_sum / args.tries


def _get_param_key_by_thread(p: int) -> str:
    return f'p_{p}'


def _get_speedup_table_row_columns() -> list[tuple[str, str]]:
    return [(_FIELD_METHOD, 'Method')] + list(map(lambda x: (_get_param_key_by_thread(x), f'$p = {x}$'), THREAD_COUNTS))


_SpeedupTableRow = collections.namedtuple(
    '_SpeedupTableRow', typst.get_params_from_generator(_get_speedup_table_row_columns)
)


def _run_method_for_threads(
    averages_by_method: dict[str, list[float]],
    family: dict[str, object],
    matrix_data: common.MatrixData,
    method_name: str,
    method: typing.Callable,
    args: argparse.Namespace,
) -> _SpeedupTableRow:
    label = methods.get_method_label(family, method_name)
    avg_timings = []

    for num_threads in THREAD_COUNTS:
        args.num_threads = num_threads
        print(f'running {label} with p = {num_threads}')
        avg_timings.append(_get_average_running_time(method, matrix_data, num_threads, args))

    averages_by_method[label] = avg_timings
    return _build_speedup_table_row(avg_timings, label)


def _build_speedup_table_row(avg_timings: list[float], label) -> _SpeedupTableRow:
    baseline_timing = avg_timings[0]
    p_to_value = {_FIELD_METHOD: label}

    for p, current_timing in zip(THREAD_COUNTS, avg_timings):
        speedup = baseline_timing / current_timing if baseline_timing > 0.0 and current_timing > 0.0 else np.nan
        effectiveness = speedup / p
        p_to_value[_get_param_key_by_thread(p)] = f'$t={current_timing:.2f}$s\n$S_p={speedup:.2f}$\n$E_p={effectiveness:.2f}$'

    return _SpeedupTableRow(**p_to_value)


def _plot_charts(matrix_data: common.MatrixData, chart_name: str, figure_width: float, figure_height: float, averages_by_method: dict[str, float]):
    plt.figure(figsize=(figure_width, figure_height))
    for label, averages in averages_by_method.items():
        plt.plot(THREAD_COUNTS, averages, marker='o', linewidth=2, label=label)

    plt.xlabel('Number of threads')
    plt.ylabel('Average runtime (s)')
    plt.title(f'{matrix_data.name} thread scaling')
    plt.xticks(THREAD_COUNTS)
    plt.grid(True, alpha=0.5)
    plt.legend()

    matrix_data.dataset_dir.mkdir(parents=True, exist_ok=True)
    chart_path = matrix_data.dataset_dir / f'{chart_name}_thread_speedup.png'
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f'saved {chart_path}')


def run_speedup_analysis(matrix_data: common.MatrixData, args: argparse.Namespace) -> None:
    matrix_data.load_data()
    speedup_table = typst.TypstTable(_get_speedup_table_row_columns())
    families = methods.get_method_families(args.method_family)
    averages_by_method = {}

    for family in families:
        for method_name, method in methods.generate_methods_by_family(family):
            speedup_table.add_row(
                _run_method_for_threads(
                    averages_by_method,
                    family=family,
                    matrix_data=matrix_data,
                    method_name=method_name,
                    method=method,
                    args=args,
                )
            )

    chart_name = args.method_family
    _plot_charts(
        matrix_data=matrix_data,
        chart_name=chart_name,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        averages_by_method=averages_by_method,
    )
    speedup_table.dump(matrix_data.dataset_dir / f'{chart_name}_speedup_summary.typ')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='*', default=common.SAMPLE_DATASETS)
    parser.add_argument('--datasets-dir', type=Path, default=common.DEFAULT_SAMPLE_DATASETS_DIR)
    parser.add_argument('--output-dir', type=Path, default=common.SCRIPTS / 'results_parallelization_speedup')
    parser.add_argument('--figure-width', type=float, default=common.DEFAULT_FIGURE_WIDTH)
    parser.add_argument('--figure-height', type=float, default=common.DEFAULT_FIGURE_HEIGHT)
    parser.add_argument('--method-family', choices=['all', 'global_spai', 'inner_outer'], default='all')
    parser.add_argument('--tries', '-N', type=int, required=True)
    parser.add_argument('--max-iterations', type=int, default=100)
    parser.add_argument('--tolerance', type=float, default=1e-9)
    parser.add_argument('--max-density', type=float, default=0.03)
    parser.add_argument('--num-threads', type=int, default=1)  # fake parameter - never used
    dropping = parser.add_mutually_exclusive_group()
    dropping.add_argument('--disable-dropping', dest='enable_dropping', action='store_false', default=True)
    args = parser.parse_args()

    if args.tries <= 0:
        raise ValueError('--tries / -N must be positive')

    common.update_output_folder(args)

    for dataset in common.select_sample_datasets(args):
        run_speedup_analysis(dataset, args)


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        sys.exit(str(e))
    except KeyboardInterrupt:
        sys.exit('\nInterrupted')
