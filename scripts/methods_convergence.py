import argparse
from pathlib import Path
import sys

import common


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=common.SCRIPTS / 'results_sample')
    parser.add_argument('--datasets', nargs='*', default=common.SAMPLE_DATASETS)
    parser.add_argument('--datasets-dir', type=Path, default=common.DEFAULT_SAMPLE_DATASETS_DIR)
    parser.add_argument('--figure-width', type=float, default=common.DEFAULT_FIGURE_WIDTH)
    parser.add_argument('--figure-height', type=float, default=common.DEFAULT_FIGURE_HEIGHT)
    parser.add_argument('--method-family', choices=['all', 'global_spai', 'inner_outer'], default='all')
    parser.add_argument('--max-iterations', type=int, default=100)
    parser.add_argument('--tolerance', type=float, default=1e-9)
    parser.add_argument('--max-density', type=float, default=0.03)
    parser.add_argument('--write-preconditioners', action='store_true')
    parser.add_argument('--backward-error', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--max-backward-error-iterations', type=int, default=150)
    parser.add_argument('--backward-error-seed', type=int, default=42)
    parser.add_argument('--backward-error-realizations', type=int, default=1)
    dropping = parser.add_mutually_exclusive_group()
    dropping.add_argument('--disable-dropping', dest='enable_dropping', action='store_false', default=True)
    parser.add_argument('--num-threads', type=int, default=1)
    args = parser.parse_args()
    if args.max_backward_error_iterations < 0:
        parser.error('--max-backward-error-iterations must be nonnegative')
    if args.backward_error_realizations <= 0:
        parser.error('--backward-error-realizations must be positive')
    if args.backward_error_seed < 0:
        parser.error('--backward-error-seed must be nonnegative')

    common.update_output_folder(args)

    for dataset in common.select_sample_datasets(args):
        common.run_methods(dataset, args)


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        sys.exit(str(e))
    except KeyboardInterrupt:
        sys.exit('\nInterrupted.')
