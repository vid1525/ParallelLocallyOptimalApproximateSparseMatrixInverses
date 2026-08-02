import argparse
from pathlib import Path
import sys

import common


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets-dir', type=Path, default=common.DEFAULT_LARGE_DATASET_DIR)
    parser.add_argument('--config', type=Path, default=common.DEFAULT_LARGE_DATASET_CONFIG)
    parser.add_argument('--output-dir', type=Path, default=common.SCRIPTS / 'results_large')
    parser.add_argument('--figure-width', type=float, default=common.DEFAULT_FIGURE_WIDTH)
    parser.add_argument('--figure-height', type=float, default=common.DEFAULT_FIGURE_HEIGHT)
    parser.add_argument('--datasets', nargs='*', help='Dataset names, defaults to all config entries')
    parser.add_argument('--method-family', choices=['all', 'global_spai', 'inner_outer'], default='all')
    parser.add_argument('--max-iterations', type=int, default=20)
    parser.add_argument('--tolerance', type=float, default=1e-9)
    parser.add_argument('--max-density', type=float, default=1e-5)
    dropping = parser.add_mutually_exclusive_group()
    dropping.add_argument('--disable-dropping', dest='enable_dropping', action='store_false', default=True)
    parser.add_argument('--num-threads', type=int, default=16)
    args = parser.parse_args()

    common.update_output_folder(args)

    for dataset in common.select_large_datasets(args):
        common.run_methods(dataset, args)


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        sys.exit(str(e))
    except KeyboardInterrupt:
        sys.exit('\nInterrupted')
