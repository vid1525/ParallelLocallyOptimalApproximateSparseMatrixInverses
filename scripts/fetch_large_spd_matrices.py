import argparse
import shutil
import tarfile
import tempfile
import tqdm
import urllib.request
from pathlib import Path

import common


def download(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + '.part')
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dst)


def extract_mtx(archive: Path, output_path: Path):
    with tempfile.TemporaryDirectory(prefix='tmp-zip-') as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        with tarfile.open(archive, 'r:gz') as tar:
            tar.extractall(tmp_dir)

        files = list(tmp_dir.rglob('*.mtx'))
        if not files:
            raise RuntimeError(
                f'Unexpected number of .mtx files in archive {archive}, expected 1 - got {0 if files is None else len(files)}'
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(files[0]), output_path)


def fetch_dataset(ds: common.MatrixData, output_dir: Path):
    archive_filename = output_dir / f'{ds.name}-{ds.group}.tar.gz'
    matrix_filename = common.large_dataset_matrix_path(ds, output_dir)
    download(ds.source_url, archive_filename)
    extract_mtx(archive_filename, matrix_filename)
    if archive_filename.exists():
        archive_filename.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=common.DEFAULT_LARGE_DATASET_CONFIG)
    parser.add_argument('--output-dir', type=Path, default=common.DEFAULT_LARGE_DATASET_DIR)
    parser.add_argument('--datasets-dir', type=Path, default=common.DEFAULT_LARGE_DATASET_DIR)
    parser.add_argument('--datasets', nargs='*', help='Dataset names. Defaults to all config entries.')
    args = parser.parse_args()

    for dataset in tqdm.tqdm(common.select_large_datasets(args)):
        fetch_dataset(dataset, args.output_dir)


if __name__ == '__main__':
    main()
