"""Download the instructor's public stage_3_data.zip and verify its checksums.

Uses only the public link in the course project document; no account is needed.
The pickles are extracted but never loaded here. Already-valid data is left alone.
"""
import argparse
from pathlib import Path
import sys
import tempfile
import urllib.request
import zipfile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from local_code.stage_3_code.Dataset_Loader import DATASET_SPECS  # noqa: E402
from local_code.stage_3_code.Runtime import sha256_file  # noqa: E402

DRIVE_FILE_ID = '1-udd9jkldM1gEQ-OO-9HKXds42AiLOL6'
DOWNLOAD_URL = ('https://drive.usercontent.google.com/download'
                f'?id={DRIVE_FILE_ID}&export=download&confirm=t')
ARCHIVE_SHA256 = 'bc7592ec3117542d67625695ee68b7df1a2d9eb9eab39b9f4b44bc415b030d04'
EXTRA_FILES = ('ReadMe.txt', 'ReadMe.pdf', 'script_data_loader.py')


def data_is_valid(folder):
    return all((folder / spec.file_name).is_file()
               and sha256_file(folder / spec.file_name) == spec.sha256
               for spec in DATASET_SPECS.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output-dir', type=Path, default=REPO / 'data' / 'stage_3_data')
    args = parser.parse_args()
    if data_is_valid(args.output_dir):
        print(f'{args.output_dir}: checksums match; nothing to do')
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / 'stage_3_data.zip'
        print('downloading the public Stage 3 archive...')
        request = urllib.request.Request(DOWNLOAD_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(request) as response, open(archive, 'wb') as handle:
            while chunk := response.read(1 << 22):
                handle.write(chunk)
        if sha256_file(archive) != ARCHIVE_SHA256:
            raise SystemExit('Downloaded archive does not match the recorded checksum')
        wanted = [spec.file_name for spec in DATASET_SPECS.values()] + list(EXTRA_FILES)
        with zipfile.ZipFile(archive) as bundle:
            for name in wanted:
                (args.output_dir / name).write_bytes(bundle.read(f'stage_3_data/{name}'))
    if not data_is_valid(args.output_dir):
        raise SystemExit('Extracted data does not match the recorded checksums')
    print(f'{args.output_dir}: ready')


if __name__ == '__main__':
    main()
