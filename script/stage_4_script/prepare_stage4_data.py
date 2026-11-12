#!/usr/bin/env python3
"""Extract the instructor ZIP, excluding macOS metadata; never execute archive code."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--destination', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    destination = args.destination.resolve()
    count = 0
    with zipfile.ZipFile(args.archive) as archive:
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe archive member path')
            if not path.parts or path.parts[0] != 'stage_4_data':
                continue
            if info.is_dir() or any(part.startswith('.') for part in path.parts):
                continue
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Archive symlinks are not supported')
            target = destination.joinpath(*path.parts)
            if not target.resolve().is_relative_to(destination):
                raise ValueError('Destination escapes data directory')
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open('wb') as sink:
                shutil.copyfileobj(source, sink)
            count += 1
    digest = hashlib.file_digest(args.archive.open('rb'), 'sha256').hexdigest()
    manifest = dict(source_archive=args.archive.name, sha256=digest, extracted_files=count)
    (destination / 'stage_4_data' / 'archive_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
