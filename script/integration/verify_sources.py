"""Verify each imported branch file against its recorded SHA-256 digest.

Use --check-history in a full Git clone to also verify the four merge parents.
No model checkpoint or pickle is loaded or executed by this check.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def verify(root=ROOT, check_history=False):
    manifest = json.loads((root / "provenance.json").read_text())
    errors = []
    count = 0
    for member in manifest["implementations"]:
        base = root / member["directory"]
        for relative, expected in member["files"].items():
            path = base / relative
            if not path.is_file():
                errors.append(f"Missing: {path.relative_to(root)}")
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                errors.append(f"Changed: {path.relative_to(root)}")
            count += 1
        if check_history:
            result = subprocess.run(
                ["git", "merge-base", "--is-ancestor", member["commit"], "HEAD"],
                cwd=root, capture_output=True, text=True,
            )
            if result.returncode:
                errors.append(f"Missing ancestry for {member['branch']}: {member['commit']}")
    if errors:
        raise ValueError("\n".join(errors))
    return len(manifest["implementations"]), count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-history", action="store_true")
    args = parser.parse_args()
    try:
        members, count = verify(check_history=args.check_history)
    except (ValueError, OSError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Verified {count} original files across {members} implementations.")
    if args.check_history:
        print("All four source commits are ancestors of HEAD.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
