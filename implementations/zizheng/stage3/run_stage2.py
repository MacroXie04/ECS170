"""Run preserved Stage 2 implementations in isolated, temporary working copies."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
MEMBERS = ("hongzhe", "aileen", "zizheng", "xiaoyu")
NESTED_TEMPLATE = "ECS170_Fall_2026_Source_Code_Template"


def inside(path, directory):
    return path == directory or directory in path.parents


def source_root(member, repo=ROOT):
    root = Path(repo) / "implementations" / member
    return root / NESTED_TEMPLATE if member == "xiaoyu" else root


def validate_output(output, repo, data):
    """Require a fresh destination outside source trees and the input dataset."""
    output, repo, data = Path(output).resolve(), Path(repo).resolve(), Path(data).resolve()
    # Runtime copies also use runs/; never follow it into data or source trees.
    if (repo / "runs").is_symlink():
        raise ValueError("The runs directory must not be a symlink")
    if inside(output, repo) and not inside(output, repo / "runs"):
        raise ValueError("Inside the repository, outputs must be under runs/")
    if inside(output, data) or inside(data, output):
        raise ValueError("Output and input dataset directories must not overlap")
    if inside(repo / "runs", data) or inside(data, repo / "runs"):
        raise ValueError("The dataset must not overlap the temporary runtime area runs/")
    if output.exists():
        raise ValueError(f"Output directory already exists; choose a new one: {output}")
    return output


def build_plan(implementation, data_dir, output_dir=None, repo=ROOT):
    """Validate paths and return a serializable plan without writing or training."""
    repo = Path(repo).resolve()
    if implementation not in (*MEMBERS, "all"):
        raise ValueError(f"Unknown implementation: {implementation}")
    data = Path(data_dir).expanduser()
    if not data.is_absolute():
        raise ValueError("--data-dir must be an absolute directory containing train.csv and test.csv")
    data = data.resolve()
    for name in ("train.csv", "test.csv"):
        if not (data / name).is_file():
            raise ValueError(f"Missing dataset file: {data / name}")
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        output_dir = repo / "runs" / stamp
    output = validate_output(Path(output_dir).expanduser(), repo, data)
    members = MEMBERS if implementation == "all" else (implementation,)
    entries = []
    for member in members:
        source = source_root(member, repo)
        if not source.is_dir():
            raise ValueError(f"Missing preserved implementation: {source}")
        entries.append({"member": member, "source": str(source), "output": str(output / member)})
    return {"repo": str(repo), "data": str(data), "output": str(output), "entries": entries}


def build_commands(member, data, output, python=sys.executable):
    """Commands run in that member's isolated root; no member modules are imported."""
    scripts = "script/stage_2_script/"
    if member == "hongzhe":
        return [[python, "-u", scripts + "run_experiments.py", "--device", "cpu",
                 "--data-dir", str(data), "--output-dir", str(output)]]
    if member == "aileen":
        return [[python, "-u", scripts + "script_mlp_tuned.py"]]
    if member == "zizheng":
        return [[python, "-u", scripts + "script_mlp.py", mode, "--device", "cpu",
                 "--data", str(data), "--output", str(output)] for mode in ("compare", "final")]
    if member == "xiaoyu":
        return [[python, "-u", scripts + "script_experiments.py", "--device", "cpu"]]
    raise ValueError(f"Unknown implementation: {member}")


def prepare_runtime(source, runtime, data, output):
    """Copy code, excluding archived outputs, and wire unchanged relative paths."""
    shutil.copytree(source, runtime, ignore=shutil.ignore_patterns(
        ".git", ".github", ".idea", ".venv", "venv", "__pycache__", "*.pyc",
        "data", "result", "results", "runs", "tests"))
    runtime, data, output = Path(runtime), Path(data), Path(output)
    (runtime / "data").mkdir()
    for filename in ("train.csv", "test.csv"):
        (runtime / "data" / filename).symlink_to(data / filename)
    (runtime / "data" / "stage_2_data").symlink_to(data, target_is_directory=True)
    (runtime / "result").mkdir()
    (runtime / "result" / "stage_2_result").symlink_to(output, target_is_directory=True)
    (runtime / "results").symlink_to(output, target_is_directory=True)


def execute_plan(plan):
    """Fail immediately on a child error; never resume or reuse saved results."""
    repo, data = Path(plan["repo"]), Path(plan["data"])
    output = validate_output(plan["output"], repo, data)
    runtime_parent = repo / "runs"
    # Keeping runtime copies below the repo lets original provenance code query
    # the integration commit with git, without putting .git in the copy.
    output.mkdir(parents=True, exist_ok=False)
    runtime_parent.mkdir(parents=True, exist_ok=True)
    (output / "launcher.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory(prefix=".stage2-runtime-", dir=runtime_parent) as temporary:
        for entry in plan["entries"]:
            member = entry["member"]
            member_output = Path(entry["output"])
            member_output.mkdir(exist_ok=False)
            runtime = Path(temporary) / member
            prepare_runtime(entry["source"], runtime, data, member_output)
            print(f"Running {member}; new results: {member_output}", flush=True)
            for command in build_commands(member, data, member_output):
                subprocess.run(command, cwd=runtime, env=env, check=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=(*MEMBERS, "all"), default="all")
    parser.add_argument("--data-dir", type=Path, help="Absolute folder containing train.csv and test.csv")
    parser.add_argument("--output-dir", type=Path,
                        help="Fresh run folder; results go in <folder>/<member> (default: runs/<UTC timestamp>)")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan without writing or training")
    parser.add_argument("--list", action="store_true", help="List implementations without importing or running them")
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(MEMBERS))
        return 0
    if args.data_dir is None:
        parser.error("--data-dir is required except with --list")
    try:
        plan = build_plan(args.implementation, args.data_dir, args.output_dir)
        if args.dry_run:
            preview = dict(plan)
            preview["commands"] = {entry["member"]: build_commands(entry["member"], plan["data"], entry["output"])
                                   for entry in plan["entries"]}
            print(json.dumps(preview, indent=2))
            return 0
        output = execute_plan(plan)
        print(f"Finished. Results: {output}")
        return 0
    except subprocess.CalledProcessError as error:
        print(f"Training failed (exit {error.returncode}); stopping. Partial results remain in {plan['output']}", file=sys.stderr)
        return error.returncode if error.returncode > 0 else 1
    except (ValueError, OSError) as error:
        parser.exit(1, f"Launcher error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
