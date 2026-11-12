"""Launcher boundaries and process isolation, without ML training or course data."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("stage2_launcher", REPO / "run_stage2.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.data = self.root / "data"
        self.data.mkdir()
        for name in ("train.csv", "test.csv"):
            (self.data / name).write_text("0,1,2\n")
        for member in launcher.MEMBERS:
            source = launcher.source_root(member, self.repo)
            (source / "local_code").mkdir(parents=True)
            (source / "local_code" / "__init__.py").write_text(f"OWNER = {member!r}\n")
            (source / "script" / "stage_2_script").mkdir(parents=True)
            for archive in ("result", "results", "data"):
                (source / archive).mkdir()
                (source / archive / "ARCHIVE").write_text("must remain untouched")

    def plan(self, member="all", output=None):
        return launcher.build_plan(member, self.data, output or self.repo / "runs" / "new", self.repo)

    def test_plan_is_read_only_and_dispatches_each_original_entry_point(self):
        plan = self.plan()
        self.assertFalse((self.repo / "runs").exists())
        commands = {e["member"]: launcher.build_commands(e["member"], self.data, e["output"])
                    for e in plan["entries"]}
        self.assertEqual([c[3] for c in commands["zizheng"]], ["compare", "final"])
        self.assertIn("--data-dir", commands["hongzhe"][0])
        self.assertEqual(len(commands["aileen"][0]), 3)
        self.assertTrue(commands["aileen"][0][2].endswith("script_mlp_tuned.py"))
        self.assertNotIn("--only", commands["xiaoyu"][0])
        self.assertNotIn("--epochs", commands["xiaoyu"][0])
        self.assertIn(launcher.NESTED_TEMPLATE, plan["entries"][-1]["source"])

    def test_rejects_relative_missing_data_and_existing_output(self):
        with self.assertRaises(ValueError):
            launcher.build_plan("all", Path("relative"), repo=self.repo)
        (self.data / "test.csv").unlink()
        with self.assertRaises(ValueError):
            self.plan()
        (self.data / "test.csv").write_text("0,1,2\n")
        output = self.root / "already-present"
        output.mkdir()
        with self.assertRaises(ValueError):
            self.plan(output=output)

    def test_rejects_source_and_data_outputs_including_symlink_alias(self):
        for output in (self.repo / "implementations" / "new", self.repo / "script" / "new",
                       self.repo / "tests" / "new", self.data / "new"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                self.plan(output=output)
        (self.repo / "runs").mkdir()
        (self.repo / "runs" / "alias").symlink_to(self.repo / "implementations", target_is_directory=True)
        with self.assertRaises(ValueError):
            self.plan(output=self.repo / "runs" / "alias" / "new")

    def test_rejects_symlinked_runtime_parent(self):
        (self.repo / "runs").symlink_to(self.data, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.plan(output=self.root / "new-output")

    def test_runtime_excludes_archives_and_links_only_fresh_outputs(self):
        source = launcher.source_root("hongzhe", self.repo)
        runtime = self.root / "runtime"
        output = self.root / "output"
        output.mkdir()
        launcher.prepare_runtime(source, runtime, self.data, output)
        self.assertFalse((runtime / "results" / "ARCHIVE").exists())
        self.assertFalse((runtime / "result" / "ARCHIVE").exists())
        self.assertEqual((runtime / "data" / "train.csv").resolve(), (self.data / "train.csv").resolve())
        self.assertEqual((runtime / "data" / "stage_2_data").resolve(), self.data.resolve())
        for result_path in (runtime / "results", runtime / "result" / "stage_2_result"):
            self.assertEqual(result_path.resolve(), output.resolve())
        self.assertEqual((source / "results" / "ARCHIVE").read_text(), "must remain untouched")

    def install_fake_scripts(self, fail_member=None):
        for member in launcher.MEMBERS:
            source = launcher.source_root(member, self.repo)
            filename = launcher.build_commands(member, self.data, self.root / "placeholder")[0][2]
            body = """import json, sys
from pathlib import Path
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
from local_code import OWNER
assert not (root / 'results' / 'ARCHIVE').exists()
assert (root / 'data' / 'train.csv').is_file()
assert (root / 'data' / 'stage_2_data' / 'test.csv').is_file()
output = root / 'results'
mode = sys.argv[1] if OWNER == 'zizheng' else 'run'
if mode == 'final':
    assert (output / 'compare.json').is_file()
(output / (mode + '.json')).write_text(json.dumps({'owner': OWNER, 'runtime': str(root)}))
"""
            if member == fail_member:
                body += "raise SystemExit(7)\n"
            (source / filename).write_text(body)

    def test_real_subprocesses_are_isolated_and_outputs_survive_cleanup(self):
        self.install_fake_scripts()
        plan = self.plan()
        launcher.execute_plan(plan)
        for entry in plan["entries"]:
            name = "final.json" if entry["member"] == "zizheng" else "run.json"
            record = json.loads((Path(entry["output"]) / name).read_text())
            self.assertEqual(record["owner"], entry["member"])
            self.assertFalse(Path(record["runtime"]).exists())
            self.assertTrue((Path(entry["source"]) / "results" / "ARCHIVE").is_file())

    def test_failure_stops_following_members_and_prevents_output_reuse(self):
        self.install_fake_scripts(fail_member="hongzhe")
        plan = self.plan()
        with self.assertRaises(subprocess.CalledProcessError) as error:
            launcher.execute_plan(plan)
        self.assertEqual(error.exception.returncode, 7)
        self.assertFalse((Path(plan["output"]) / "aileen").exists())
        self.assertFalse(list((self.repo / "runs").glob(".stage2-runtime-*")))
        with self.assertRaises(ValueError):
            launcher.execute_plan(plan)

    def test_fresh_runs_directory_can_itself_be_the_output(self):
        self.install_fake_scripts()
        output = self.repo / "runs"
        launcher.execute_plan(self.plan("aileen", output=output))
        self.assertTrue((output / "aileen" / "run.json").is_file())
        self.assertFalse(list(output.glob(".stage2-runtime-*")))

    def test_cli_help_and_list_never_import_or_launch_member_code(self):
        with mock.patch.object(launcher, "execute_plan", side_effect=AssertionError("must not run")):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(launcher.main(["--list"]), 0)
            self.assertIn("aileen", output.getvalue())
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stopped:
                launcher.main(["--help"])
            self.assertEqual(stopped.exception.code, 0)

    def test_cli_dry_run_does_not_execute_or_create_output(self):
        plan = self.plan("aileen")
        with mock.patch.object(launcher, "build_plan", return_value=plan), \
             mock.patch.object(launcher, "execute_plan", side_effect=AssertionError("must not run")), \
             contextlib.redirect_stdout(io.StringIO()) as text:
            self.assertEqual(launcher.main(["--implementation", "aileen", "--data-dir", str(self.data), "--dry-run"]), 0)
        self.assertIn("script_mlp_tuned.py", text.getvalue())
        self.assertFalse(Path(plan["output"]).exists())


if __name__ == "__main__":
    unittest.main()
