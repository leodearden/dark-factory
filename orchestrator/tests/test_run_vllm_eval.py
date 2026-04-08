"""Tests for the multi-task vLLM eval launcher (scripts/run_vllm_eval.py)."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

# Stub out runpod_toolkit BEFORE importing the launcher. The real package has
# a transitive paramiko dependency that's not installed in the orchestrator
# test venv, and we never need real RunPod calls in tests anyway — pod
# lifecycle is mocked at the launcher level.
_rpt = types.ModuleType("runpod_toolkit")
_rpt_config = types.ModuleType("runpod_toolkit.config")
_rpt_compute = types.ModuleType("runpod_toolkit.compute")


class _StubRunPodConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _StubRunPodClient:
    def __init__(self, config):
        self.config = config


class _StubPodStatus:
    RUNNING = "RUNNING"


_rpt_config.RunPodConfig = _StubRunPodConfig  # type: ignore[attr-defined]
_rpt_compute.RunPodClient = _StubRunPodClient  # type: ignore[attr-defined]
_rpt_compute.PodStatus = _StubPodStatus  # type: ignore[attr-defined]

sys.modules.setdefault("runpod_toolkit", _rpt)
sys.modules.setdefault("runpod_toolkit.config", _rpt_config)
sys.modules.setdefault("runpod_toolkit.compute", _rpt_compute)

# Make scripts/run_vllm_eval.py importable. The launcher modifies sys.path
# itself at import time to find runpod_toolkit + orchestrator.
_SCRIPTS = Path(__file__).parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_vllm_eval as launcher  # type: ignore[import-not-found]  # noqa: E402
from run_vllm_eval import (  # type: ignore[import-not-found]  # noqa: E402
    EvalSummary,
    PodHandle,
    TaskSpecNotFound,
    _build_ssh_tunnel_argv,
    _pick_free_port,
    compute_exit_code,
    find_new_result_file,
    load_task_spec,
    parse_args,
    parse_result_file,
    preflight_baseline,
    print_summary_table,
    resolve_task_ids,
    tear_down_pod,
    wait_for_vllm,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def tmp_repo(tmp_path: Path) -> tuple[Path, str]:
    """Tiny git repo with one commit. Returns (repo_path, commit_sha)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], cwd=repo)
    _git(["config", "user.email", "test@example.com"], cwd=repo)
    _git(["config", "user.name", "Test User"], cwd=repo)
    _git(["config", "commit.gpgsign", "false"], cwd=repo)

    (repo / "README.md").write_text("hello\n")
    _git(["add", "README.md"], cwd=repo)
    _git(["commit", "-q", "-m", "initial"], cwd=repo)
    sha = _git(["rev-parse", "HEAD"], cwd=repo)
    return repo, sha


def _make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        config="reap-139b-nvfp4-new",
        task=None,
        tasks=None,
        all_tasks=False,
        port=8100,
        datacenter="US-NC-1",
        no_volume=False,
        image=None,
        gpu_type=None,
        concurrency=1,
        verify_baseline_clean="strict",
        task_timeout_min=70,
        stop_on_first_failure=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# resolve_task_ids
# ---------------------------------------------------------------------------


class TestResolveTaskIds:
    def test_single_task_alias(self):
        args = _make_args(task="df_task_12")
        assert resolve_task_ids(args) == ["df_task_12"]

    def test_tasks_csv(self):
        args = _make_args(tasks="df_task_12,df_task_13,df_task_18")
        assert resolve_task_ids(args) == ["df_task_12", "df_task_13", "df_task_18"]

    def test_tasks_csv_strips_whitespace(self):
        args = _make_args(tasks="df_task_12, df_task_13 ,df_task_18")
        assert resolve_task_ids(args) == ["df_task_12", "df_task_13", "df_task_18"]

    def test_all_tasks_globs_dir(self, tmp_path, monkeypatch):
        fake_tasks = tmp_path / "tasks"
        fake_tasks.mkdir()
        for name in (
            "df_task_12.json",
            "df_task_18.json",
            "reify_task_27.json",
            "not_a_task.json",
        ):
            (fake_tasks / name).write_text("{}")
        monkeypatch.setattr(launcher, "TASKS_DIR", fake_tasks)

        args = _make_args(all_tasks=True)
        result = resolve_task_ids(args)
        # Both df_task_*.json and reify_task_*.json are picked up.
        # Files not matching <project>_task_*.json are excluded.
        assert result == ["df_task_12", "df_task_18", "reify_task_27"]

    def test_no_selection_raises(self):
        args = _make_args()
        with pytest.raises(ValueError, match="No task selection"):
            resolve_task_ids(args)

    def test_parse_args_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            parse_args(
                [
                    "--config",
                    "reap-139b-nvfp4-new",
                    "--task",
                    "df_task_12",
                    "--tasks",
                    "df_task_13",
                ]
            )


# ---------------------------------------------------------------------------
# load_task_spec
# ---------------------------------------------------------------------------


class TestLoadTaskSpec:
    def test_loads_real_spec(self, monkeypatch, tmp_path):
        fake_tasks = tmp_path / "tasks"
        fake_tasks.mkdir()
        (fake_tasks / "df_task_99.json").write_text(
            json.dumps({"id": "df_task_99", "pre_task_commit": "abc"})
        )
        monkeypatch.setattr(launcher, "TASKS_DIR", fake_tasks)

        spec = load_task_spec("df_task_99")
        assert spec["id"] == "df_task_99"
        assert spec["pre_task_commit"] == "abc"

    def test_missing_spec_raises(self, monkeypatch, tmp_path):
        fake_tasks = tmp_path / "tasks"
        fake_tasks.mkdir()
        monkeypatch.setattr(launcher, "TASKS_DIR", fake_tasks)

        with pytest.raises(TaskSpecNotFound):
            load_task_spec("df_task_999")


# ---------------------------------------------------------------------------
# find_new_result_file + parse_result_file
# ---------------------------------------------------------------------------


def _write_result(
    results_dir: Path, task_id: str, config_name: str, run_id: str, **extra
) -> Path:
    payload = {
        "task_id": task_id,
        "config_name": config_name,
        "run_id": run_id,
        "outcome": extra.pop("outcome", "done"),
        "metrics": extra.pop("metrics", {"cost_usd": 1.23, "tests_pass": True}),
    }
    payload.update(extra)
    p = results_dir / f"{task_id}__{config_name}__{run_id}.json"
    p.write_text(json.dumps(payload))
    return p


class TestFindNewResultFile:
    def test_returns_newest_after_watermark(self, tmp_path, monkeypatch):
        results = tmp_path / "results"
        results.mkdir()
        monkeypatch.setattr(launcher, "RESULTS_DIR", results)

        old = _write_result(results, "df_task_12", "cfg", "old00000")
        watermark = old.stat().st_mtime
        time.sleep(0.05)
        new = _write_result(results, "df_task_12", "cfg", "new00000")

        found = find_new_result_file("df_task_12", "cfg", watermark)
        assert found == new

    def test_ignores_files_at_or_before_watermark(self, tmp_path, monkeypatch):
        results = tmp_path / "results"
        results.mkdir()
        monkeypatch.setattr(launcher, "RESULTS_DIR", results)

        only = _write_result(results, "df_task_12", "cfg", "only00000")
        watermark = only.stat().st_mtime + 1.0  # in the future

        assert find_new_result_file("df_task_12", "cfg", watermark) is None

    def test_returns_none_when_no_new_file(self, tmp_path, monkeypatch):
        results = tmp_path / "results"
        results.mkdir()
        monkeypatch.setattr(launcher, "RESULTS_DIR", results)
        assert find_new_result_file("df_task_99", "cfg", 0.0) is None

    def test_isolates_by_task_id(self, tmp_path, monkeypatch):
        results = tmp_path / "results"
        results.mkdir()
        monkeypatch.setattr(launcher, "RESULTS_DIR", results)

        _write_result(results, "df_task_13", "cfg", "abcd1234")
        assert find_new_result_file("df_task_12", "cfg", 0.0) is None

    def test_isolates_by_config_name(self, tmp_path, monkeypatch):
        results = tmp_path / "results"
        results.mkdir()
        monkeypatch.setattr(launcher, "RESULTS_DIR", results)

        _write_result(results, "df_task_12", "other-cfg", "abcd1234")
        assert find_new_result_file("df_task_12", "cfg", 0.0) is None


class TestFindNewResultFileExclude:
    """exclude_paths fixes the ext4 1s-mtime-granularity collision risk
    that surfaces when multiple concurrent tasks write into the same
    results dir.
    """

    def test_excludes_pre_files_paths(self, tmp_path, monkeypatch):
        results = tmp_path / "results"
        results.mkdir()
        monkeypatch.setattr(launcher, "RESULTS_DIR", results)

        a = _write_result(results, "df_task_12", "cfg", "aaaa0000")
        watermark = a.stat().st_mtime - 1.0  # both files are above the watermark
        time.sleep(0.05)
        b = _write_result(results, "df_task_12", "cfg", "bbbb0000")

        # Without exclude_paths, the newer file (b) wins.
        assert find_new_result_file("df_task_12", "cfg", watermark) == b

        # With a in exclude_paths, b still wins (same answer here, since b is
        # newer). The interesting case is when a happens to have a *later*
        # mtime — exclude_paths must still suppress it.
        import os

        os.utime(a, (a.stat().st_atime, b.stat().st_mtime + 1.0))
        assert (
            find_new_result_file(
                "df_task_12", "cfg", watermark, exclude_paths={a}
            )
            == b
        )

    def test_exclude_paths_default_none(self, tmp_path, monkeypatch):
        """Old 3-positional-arg call sites still work."""
        results = tmp_path / "results"
        results.mkdir()
        monkeypatch.setattr(launcher, "RESULTS_DIR", results)

        only = _write_result(results, "df_task_12", "cfg", "only00000")
        # No exclude_paths arg at all — must still find the file.
        assert find_new_result_file("df_task_12", "cfg", 0.0) == only

    def test_concurrent_collision_simulation(self, tmp_path, monkeypatch):
        """5 result files, all with the same mtime, each task isolated by
        its own pre_files snapshot."""
        import os

        results = tmp_path / "results"
        results.mkdir()
        monkeypatch.setattr(launcher, "RESULTS_DIR", results)

        # Write 5 files all with mtime = 1000.0
        files = {}
        for i in range(5):
            tid = f"df_task_1{i}"
            p = _write_result(results, tid, "cfg", f"r{i:08d}")
            os.utime(p, (1000.0, 1000.0))
            files[tid] = p

        # Each "task" sees its own file as pre-existing-empty (i.e. nothing
        # to exclude — it was the writer of its own file). The other 4 files
        # were written before this task's watermark. Each must find ITS own
        # result file even though all 5 share the same mtime.
        for tid, expected in files.items():
            other_files = {p for t, p in files.items() if t != tid}
            found = find_new_result_file(
                tid,
                "cfg",
                999.0,  # below 1000.0 so all candidates pass the mtime gate
                exclude_paths=other_files,
            )
            assert found == expected, f"task {tid} got {found}, expected {expected}"


class TestParseResultFile:
    def test_extracts_outcome_and_cost(self, tmp_path):
        p = tmp_path / "r.json"
        p.write_text(
            json.dumps(
                {
                    "outcome": "done",
                    "metrics": {
                        "cost_usd": 12.34,
                        "tests_pass": True,
                        "lint_clean": True,
                        "typecheck_clean": True,
                    },
                    "run_id": "abcd1234",
                }
            )
        )
        data = parse_result_file(p)
        assert data["outcome"] == "done"
        assert data["metrics"]["cost_usd"] == 12.34
        assert data["run_id"] == "abcd1234"

    def test_handles_missing_metrics_gracefully(self, tmp_path):
        p = tmp_path / "r.json"
        p.write_text(json.dumps({"outcome": "blocked", "run_id": "x"}))
        data = parse_result_file(p)
        assert data["outcome"] == "blocked"
        assert data["metrics"] == {}
        assert data["run_id"] == "x"

    def test_handles_null_metrics(self, tmp_path):
        p = tmp_path / "r.json"
        p.write_text(json.dumps({"outcome": "done", "metrics": None}))
        data = parse_result_file(p)
        assert data["metrics"] == {}


# ---------------------------------------------------------------------------
# Pod lifecycle (mocked)
# ---------------------------------------------------------------------------


class _FakePod:
    def __init__(self, pod_id: str = "pod-fake-1"):
        self.id = pod_id
        self.ssh_host = "1.2.3.4"
        self.ssh_port = 22000


class _FakeClient:
    def __init__(self):
        self.create_calls: list[dict] = []
        self.terminate_calls: list[str] = []

    def create_pod(self, **kwargs):
        self.create_calls.append(kwargs)
        return _FakePod()

    def wait_for_pod(self, **kwargs):
        return _FakePod(pod_id=kwargs.get("pod_id", "pod-fake-1"))

    def terminate_pod(self, pod_id: str) -> bool:
        self.terminate_calls.append(pod_id)
        return True


class _FakeTunnel:
    """Stand-in for the SSH tunnel Popen — supports terminate/wait/kill."""

    def __init__(self):
        self.terminated = False

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.terminated = True


def _patch_pod_infra(monkeypatch, *, vllm_healthy_first=True):
    """Patch RunPodClient + SSH tunnel + wait_for_vllm so bring_up_pod returns fast.

    Real ``subprocess.Popen`` is preserved for everything other than the SSH
    tunnel command, so preflight_baseline's git calls still work end-to-end.
    """
    fake_client = _FakeClient()
    monkeypatch.setattr(launcher, "RunPodClient", lambda config: fake_client)
    monkeypatch.setattr(
        launcher,
        "wait_for_vllm",
        lambda port, expected_model, timeout=900: True,
    )
    monkeypatch.setattr(launcher, "vllm_healthy", lambda port, timeout=5: vllm_healthy_first)

    real_popen = subprocess.Popen

    def smart_popen(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd and cmd[0] == "ssh" and "-N" in cmd:
            return _FakeTunnel()
        return real_popen(cmd, *args, **kwargs)

    monkeypatch.setattr(launcher.subprocess, "Popen", smart_popen)

    # time.sleep noop so the 3s SSH-tunnel-stabilize sleep doesn't slow tests
    monkeypatch.setattr(launcher.time, "sleep", lambda *_: None)

    return fake_client


def _patch_subprocess_run_success(monkeypatch, results_dir: Path):
    """Patch subprocess.run so 'orchestrator eval' calls write a fake result file."""
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if (
            isinstance(cmd, list)
            and len(cmd) >= 4
            and cmd[:4] == ["uv", "run", "orchestrator", "eval"]
        ):
            # Extract --task and --config-name from the command list
            task_arg = cmd[cmd.index("--task") + 1]
            config_name = cmd[cmd.index("--config-name") + 1]
            task_id = Path(task_arg).stem
            results_dir.mkdir(parents=True, exist_ok=True)
            run_id = f"r{int(time.time() * 1000) % 100000000:08d}"
            _write_result(results_dir, task_id, config_name, run_id)
            return SimpleNamespace(returncode=0)
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)


class TestPodLifecycle:
    @pytest.mark.parametrize("concurrency", [1, 3])
    def test_multi_task_terminates_pod_once(
        self, monkeypatch, tmp_path, concurrency
    ):
        results = tmp_path / "results"
        monkeypatch.setattr(launcher, "RESULTS_DIR", results)
        monkeypatch.setattr(launcher, "EVAL_LOG_DIR", tmp_path / "logs")

        fake_tasks = tmp_path / "tasks"
        fake_tasks.mkdir()
        for tid in ("df_task_12", "df_task_13", "df_task_18"):
            (fake_tasks / f"{tid}.json").write_text(
                json.dumps(
                    {
                        "id": tid,
                        "pre_task_commit": "deadbeef",
                        "verify_commands": {"lint": "true", "typecheck": "true"},
                    }
                )
            )
        monkeypatch.setattr(launcher, "TASKS_DIR", fake_tasks)

        fake_client = _patch_pod_infra(monkeypatch)
        _patch_subprocess_run_success(monkeypatch, results)

        # Skip preflight (no real lint/typecheck environment).
        rc = launcher.main(
            [
                "--config",
                "reap-139b-nvfp4-new",
                "--tasks",
                "df_task_12,df_task_13,df_task_18",
                "--verify-baseline-clean",
                "skip",
                "--concurrency",
                str(concurrency),
            ]
        )

        assert rc == 0
        assert len(fake_client.create_calls) == 1
        assert fake_client.terminate_calls == ["pod-fake-1"]

    def test_pod_terminated_when_middle_task_crashes(self, monkeypatch, tmp_path):
        results = tmp_path / "results"
        monkeypatch.setattr(launcher, "RESULTS_DIR", results)

        fake_tasks = tmp_path / "tasks"
        fake_tasks.mkdir()
        for tid in ("df_task_12", "df_task_13", "df_task_18"):
            (fake_tasks / f"{tid}.json").write_text(
                json.dumps({"id": tid, "pre_task_commit": "deadbeef"})
            )
        monkeypatch.setattr(launcher, "TASKS_DIR", fake_tasks)

        fake_client = _patch_pod_infra(monkeypatch)

        crash_count = {"n": 0}

        def fake_run(cmd, *args, **kwargs):
            if (
                isinstance(cmd, list)
                and cmd[:4] == ["uv", "run", "orchestrator", "eval"]
            ):
                task_arg = cmd[cmd.index("--task") + 1]
                config_name = cmd[cmd.index("--config-name") + 1]
                task_id = Path(task_arg).stem
                if task_id == "df_task_13":
                    crash_count["n"] += 1
                    raise RuntimeError("simulated subprocess crash")
                results.mkdir(parents=True, exist_ok=True)
                _write_result(results, task_id, config_name, f"r{crash_count['n']:08d}")
                return SimpleNamespace(returncode=0)
            return subprocess.run(cmd, *args, **kwargs)

        monkeypatch.setattr(launcher.subprocess, "run", fake_run)

        rc = launcher.main(
            [
                "--config",
                "reap-139b-nvfp4-new",
                "--tasks",
                "df_task_12,df_task_13,df_task_18",
                "--verify-baseline-clean",
                "skip",
            ]
        )

        # Exit 1 because one task crashed
        assert rc == 1
        # All 3 tasks attempted (loop kept going past the crash)
        assert crash_count["n"] == 1
        # Pod still terminated exactly once
        assert fake_client.terminate_calls == ["pod-fake-1"]

    def test_pod_terminated_on_keyboard_interrupt(self, monkeypatch, tmp_path):
        results = tmp_path / "results"
        monkeypatch.setattr(launcher, "RESULTS_DIR", results)

        fake_tasks = tmp_path / "tasks"
        fake_tasks.mkdir()
        (fake_tasks / "df_task_12.json").write_text(
            json.dumps({"id": "df_task_12", "pre_task_commit": "deadbeef"})
        )
        monkeypatch.setattr(launcher, "TASKS_DIR", fake_tasks)

        fake_client = _patch_pod_infra(monkeypatch)

        def fake_run(cmd, *args, **kwargs):
            if (
                isinstance(cmd, list)
                and cmd[:4] == ["uv", "run", "orchestrator", "eval"]
            ):
                raise KeyboardInterrupt
            return subprocess.run(cmd, *args, **kwargs)

        monkeypatch.setattr(launcher.subprocess, "run", fake_run)

        with pytest.raises(KeyboardInterrupt):
            launcher.main(
                [
                    "--config",
                    "reap-139b-nvfp4-new",
                    "--task",
                    "df_task_12",
                    "--verify-baseline-clean",
                    "skip",
                ]
            )

        # finally still terminated the pod
        assert fake_client.terminate_calls == ["pod-fake-1"]

    def test_pod_bringup_failure_short_circuits(self, monkeypatch, tmp_path):
        results = tmp_path / "results"
        monkeypatch.setattr(launcher, "RESULTS_DIR", results)

        fake_tasks = tmp_path / "tasks"
        fake_tasks.mkdir()
        (fake_tasks / "df_task_12.json").write_text(
            json.dumps({"id": "df_task_12", "pre_task_commit": "deadbeef"})
        )
        monkeypatch.setattr(launcher, "TASKS_DIR", fake_tasks)

        # Patch infra so wait_for_vllm fails (simulates vLLM never becoming healthy)
        fake_client = _FakeClient()
        monkeypatch.setattr(launcher, "RunPodClient", lambda config: fake_client)
        monkeypatch.setattr(
            launcher,
            "wait_for_vllm",
            lambda port, expected_model, timeout=900: False,
        )
        fake_tunnel = SimpleNamespace(
            terminate=lambda: None,
            wait=lambda timeout=None: 0,
            kill=lambda: None,
        )
        monkeypatch.setattr(
            launcher.subprocess, "Popen", lambda *args, **kwargs: fake_tunnel
        )
        monkeypatch.setattr(launcher.time, "sleep", lambda *_: None)

        # Subprocess.run for eval should NEVER be called.
        eval_called = {"n": 0}

        def fake_run(cmd, *args, **kwargs):
            if (
                isinstance(cmd, list)
                and cmd[:4] == ["uv", "run", "orchestrator", "eval"]
            ):
                eval_called["n"] += 1
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(launcher.subprocess, "run", fake_run)

        rc = launcher.main(
            [
                "--config",
                "reap-139b-nvfp4-new",
                "--task",
                "df_task_12",
                "--verify-baseline-clean",
                "skip",
            ]
        )

        assert rc == 1
        assert eval_called["n"] == 0
        # Pod was created then torn down by bring_up_pod's own except path
        assert fake_client.terminate_calls == ["pod-fake-1"]

    def test_single_task_path_runs_subprocess_once(self, monkeypatch, tmp_path):
        results = tmp_path / "results"
        monkeypatch.setattr(launcher, "RESULTS_DIR", results)

        fake_tasks = tmp_path / "tasks"
        fake_tasks.mkdir()
        (fake_tasks / "df_task_12.json").write_text(
            json.dumps({"id": "df_task_12", "pre_task_commit": "deadbeef"})
        )
        monkeypatch.setattr(launcher, "TASKS_DIR", fake_tasks)

        fake_client = _patch_pod_infra(monkeypatch)

        eval_calls = {"n": 0}
        real_run = subprocess.run

        def fake_run(cmd, *args, **kwargs):
            if (
                isinstance(cmd, list)
                and cmd[:4] == ["uv", "run", "orchestrator", "eval"]
            ):
                eval_calls["n"] += 1
                task_arg = cmd[cmd.index("--task") + 1]
                config_name = cmd[cmd.index("--config-name") + 1]
                task_id = Path(task_arg).stem
                results.mkdir(parents=True, exist_ok=True)
                _write_result(results, task_id, config_name, "single001")
                return SimpleNamespace(returncode=0)
            return real_run(cmd, *args, **kwargs)

        monkeypatch.setattr(launcher.subprocess, "run", fake_run)

        rc = launcher.main(
            [
                "--config",
                "reap-139b-nvfp4-new",
                "--task",
                "df_task_12",
                "--verify-baseline-clean",
                "skip",
            ]
        )

        assert rc == 0
        assert eval_calls["n"] == 1
        assert fake_client.terminate_calls == ["pod-fake-1"]


# ---------------------------------------------------------------------------
# Per-task log file plumbing
# ---------------------------------------------------------------------------


class TestPerTaskLogFiles:
    """Concurrent mode redirects subprocess output to per-task log files;
    sequential mode keeps streaming live to the launcher TTY.
    """

    def _setup_one_task(self, monkeypatch, tmp_path):
        results = tmp_path / "results"
        monkeypatch.setattr(launcher, "RESULTS_DIR", results)

        fake_tasks = tmp_path / "tasks"
        fake_tasks.mkdir()
        (fake_tasks / "df_task_12.json").write_text(
            json.dumps(
                {
                    "id": "df_task_12",
                    "pre_task_commit": "deadbeef",
                    "verify_commands": {"lint": "true", "typecheck": "true"},
                }
            )
        )
        monkeypatch.setattr(launcher, "TASKS_DIR", fake_tasks)

        log_dir = tmp_path / "eval-logs"
        monkeypatch.setattr(launcher, "EVAL_LOG_DIR", log_dir)
        return results, log_dir

    def test_log_file_created_in_concurrent_mode(self, monkeypatch, tmp_path):
        results, log_dir = self._setup_one_task(monkeypatch, tmp_path)

        fake_client = _patch_pod_infra(monkeypatch)

        captured_kwargs: list[dict] = []

        def fake_run(cmd, *args, **kwargs):
            if (
                isinstance(cmd, list)
                and cmd[:4] == ["uv", "run", "orchestrator", "eval"]
            ):
                captured_kwargs.append(kwargs)
                # Write a fake "subprocess wrote to my stdout" line into the log fh.
                fh = kwargs.get("stdout")
                if fh is not None and hasattr(fh, "write"):
                    fh.write("hello from fake subprocess\n")
                    fh.flush()
                task_arg = cmd[cmd.index("--task") + 1]
                config_name = cmd[cmd.index("--config-name") + 1]
                task_id = Path(task_arg).stem
                results.mkdir(parents=True, exist_ok=True)
                _write_result(results, task_id, config_name, "log00001")
                return SimpleNamespace(returncode=0)
            return subprocess.run(cmd, *args, **kwargs)

        monkeypatch.setattr(launcher.subprocess, "run", fake_run)

        rc = launcher.main(
            [
                "--config",
                "reap-139b-nvfp4-new",
                "--task",
                "df_task_12",
                "--verify-baseline-clean",
                "skip",
                "--concurrency",
                "2",
            ]
        )

        assert rc == 0
        assert fake_client.terminate_calls == ["pod-fake-1"]
        # Concurrent path took the redirect branch
        assert captured_kwargs and captured_kwargs[0]["stdout"] is not None
        assert captured_kwargs[0]["stderr"] == subprocess.STDOUT
        # Log file exists and contains the fake subprocess output
        log_files = list(log_dir.glob("eval-*-df_task_12-*.log"))
        assert len(log_files) == 1, f"expected 1 log file, got {log_files}"
        assert "hello from fake subprocess" in log_files[0].read_text()

    def test_no_log_file_in_sequential_mode(self, monkeypatch, tmp_path):
        results, log_dir = self._setup_one_task(monkeypatch, tmp_path)
        fake_client = _patch_pod_infra(monkeypatch)

        captured_kwargs: list[dict] = []
        real_run = subprocess.run

        def fake_run(cmd, *args, **kwargs):
            if (
                isinstance(cmd, list)
                and cmd[:4] == ["uv", "run", "orchestrator", "eval"]
            ):
                captured_kwargs.append(kwargs)
                task_arg = cmd[cmd.index("--task") + 1]
                config_name = cmd[cmd.index("--config-name") + 1]
                task_id = Path(task_arg).stem
                results.mkdir(parents=True, exist_ok=True)
                _write_result(results, task_id, config_name, "seq00001")
                return SimpleNamespace(returncode=0)
            return real_run(cmd, *args, **kwargs)

        monkeypatch.setattr(launcher.subprocess, "run", fake_run)

        rc = launcher.main(
            [
                "--config",
                "reap-139b-nvfp4-new",
                "--task",
                "df_task_12",
                "--verify-baseline-clean",
                "skip",
                # default concurrency=1
            ]
        )
        assert rc == 0
        assert fake_client.terminate_calls == ["pod-fake-1"]
        # Sequential path: no redirect, so stdout/stderr kwargs should be None
        assert captured_kwargs and captured_kwargs[0].get("stdout") is None
        assert captured_kwargs[0].get("stderr") is None
        # No log files created
        assert not log_dir.exists() or not any(log_dir.glob("eval-*.log"))

    def test_log_file_path_in_summary(self, monkeypatch, tmp_path):
        results, log_dir = self._setup_one_task(monkeypatch, tmp_path)
        _patch_pod_infra(monkeypatch)

        captured: dict = {}
        real_print = launcher.print_summary_table

        def capture_print(summaries):
            captured["summaries"] = summaries
            return real_print(summaries)

        monkeypatch.setattr(launcher, "print_summary_table", capture_print)
        _patch_subprocess_run_success(monkeypatch, results)

        # Concurrent mode: log_path populated.
        launcher.main(
            [
                "--config",
                "reap-139b-nvfp4-new",
                "--task",
                "df_task_12",
                "--verify-baseline-clean",
                "skip",
                "--concurrency",
                "2",
            ]
        )
        assert len(captured["summaries"]) == 1
        assert captured["summaries"][0].log_path is not None
        assert captured["summaries"][0].log_path.parent == log_dir

        # Sequential mode: log_path is None.
        captured.clear()
        launcher.main(
            [
                "--config",
                "reap-139b-nvfp4-new",
                "--task",
                "df_task_12",
                "--verify-baseline-clean",
                "skip",
            ]
        )
        assert len(captured["summaries"]) == 1
        assert captured["summaries"][0].log_path is None

    def test_log_dir_created_if_missing(self, monkeypatch, tmp_path):
        log_dir = tmp_path / "deeply" / "nested" / "eval-logs"
        monkeypatch.setattr(launcher, "EVAL_LOG_DIR", log_dir)
        assert not log_dir.exists()

        path, fh = launcher._open_task_log("df_task_12", "cfg")
        try:
            assert log_dir.exists()
            assert path.parent == log_dir
            assert path.exists()
        finally:
            fh.close()


# ---------------------------------------------------------------------------
# Concurrent fanout (windowed ThreadPoolExecutor)
# ---------------------------------------------------------------------------

# Bind real time.sleep BEFORE _patch_pod_infra monkeypatches launcher.time.sleep,
# so the delay helper actually blocks instead of being a no-op.
_REAL_SLEEP = time.sleep


def _patch_subprocess_run_with_delay(monkeypatch, results_dir: Path, delay_s: float):
    """Like _patch_subprocess_run_success but adds a real sleep so worker
    threads overlap in time. Used to verify parallelism via wall-clock.
    """
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if (
            isinstance(cmd, list)
            and cmd[:4] == ["uv", "run", "orchestrator", "eval"]
        ):
            _REAL_SLEEP(delay_s)  # noqa — intentional, exercises threading
            task_arg = cmd[cmd.index("--task") + 1]
            config_name = cmd[cmd.index("--config-name") + 1]
            task_id = Path(task_arg).stem
            results_dir.mkdir(parents=True, exist_ok=True)
            run_id = uuid4().hex[:8]
            _write_result(results_dir, task_id, config_name, run_id)
            # Honor the redirect target so the per-task log file is non-empty.
            fh = kwargs.get("stdout")
            if fh is not None and hasattr(fh, "write"):
                fh.write(f"fake subprocess for {task_id}\n")
                fh.flush()
            return SimpleNamespace(returncode=0)
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)


def _write_n_task_specs(fake_tasks: Path, task_ids: list[str]) -> None:
    fake_tasks.mkdir(parents=True, exist_ok=True)
    for tid in task_ids:
        (fake_tasks / f"{tid}.json").write_text(
            json.dumps(
                {
                    "id": tid,
                    "pre_task_commit": "deadbeef",
                    "verify_commands": {"lint": "true", "typecheck": "true"},
                }
            )
        )


class TestConcurrentLoop:
    """Windowed-submission ThreadPoolExecutor path in main()."""

    def test_concurrent_runs_three_tasks_in_parallel(self, monkeypatch, tmp_path):
        """3 tasks at concurrency=3 with a 0.3s fake-subprocess delay should
        finish in ~0.3s wall-clock, not ~0.9s."""
        results = tmp_path / "results"
        monkeypatch.setattr(launcher, "RESULTS_DIR", results)
        monkeypatch.setattr(launcher, "EVAL_LOG_DIR", tmp_path / "logs")

        task_ids = ["df_task_12", "df_task_13", "df_task_18"]
        fake_tasks = tmp_path / "tasks"
        _write_n_task_specs(fake_tasks, task_ids)
        monkeypatch.setattr(launcher, "TASKS_DIR", fake_tasks)

        fake_client = _patch_pod_infra(monkeypatch)
        _patch_subprocess_run_with_delay(monkeypatch, results, delay_s=0.3)

        t0 = time.monotonic()
        rc = launcher.main(
            [
                "--config",
                "reap-139b-nvfp4-new",
                "--tasks",
                ",".join(task_ids),
                "--verify-baseline-clean",
                "skip",
                "--concurrency",
                "3",
            ]
        )
        wall_clock = time.monotonic() - t0

        assert rc == 0
        assert len(fake_client.create_calls) == 1
        assert fake_client.terminate_calls == ["pod-fake-1"]
        # Parallel: ~0.3s; serial: ~0.9s. Allow 0.6s for thread spinup overhead.
        assert wall_clock < 0.6, (
            f"expected ~0.3s parallel wall-clock, got {wall_clock:.2f}s "
            f"(serial would be ~0.9s)"
        )
        # All 3 result files written
        result_files = sorted(p.name for p in results.glob("*.json"))
        assert len(result_files) == 3

    def test_concurrent_with_higher_count_than_concurrency(
        self, monkeypatch, tmp_path
    ):
        """5 tasks, concurrency=2 → wall-clock ≈ ⌈5/2⌉ × 0.3s = 0.9s."""
        results = tmp_path / "results"
        monkeypatch.setattr(launcher, "RESULTS_DIR", results)
        monkeypatch.setattr(launcher, "EVAL_LOG_DIR", tmp_path / "logs")

        task_ids = [f"df_task_{i}" for i in (12, 13, 14, 15, 18)]
        fake_tasks = tmp_path / "tasks"
        _write_n_task_specs(fake_tasks, task_ids)
        monkeypatch.setattr(launcher, "TASKS_DIR", fake_tasks)

        fake_client = _patch_pod_infra(monkeypatch)
        _patch_subprocess_run_with_delay(monkeypatch, results, delay_s=0.3)

        t0 = time.monotonic()
        rc = launcher.main(
            [
                "--config",
                "reap-139b-nvfp4-new",
                "--tasks",
                ",".join(task_ids),
                "--verify-baseline-clean",
                "skip",
                "--concurrency",
                "2",
            ]
        )
        wall_clock = time.monotonic() - t0

        assert rc == 0
        assert len(fake_client.create_calls) == 1
        assert len(list(results.glob("*.json"))) == 5
        # 3 waves × 0.3s = 0.9s. Allow 1.5s for overhead.
        assert wall_clock < 1.5, f"wall-clock {wall_clock:.2f}s too high"
        # And > 0.6s — anything less means we somehow ran > 2 in parallel.
        assert wall_clock > 0.6, (
            f"wall-clock {wall_clock:.2f}s too low; concurrency cap "
            f"may not be enforced"
        )

    def test_concurrent_no_result_collision(self, monkeypatch, tmp_path):
        """4 result files written with the same mtime — each task must
        find ITS own result, not a sibling's."""
        import os

        results = tmp_path / "results"
        monkeypatch.setattr(launcher, "RESULTS_DIR", results)
        monkeypatch.setattr(launcher, "EVAL_LOG_DIR", tmp_path / "logs")

        task_ids = ["df_task_12", "df_task_13", "df_task_14", "df_task_15"]
        fake_tasks = tmp_path / "tasks"
        _write_n_task_specs(fake_tasks, task_ids)
        monkeypatch.setattr(launcher, "TASKS_DIR", fake_tasks)

        _patch_pod_infra(monkeypatch)

        # Force every result file to share an identical mtime, so the watermark
        # alone is insufficient and exclude_paths must do the work.
        forced_mtime = 1_700_000_000.0

        def fake_run(cmd, *args, **kwargs):
            if (
                isinstance(cmd, list)
                and cmd[:4] == ["uv", "run", "orchestrator", "eval"]
            ):
                _REAL_SLEEP(0.05)
                task_arg = cmd[cmd.index("--task") + 1]
                config_name = cmd[cmd.index("--config-name") + 1]
                task_id = Path(task_arg).stem
                results.mkdir(parents=True, exist_ok=True)
                p = _write_result(results, task_id, config_name, "collide00")
                os.utime(p, (forced_mtime, forced_mtime))
                return SimpleNamespace(returncode=0)
            return subprocess.run(cmd, *args, **kwargs)

        monkeypatch.setattr(launcher.subprocess, "run", fake_run)

        captured: dict = {}
        real_print = launcher.print_summary_table

        def capture_print(summaries):
            captured["summaries"] = summaries
            return real_print(summaries)

        monkeypatch.setattr(launcher, "print_summary_table", capture_print)

        rc = launcher.main(
            [
                "--config",
                "reap-139b-nvfp4-new",
                "--tasks",
                ",".join(task_ids),
                "--verify-baseline-clean",
                "skip",
                "--concurrency",
                "4",
            ]
        )
        assert rc == 0
        # Each summary's result_path must reference its own task_id.
        for s in captured["summaries"]:
            assert s.result_path is not None
            assert s.task_id in s.result_path.name, (
                f"task {s.task_id} got result_path {s.result_path} "
                f"belonging to a different task"
            )

    def test_concurrent_stop_on_first_failure_drains(self, monkeypatch, tmp_path):
        """Failing task stops the queue but in-flight tasks finish."""
        results = tmp_path / "results"
        monkeypatch.setattr(launcher, "RESULTS_DIR", results)
        monkeypatch.setattr(launcher, "EVAL_LOG_DIR", tmp_path / "logs")

        task_ids = ["df_task_10", "df_task_11", "df_task_12", "df_task_13", "df_task_14"]
        fake_tasks = tmp_path / "tasks"
        _write_n_task_specs(fake_tasks, task_ids)
        monkeypatch.setattr(launcher, "TASKS_DIR", fake_tasks)

        fake_client = _patch_pod_infra(monkeypatch)

        attempted: list[str] = []
        attempted_lock_marker: list[str] = []  # for assertion ordering

        def fake_run(cmd, *args, **kwargs):
            if (
                isinstance(cmd, list)
                and cmd[:4] == ["uv", "run", "orchestrator", "eval"]
            ):
                task_arg = cmd[cmd.index("--task") + 1]
                config_name = cmd[cmd.index("--config-name") + 1]
                task_id = Path(task_arg).stem
                attempted.append(task_id)
                attempted_lock_marker.append(task_id)
                _REAL_SLEEP(0.1)
                if task_id == "df_task_11":
                    # Simulate failure: write a "blocked" outcome.
                    results.mkdir(parents=True, exist_ok=True)
                    _write_result(
                        results,
                        task_id,
                        config_name,
                        "fail00001",
                        outcome="blocked",
                        metrics={"cost_usd": 5.0},
                    )
                    return SimpleNamespace(returncode=1)
                results.mkdir(parents=True, exist_ok=True)
                _write_result(results, task_id, config_name, f"r{len(attempted):08d}")
                return SimpleNamespace(returncode=0)
            return subprocess.run(cmd, *args, **kwargs)

        monkeypatch.setattr(launcher.subprocess, "run", fake_run)

        captured: dict = {}
        real_print = launcher.print_summary_table

        def capture_print(summaries):
            captured["summaries"] = summaries
            return real_print(summaries)

        monkeypatch.setattr(launcher, "print_summary_table", capture_print)

        rc = launcher.main(
            [
                "--config",
                "reap-139b-nvfp4-new",
                "--tasks",
                ",".join(task_ids),
                "--verify-baseline-clean",
                "skip",
                "--concurrency",
                "2",
                "--stop-on-first-failure",
            ]
        )

        assert rc == 1  # one task blocked → non-zero exit
        assert fake_client.terminate_calls == ["pod-fake-1"]
        # Tasks 10 and 11 were the initial wave; 11 failed. Task 12 may or
        # may not have been picked up depending on scheduling, but 13 and 14
        # must NOT have been submitted.
        assert "df_task_10" in attempted
        assert "df_task_11" in attempted
        assert "df_task_13" not in attempted
        assert "df_task_14" not in attempted
        # Summaries cover whatever was attempted.
        summary_ids = {s.task_id for s in captured["summaries"]}
        assert summary_ids == set(attempted)
        # The blocked task is in the summaries.
        blocked = [s for s in captured["summaries"] if s.task_id == "df_task_11"]
        assert blocked and blocked[0].status == "blocked"

    def test_concurrent_keyboard_interrupt_lets_in_flight_finish(
        self, monkeypatch, tmp_path
    ):
        """When the executor's wait() raises KeyboardInterrupt, the with-block's
        shutdown(wait=True) drains in-flight tasks before re-raising.

        Per plan risk #12, we test the *invariant* (pod terminated, KI
        propagated) by injecting KI through a wait() patch instead of via
        signal injection from another thread.
        """
        results = tmp_path / "results"
        monkeypatch.setattr(launcher, "RESULTS_DIR", results)
        monkeypatch.setattr(launcher, "EVAL_LOG_DIR", tmp_path / "logs")

        task_ids = ["df_task_12", "df_task_13", "df_task_18"]
        fake_tasks = tmp_path / "tasks"
        _write_n_task_specs(fake_tasks, task_ids)
        monkeypatch.setattr(launcher, "TASKS_DIR", fake_tasks)

        fake_client = _patch_pod_infra(monkeypatch)

        completed_tasks: list[str] = []

        def fake_run(cmd, *args, **kwargs):
            if (
                isinstance(cmd, list)
                and cmd[:4] == ["uv", "run", "orchestrator", "eval"]
            ):
                task_arg = cmd[cmd.index("--task") + 1]
                config_name = cmd[cmd.index("--config-name") + 1]
                task_id = Path(task_arg).stem
                _REAL_SLEEP(0.2)  # let executor.shutdown(wait=True) actually wait
                results.mkdir(parents=True, exist_ok=True)
                _write_result(results, task_id, config_name, "ki000001")
                completed_tasks.append(task_id)
                return SimpleNamespace(returncode=0)
            return subprocess.run(cmd, *args, **kwargs)

        monkeypatch.setattr(launcher.subprocess, "run", fake_run)

        # Make the FIRST call to launcher.wait raise KeyboardInterrupt.
        real_wait = launcher.wait
        wait_calls = {"n": 0}

        def interrupt_on_first_wait(*args, **kwargs):
            wait_calls["n"] += 1
            if wait_calls["n"] == 1:
                raise KeyboardInterrupt
            return real_wait(*args, **kwargs)

        monkeypatch.setattr(launcher, "wait", interrupt_on_first_wait)

        with pytest.raises(KeyboardInterrupt):
            launcher.main(
                [
                    "--config",
                    "reap-139b-nvfp4-new",
                    "--tasks",
                    ",".join(task_ids),
                    "--verify-baseline-clean",
                    "skip",
                    "--concurrency",
                    "3",
                ]
            )

        # Pod was still terminated by the outer finally
        assert fake_client.terminate_calls == ["pod-fake-1"]
        # All 3 in-flight tasks completed (executor.shutdown(wait=True) drained)
        assert sorted(completed_tasks) == sorted(task_ids)


# ---------------------------------------------------------------------------
# preflight_baseline (B3)
# ---------------------------------------------------------------------------


class TestPreflightBaseline:
    def test_clean_baseline_passes(self, tmp_repo):
        repo, sha = tmp_repo
        spec = {
            "id": "test_task",
            "pre_task_commit": sha,
            "verify_commands": {"lint": "true", "typecheck": "true"},
        }
        status = preflight_baseline(spec, repo_root=repo)
        assert status.is_clean
        assert status.head_matches
        assert status.lint_clean
        assert status.typecheck_clean

    def test_dirty_lint_fails(self, tmp_repo):
        repo, sha = tmp_repo
        spec = {
            "id": "test_task",
            "pre_task_commit": sha,
            "verify_commands": {"lint": "false", "typecheck": "true"},
        }
        status = preflight_baseline(spec, repo_root=repo)
        assert not status.is_clean
        assert not status.lint_clean
        assert status.typecheck_clean

    def test_dirty_typecheck_fails(self, tmp_repo):
        repo, sha = tmp_repo
        spec = {
            "id": "test_task",
            "pre_task_commit": sha,
            "verify_commands": {"lint": "true", "typecheck": "false"},
        }
        status = preflight_baseline(spec, repo_root=repo)
        assert not status.is_clean
        assert status.lint_clean
        assert not status.typecheck_clean

    def test_missing_verify_commands(self, tmp_repo):
        repo, sha = tmp_repo
        spec = {"id": "test_task", "pre_task_commit": sha}
        status = preflight_baseline(spec, repo_root=repo)
        assert not status.is_clean
        assert "missing verify_commands" in (status.error or "")

    def test_setup_commands_run_before_lint(self, tmp_repo):
        repo, sha = tmp_repo
        # Setup creates a marker file; lint checks for it.
        spec = {
            "id": "test_task",
            "pre_task_commit": sha,
            "setup_commands": ["touch SETUP_MARKER"],
            "verify_commands": {
                "lint": "test -f SETUP_MARKER",
                "typecheck": "true",
            },
        }
        status = preflight_baseline(spec, repo_root=repo)
        assert status.is_clean

    def test_cleans_up_worktree(self, tmp_repo):
        repo, sha = tmp_repo
        spec = {
            "id": "test_task",
            "pre_task_commit": sha,
            "verify_commands": {"lint": "true", "typecheck": "true"},
        }
        preflight_baseline(spec, repo_root=repo)

        # Either the directory was never created or only contains stale dirs
        # that are NOT registered as live worktrees.
        live = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=str(repo),
            capture_output=True,
            text=True,
        ).stdout
        assert "preflight-" not in live

    def test_strict_policy_aborts_before_pod(self, monkeypatch, tmp_path, tmp_repo):
        """Strict mode must refuse without ever creating a pod."""
        repo, sha = tmp_repo

        results = tmp_path / "results"
        monkeypatch.setattr(launcher, "RESULTS_DIR", results)

        fake_tasks = tmp_path / "tasks"
        fake_tasks.mkdir()
        (fake_tasks / "df_task_12.json").write_text(
            json.dumps(
                {
                    "id": "df_task_12",
                    "pre_task_commit": sha,
                    "verify_commands": {"lint": "false", "typecheck": "true"},
                }
            )
        )
        monkeypatch.setattr(launcher, "TASKS_DIR", fake_tasks)
        monkeypatch.setattr(launcher, "PROJECT_ROOT", repo)
        # main() fail-fasts on missing RUNPOD_API_KEY before preflight runs.
        # PROJECT_ROOT is monkeypatched to a tmp dir without .env, so inject
        # a fake key via env so we exercise the preflight logic, not the
        # credential-loading guard.
        monkeypatch.setenv("RUNPOD_API_KEY", "rpa_test_fake_key")

        fake_client = _FakeClient()
        monkeypatch.setattr(launcher, "RunPodClient", lambda config: fake_client)

        rc = launcher.main(
            [
                "--config",
                "reap-139b-nvfp4-new",
                "--task",
                "df_task_12",
                "--verify-baseline-clean",
                "strict",
            ]
        )
        assert rc == 2
        assert fake_client.create_calls == []
        assert fake_client.terminate_calls == []

    def test_warn_policy_proceeds(self, monkeypatch, tmp_path, tmp_repo):
        repo, sha = tmp_repo
        results = tmp_path / "results"
        monkeypatch.setattr(launcher, "RESULTS_DIR", results)

        fake_tasks = tmp_path / "tasks"
        fake_tasks.mkdir()
        (fake_tasks / "df_task_12.json").write_text(
            json.dumps(
                {
                    "id": "df_task_12",
                    "pre_task_commit": sha,
                    "verify_commands": {"lint": "false", "typecheck": "true"},
                }
            )
        )
        monkeypatch.setattr(launcher, "TASKS_DIR", fake_tasks)
        monkeypatch.setattr(launcher, "PROJECT_ROOT", repo)
        # See test_strict_policy_aborts_before_pod for why this env var is set.
        monkeypatch.setenv("RUNPOD_API_KEY", "rpa_test_fake_key")

        fake_client = _patch_pod_infra(monkeypatch)
        _patch_subprocess_run_success(monkeypatch, results)

        rc = launcher.main(
            [
                "--config",
                "reap-139b-nvfp4-new",
                "--task",
                "df_task_12",
                "--verify-baseline-clean",
                "warn",
            ]
        )
        assert rc == 0
        assert len(fake_client.create_calls) == 1
        assert fake_client.terminate_calls == ["pod-fake-1"]

    def test_skip_policy_does_not_run_lint(self, monkeypatch, tmp_path):
        results = tmp_path / "results"
        monkeypatch.setattr(launcher, "RESULTS_DIR", results)

        fake_tasks = tmp_path / "tasks"
        fake_tasks.mkdir()
        (fake_tasks / "df_task_12.json").write_text(
            json.dumps(
                {
                    "id": "df_task_12",
                    "pre_task_commit": "this-sha-does-not-exist",
                    "verify_commands": {
                        "lint": "this-command-would-fail",
                        "typecheck": "this-command-would-fail",
                    },
                }
            )
        )
        monkeypatch.setattr(launcher, "TASKS_DIR", fake_tasks)

        fake_client = _patch_pod_infra(monkeypatch)
        _patch_subprocess_run_success(monkeypatch, results)

        # If skip is honored, the bogus pre_task_commit and lint command should
        # never be touched, and the run proceeds normally.
        rc = launcher.main(
            [
                "--config",
                "reap-139b-nvfp4-new",
                "--task",
                "df_task_12",
                "--verify-baseline-clean",
                "skip",
            ]
        )
        assert rc == 0
        assert len(fake_client.create_calls) == 1


# ---------------------------------------------------------------------------
# Summary + exit
# ---------------------------------------------------------------------------


def _make_summary(status: str, **overrides) -> EvalSummary:
    defaults = dict(
        task_id="df_task_12",
        config_name="cfg",
        status=status,
        outcome=status,
        cost_usd=10.0,
        duration_s=120.0,
        tests_pass=True,
        lint_clean=True,
        typecheck_clean=True,
        run_id="abcd1234",
        result_path=Path("/tmp/r.json"),
    )
    defaults.update(overrides)
    return EvalSummary(**defaults)


class TestSummaryAndExit:
    def test_exit_zero_when_all_done(self):
        summaries = [
            _make_summary("done", task_id="df_task_12"),
            _make_summary("done", task_id="df_task_13"),
        ]
        assert compute_exit_code(summaries) == 0

    def test_exit_one_when_any_blocked(self):
        summaries = [
            _make_summary("done", task_id="df_task_12"),
            _make_summary("blocked", task_id="df_task_13"),
        ]
        assert compute_exit_code(summaries) == 1

    def test_exit_one_when_any_crashed(self):
        summaries = [
            _make_summary("done", task_id="df_task_12"),
            _make_summary("crashed", task_id="df_task_13", error="boom"),
        ]
        assert compute_exit_code(summaries) == 1

    def test_exit_one_when_empty(self):
        assert compute_exit_code([]) == 1

    def test_summary_table_renders(self, capsys):
        summaries = [
            _make_summary("done", task_id="df_task_12"),
            _make_summary(
                "blocked",
                task_id="df_task_13",
                tests_pass=False,
                lint_clean=False,
            ),
        ]
        print_summary_table(summaries)
        out = capsys.readouterr().out
        assert "df_task_12" in out
        assert "df_task_13" in out
        assert "done" in out
        assert "blocked" in out
        assert "RUN SUMMARY" in out
        # No log_path on either summary → both rows show "(stdout)"
        assert out.count("(stdout)") == 2

    def test_summary_table_shows_log_path(self, capsys):
        summaries = [
            _make_summary(
                "done",
                task_id="df_task_12",
                log_path=Path("/var/tmp/dark-factory-evals/eval-cfg-df_task_12-x.log"),
            ),
            _make_summary("done", task_id="df_task_13"),  # log_path=None → (stdout)
        ]
        print_summary_table(summaries)
        out = capsys.readouterr().out
        assert "/var/tmp/dark-factory-evals/eval-cfg-df_task_12-x.log" in out
        assert "(stdout)" in out


# ---------------------------------------------------------------------------
# tear_down_pod safety
# ---------------------------------------------------------------------------


class TestTearDownPod:
    def test_none_handle_is_safe(self):
        tear_down_pod(None)  # should not raise

    def test_partial_handle_no_pod(self):
        fake_tunnel = SimpleNamespace(
            terminate=lambda: None,
            wait=lambda timeout=None: 0,
            kill=lambda: None,
        )
        handle = PodHandle(
            pod=None,
            tunnel_proc=fake_tunnel,
            client=_FakeClient(),
            vllm_url="http://localhost:8100",
            local_port=8100,
            config_name="cfg",
        )
        tear_down_pod(handle)  # should not raise

    def test_partial_handle_no_tunnel(self):
        fake_client = _FakeClient()
        handle = PodHandle(
            pod=_FakePod(),
            tunnel_proc=None,
            client=fake_client,
            vllm_url="http://localhost:8100",
            local_port=8100,
            config_name="cfg",
        )
        tear_down_pod(handle)
        assert fake_client.terminate_calls == ["pod-fake-1"]


# ---------------------------------------------------------------------------
# Port and SSH tunnel — regression tests for the 2026-04-08 vLLM /v1/messages
# 404 bug (ssh -L port collision + no ExitOnForwardFailure).
# See docs/plan-vllm-404-bug-fix.md.
# ---------------------------------------------------------------------------


class TestPortAndTunnel:
    def test_pick_free_port_returns_usable_port(self):
        port = _pick_free_port()
        assert 1024 < port < 65536
        # Verify it's actually free right now.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))

    def test_pick_free_port_returns_distinct_ports_under_contention(self):
        # Hold one port so the OS must pick a different one next call.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
            held.bind(("127.0.0.1", 0))
            held_port = held.getsockname()[1]
            other = _pick_free_port()
        assert other != held_port

    def test_ssh_tunnel_argv_includes_exit_on_forward_failure(self):
        argv = _build_ssh_tunnel_argv(
            local_port=12345,
            ssh_host="1.2.3.4",
            ssh_port=22001,
            ssh_key="/tmp/key",
        )
        # The single critical assertion for the 2026-04-08 404 bug.
        assert "-o" in argv
        assert "ExitOnForwardFailure=yes" in argv
        # Sanity: port is threaded through.
        assert "127.0.0.1:12345:127.0.0.1:8000" in argv
        # Sanity: host + port + key flow through unchanged.
        assert "root@1.2.3.4" in argv
        assert "22001" in argv
        assert "/tmp/key" in argv

    def test_parse_args_default_port_is_none(self):
        # --port not passed → argparse default is None (resolution happens
        # in main() via _pick_free_port).
        args = parse_args(
            ["--config", "reap-139b-nvfp4-new", "--task", "df_task_12"]
        )
        assert args.port is None

    def test_parse_args_explicit_port_honored(self):
        args = parse_args(
            [
                "--config",
                "reap-139b-nvfp4-new",
                "--task",
                "df_task_12",
                "--port",
                "8299",
            ]
        )
        assert args.port == 8299


class TestWaitForVllmModelCheck:
    """Layer 3: /v1/models membership guards against sibling-tunnel bleed."""

    def _fake_urlopen(
        self,
        responses: "dict[str, tuple[int, bytes] | Exception]",
    ):
        """Return a urllib.request.urlopen stub that dispatches on URL.

        ``responses`` maps URL → either a ``(status, body)`` tuple or an
        Exception instance to raise.
        """

        class _Resp:
            def __init__(self, status: int, body: bytes):
                self.status = status
                self._body = body

            def read(self):
                return self._body

        def _stub(url, timeout=None):
            r = responses[url]
            if isinstance(r, Exception):
                raise r
            status, body = r
            return _Resp(status, body)

        return _stub

    def test_passes_when_model_served(self, monkeypatch):
        import urllib.request

        port = 19999
        expected = "nvidia/MiniMax-M2.5-NVFP4"
        body = json.dumps(
            {"data": [{"id": expected}, {"id": "other/model"}]}
        ).encode()
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            self._fake_urlopen(
                {
                    f"http://127.0.0.1:{port}/health": (200, b"ok"),
                    f"http://127.0.0.1:{port}/v1/models": (200, body),
                }
            ),
        )
        assert wait_for_vllm(port, expected_model=expected, timeout=1)

    def test_fails_when_sibling_serves_different_model(self, monkeypatch):
        import urllib.request

        port = 19999
        expected = "lukealonso/MiniMax-M2.5-REAP-139B-A10B-NVFP4"
        # Sibling tunnel has a different model loaded.
        body = json.dumps(
            {"data": [{"id": "Qwen/Qwen3-Coder-Next-FP8"}]}
        ).encode()
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            self._fake_urlopen(
                {
                    f"http://127.0.0.1:{port}/health": (200, b"ok"),
                    f"http://127.0.0.1:{port}/v1/models": (200, body),
                }
            ),
        )
        # Speed this up: patch time.sleep so the 1-second timeout trips fast.
        monkeypatch.setattr(launcher.time, "sleep", lambda _s: None)
        assert not wait_for_vllm(port, expected_model=expected, timeout=1)
