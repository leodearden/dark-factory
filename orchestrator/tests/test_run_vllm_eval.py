"""Tests for the multi-task vLLM eval launcher (scripts/run_vllm_eval.py)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

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


_rpt_config.RunPodConfig = _StubRunPodConfig
_rpt_compute.RunPodClient = _StubRunPodClient
_rpt_compute.PodStatus = _StubPodStatus

sys.modules.setdefault("runpod_toolkit", _rpt)
sys.modules.setdefault("runpod_toolkit.config", _rpt_config)
sys.modules.setdefault("runpod_toolkit.compute", _rpt_compute)

# Make scripts/run_vllm_eval.py importable. The launcher modifies sys.path
# itself at import time to find runpod_toolkit + orchestrator.
_SCRIPTS = Path(__file__).parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_vllm_eval as launcher  # noqa: E402
from run_vllm_eval import (  # noqa: E402
    BaselineStatus,
    EvalSummary,
    PodBringupFailed,
    PodHandle,
    TaskSpecNotFound,
    compute_exit_code,
    find_new_result_file,
    load_task_spec,
    parse_args,
    parse_result_file,
    preflight_baseline,
    print_summary_table,
    resolve_task_ids,
    tear_down_pod,
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
        for name in ("df_task_12.json", "df_task_18.json", "reify_task_27.json"):
            (fake_tasks / name).write_text("{}")
        monkeypatch.setattr(launcher, "TASKS_DIR", fake_tasks)

        args = _make_args(all_tasks=True)
        result = resolve_task_ids(args)
        # reify_task_27 is excluded; df_task_*.json only.
        assert result == ["df_task_12", "df_task_18"]

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
    monkeypatch.setattr(launcher, "wait_for_vllm", lambda port, timeout=900: True)
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
    def test_multi_task_terminates_pod_once(self, monkeypatch, tmp_path):
        results = tmp_path / "results"
        monkeypatch.setattr(launcher, "RESULTS_DIR", results)

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
        monkeypatch.setattr(launcher, "wait_for_vllm", lambda port, timeout=900: False)
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

        wt_dir = repo / ".eval-worktrees" / "test_task"
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
