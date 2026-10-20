"""Tests for the watchdog's module-removal waiver tier (task 7198).

A watched commit that deletes or renames a Python module arms a
ModuleNotFoundError in every running orchestrator that imports it lazily, so
scripts/orchestrator-watchdog.py lets such a removal bypass the fleet-deploy
clock gates once a short grace has passed. OPERATIONS.md section 8 ("Fleet
redeploy & watchdog") describes the behaviour.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time
from collections.abc import Callable
from types import ModuleType

import pytest

REPO_ROOT = pathlib.Path(__file__).parents[2]

# APPEND, never insert(0, ...): the repo root must stay LAST on sys.path so the
# workspace members resolve to their src/ packages (mirrors
# test_orchestrator_watchdog.py).
if str(REPO_ROOT.resolve()) not in sys.path:
    sys.path.append(str(REPO_ROOT.resolve()))

from df_pytest_isolation import synthetic_unit  # noqa: E402

# Saved before any test wraps subprocess.run, so the fixture's own git calls
# and the pass-through wrappers below always reach the real binary.
_REAL_RUN = subprocess.run

T0 = 1_900_000_000
DEPLOY = T0 + 100
CHANGE = T0 + 3600

GONE = "orchestrator/src/orchestrator/gone.py"
MOVED = "orchestrator/src/orchestrator/moved.py"
KEPT = "orchestrator/src/orchestrator/kept.py"
NEW = "orchestrator/src/orchestrator/new.py"
DATA = "orchestrator/src/orchestrator/data.json"
HELPER = "shared/src/shared/helper.py"
TEST_GONE = "orchestrator/tests/test_gone.py"
TOOL = "scripts/tool.py"


def _load_watchdog() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "orchestrator_watchdog", REPO_ROOT / "scripts" / "orchestrator-watchdog.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_module_removal_grace_defaults_to_600(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ORCH_MODULE_REMOVAL_GRACE_SECS", raising=False)
    assert _load_watchdog().MODULE_REMOVAL_GRACE_SECS == 600


def test_module_removal_grace_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORCH_MODULE_REMOVAL_GRACE_SECS", "42")
    assert _load_watchdog().MODULE_REMOVAL_GRACE_SECS == 42


def test_malformed_module_removal_grace_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORCH_MODULE_REMOVAL_GRACE_SECS", "ten-minutes")
    assert _load_watchdog().MODULE_REMOVAL_GRACE_SECS == 600


def test_module_removal_watched_paths_extend_watched_paths_with_shared() -> None:
    wdog = _load_watchdog()
    watched = wdog.MODULE_REMOVAL_WATCHED_PATHS
    assert set(wdog.WATCHED_PATHS) <= set(watched)
    assert "shared/src/" in watched
    missing = [path for path in watched if not (REPO_ROOT / path).exists()]
    assert not missing, f"MODULE_REMOVAL_WATCHED_PATHS entries naming no path: {missing}"


@pytest.fixture
def wdog(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.delenv("ORCH_MODULE_REMOVAL_GRACE_SECS", raising=False)
    return _load_watchdog()


def _git(repo: pathlib.Path, epoch: int, *args: str) -> str:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = f"@{epoch} +0000"
    result = _REAL_RUN(
        [
            "git",
            "-c", "user.name=t",
            "-c", "user.email=t@t",
            "-c", "commit.gpgsign=false",
            "-c", "core.hooksPath=/dev/null",
            *args,
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _build_repo(tmp_path: pathlib.Path) -> pathlib.Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, T0, "init", "-q", "-b", "main")
    for rel in (GONE, KEPT, DATA, HELPER, TEST_GONE, TOOL):
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {rel}\n")
    _git(repo, T0, "add", "-A")
    _git(repo, T0, "commit", "-q", "-m", "initial")
    return repo


def _commit(repo: pathlib.Path, epoch: int, message: str) -> None:
    _git(repo, epoch, "commit", "-q", "-m", message)


def _edit(repo: pathlib.Path, epoch: int, rel: str = KEPT) -> None:
    with (repo / rel).open("a") as handle:
        handle.write(f"# edited at {epoch}\n")
    _git(repo, epoch, "add", rel)
    _commit(repo, epoch, f"edit {rel}")


def _delete(repo: pathlib.Path, epoch: int, rel: str = GONE) -> None:
    _git(repo, epoch, "rm", "-q", rel)
    _commit(repo, epoch, f"delete {rel}")


def _rename(repo: pathlib.Path, epoch: int) -> None:
    _git(repo, epoch, "mv", GONE, MOVED)
    _commit(repo, epoch, "rename gone.py")


def _add(repo: pathlib.Path, epoch: int) -> None:
    (repo / NEW).write_text("# new\n")
    _git(repo, epoch, "add", NEW)
    _commit(repo, epoch, "add new.py")


def _delete_outside_watched_with_edit(repo: pathlib.Path, epoch: int) -> None:
    _git(repo, epoch, "rm", "-q", TEST_GONE)
    with (repo / KEPT).open("a") as handle:
        handle.write("# edited\n")
    _git(repo, epoch, "add", KEPT)
    _commit(repo, epoch, "delete a test, edit kept.py")


Scenario = Callable[[pathlib.Path, int], None]

TRIGGERS = [
    pytest.param(_delete, (GONE,), id="delete"),
    pytest.param(_rename, (GONE,), id="rename"),
    pytest.param(lambda repo, epoch: _delete(repo, epoch, HELPER), (HELPER,), id="shared-delete"),
]

NON_TRIGGERS = [
    pytest.param(_edit, id="edit"),
    pytest.param(_add, id="add"),
    pytest.param(lambda repo, epoch: _delete(repo, epoch, DATA), id="non-py-delete"),
    pytest.param(_delete_outside_watched_with_edit, id="outside-watched-delete"),
]


def _write_clock(path: pathlib.Path, ts: int) -> None:
    iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(ts))
    path.write_text(json.dumps({"ts": ts, "iso": iso}))


def _set_now(wdog: ModuleType, monkeypatch: pytest.MonkeyPatch, now: float) -> None:
    monkeypatch.setattr(wdog.time, "time", lambda: now)


def _point_at(
    wdog: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    repo: pathlib.Path,
    now: float,
) -> list[str]:
    """Aim every file and clock the waiver reads at test-owned state; return the log."""
    clock = tmp_path / "last_redeploy_orchestrator.json"
    _write_clock(clock, DEPLOY)
    monkeypatch.setattr(wdog, "REPO_DIR", str(repo))
    monkeypatch.setattr(wdog, "FLEET_DEPLOY_CLOCK_PATH", str(clock))
    monkeypatch.setattr(wdog, "FLEET_LEASE_PATH", str(tmp_path / "fleet_redeploy_lease.json"))
    monkeypatch.setattr(wdog, "ORCH_RESTART_MIN_INTERVAL_SECS", 259200)
    _set_now(wdog, monkeypatch, now)
    lines: list[str] = []
    monkeypatch.setattr(wdog, "log", lines.append)
    return lines


@pytest.mark.parametrize(("scenario", "expected"), TRIGGERS)
def test_removed_watched_module_is_due_after_grace(
    wdog: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    scenario: Scenario,
    expected: tuple[str, ...],
) -> None:
    repo = _build_repo(tmp_path)
    scenario(repo, CHANGE)
    _point_at(wdog, monkeypatch, tmp_path, repo, CHANGE + wdog.MODULE_REMOVAL_GRACE_SECS)
    assert wdog._due_module_removals() == expected


@pytest.mark.parametrize("scenario", NON_TRIGGERS)
def test_change_that_removes_no_watched_module_is_never_due(
    wdog: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    scenario: Scenario,
) -> None:
    repo = _build_repo(tmp_path)
    scenario(repo, CHANGE)
    _point_at(wdog, monkeypatch, tmp_path, repo, CHANGE + wdog.MODULE_REMOVAL_GRACE_SECS)
    assert wdog._due_module_removals() == ()


@pytest.mark.parametrize(("offset", "expected"), [(-1, ()), (0, (GONE,))])
def test_removal_becomes_due_exactly_when_grace_elapses(
    wdog: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    offset: int,
    expected: tuple[str, ...],
) -> None:
    repo = _build_repo(tmp_path)
    _delete(repo, CHANGE)
    now = CHANGE + wdog.MODULE_REMOVAL_GRACE_SECS + offset
    _point_at(wdog, monkeypatch, tmp_path, repo, now)
    assert wdog._due_module_removals() == expected


@pytest.mark.parametrize(("offset", "expected"), [(-1, ()), (0, (GONE,))])
def test_side_branch_removal_is_dated_by_its_merge_onto_main(
    wdog: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    offset: int,
    expected: tuple[str, ...],
) -> None:
    repo = _build_repo(tmp_path)
    _git(repo, CHANGE - 10_000, "checkout", "-q", "-b", "side")
    _delete(repo, CHANGE - 10_000)
    _git(repo, CHANGE, "checkout", "-q", "main")
    _git(repo, CHANGE, "merge", "-q", "--no-ff", "-m", "merge side", "side")
    now = CHANGE + wdog.MODULE_REMOVAL_GRACE_SECS + offset
    _point_at(wdog, monkeypatch, tmp_path, repo, now)
    assert wdog._due_module_removals() == expected


def test_later_removal_does_not_defer_the_first(
    wdog: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    grace = wdog.MODULE_REMOVAL_GRACE_SECS
    repo = _build_repo(tmp_path)
    _delete(repo, CHANGE)
    _delete(repo, CHANGE + grace - 30, HELPER)
    _point_at(wdog, monkeypatch, tmp_path, repo, CHANGE + grace)
    assert wdog._due_module_removals() == (GONE, HELPER)


def test_removal_before_the_last_fleet_deploy_is_not_due(
    wdog: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    repo = _build_repo(tmp_path)
    _delete(repo, T0 + 50)
    _point_at(wdog, monkeypatch, tmp_path, repo, CHANGE + wdog.MODULE_REMOVAL_GRACE_SECS)
    assert wdog._due_module_removals() == ()


Breaker = Callable[[ModuleType, pytest.MonkeyPatch, pathlib.Path], None]


def _point_repo_dir_at_a_non_repo(
    wdog: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()
    monkeypatch.setattr(wdog, "REPO_DIR", str(not_a_repo))


def _make_git_raise(exc: Exception) -> Breaker:
    def breaker(
        wdog: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
    ) -> None:
        def run(argv, *args, **kwargs):
            if argv[0] == "git":
                raise exc
            return _REAL_RUN(argv, *args, **kwargs)

        monkeypatch.setattr(wdog.subprocess, "run", run)

    return breaker


def _date_the_deploy_before_all_history(
    wdog: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    _write_clock(pathlib.Path(wdog.FLEET_DEPLOY_CLOCK_PATH), T0 - 1000)


@pytest.mark.parametrize(
    "breaker",
    [
        pytest.param(_point_repo_dir_at_a_non_repo, id="not-a-repo"),
        pytest.param(_make_git_raise(FileNotFoundError("git")), id="git-missing"),
        pytest.param(
            _make_git_raise(subprocess.TimeoutExpired(["git"], 5)), id="git-timeout"
        ),
        pytest.param(_date_the_deploy_before_all_history, id="no-mainline-base"),
    ],
)
def test_failed_probe_waives_nothing_and_warns(
    wdog: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    breaker: Breaker,
) -> None:
    repo = _build_repo(tmp_path)
    _delete(repo, CHANGE)
    lines = _point_at(wdog, monkeypatch, tmp_path, repo, CHANGE + wdog.MODULE_REMOVAL_GRACE_SECS)
    breaker(wdog, monkeypatch, tmp_path)
    assert wdog._due_module_removals() == ()
    assert any(line.startswith("WARNING:") for line in lines), lines


def test_absent_deploy_clock_waives_nothing_without_running_git(
    wdog: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    repo = _build_repo(tmp_path)
    _delete(repo, CHANGE)
    _point_at(wdog, monkeypatch, tmp_path, repo, CHANGE + wdog.MODULE_REMOVAL_GRACE_SECS)
    monkeypatch.setattr(wdog, "FLEET_DEPLOY_CLOCK_PATH", str(tmp_path / "absent-clock.json"))
    calls: list[list[str]] = []

    def recording_run(argv, *args, **kwargs):
        calls.append(list(argv))
        return _REAL_RUN(argv, *args, **kwargs)

    monkeypatch.setattr(wdog.subprocess, "run", recording_run)
    assert wdog._due_module_removals() == ()
    assert calls == []


def _wire_pass(wdog: ModuleType, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Stub only the systemctl-level unit probes; return the recorded systemd-run calls.

    git passes through to the real binary against the throwaway repo, so the
    clock and lease gates, the watched-commit probe and _delegate_fleet_restart
    all run for real.
    """
    monkeypatch.setattr(wdog, "_enumerate_running_units", lambda: [synthetic_unit("stale")])
    monkeypatch.setattr(wdog, "is_unit_enabled", lambda _unit: True)
    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", lambda _unit: 300.0)
    monkeypatch.setattr(wdog, "_unit_start_epoch", lambda _unit: DEPLOY + 10)
    delegations: list[list[str]] = []

    def run(argv, *args, **kwargs):
        if argv[0] == "git":
            return _REAL_RUN(argv, *args, **kwargs)
        if argv[0] == "systemd-run":
            delegations.append(list(argv))
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        pytest.fail(f"unexpected subprocess in staleness_pass: {argv}")

    monkeypatch.setattr(wdog.subprocess, "run", run)
    return delegations


def _fleet_redeploy_argv(wdog: ModuleType) -> list[str]:
    return [
        "systemd-run",
        "--user",
        "--collect",
        "--no-block",
        f"--unit={wdog.FLEET_STALENESS_REDEPLOY_UNIT}",
        os.path.join(wdog.REPO_DIR, "scripts", "restart-all-orchestrators.sh"),
        "--drain",
    ]


def _next_log_bucket(wdog: ModuleType, after: int) -> int:
    """The first instant at or after *after* whose throttled skip lines are emitted."""
    return after + (-after) % wdog.SKIP_LOG_INTERVAL_SECS


@pytest.mark.parametrize(("scenario", "expected"), TRIGGERS)
def test_due_removal_delegates_one_fleet_redeploy_inside_the_min_interval(
    wdog: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    scenario: Scenario,
    expected: tuple[str, ...],
) -> None:
    repo = _build_repo(tmp_path)
    scenario(repo, CHANGE)
    lines = _point_at(wdog, monkeypatch, tmp_path, repo, CHANGE + wdog.MODULE_REMOVAL_GRACE_SECS)
    delegations = _wire_pass(wdog, monkeypatch)

    wdog.staleness_pass()

    assert delegations == [_fleet_redeploy_argv(wdog)]
    assert any("module-removal" in line and expected[0] in line for line in lines), lines


def test_removal_inside_its_grace_delegates_nothing(
    wdog: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    repo = _build_repo(tmp_path)
    _delete(repo, CHANGE)
    _point_at(wdog, monkeypatch, tmp_path, repo, CHANGE + wdog.MODULE_REMOVAL_GRACE_SECS - 1)
    delegations = _wire_pass(wdog, monkeypatch)

    wdog.staleness_pass()

    assert delegations == []


@pytest.mark.parametrize("scenario", NON_TRIGGERS)
def test_stale_fleet_without_a_removal_still_waits_out_the_min_interval(
    wdog: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    scenario: Scenario,
) -> None:
    repo = _build_repo(tmp_path)
    scenario(repo, CHANGE)
    _point_at(wdog, monkeypatch, tmp_path, repo, CHANGE + wdog.MODULE_REMOVAL_GRACE_SECS)
    delegations = _wire_pass(wdog, monkeypatch)

    wdog.staleness_pass()

    assert delegations == []


def test_live_lease_still_blocks_a_due_removal(
    wdog: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    repo = _build_repo(tmp_path)
    _delete(repo, CHANGE)
    now = _next_log_bucket(wdog, CHANGE + wdog.MODULE_REMOVAL_GRACE_SECS)
    lines = _point_at(wdog, monkeypatch, tmp_path, repo, now)
    lease = {"pid": os.getpid(), "started_ts": now - 60, "current_unit": synthetic_unit("held")}
    pathlib.Path(wdog.FLEET_LEASE_PATH).write_text(json.dumps(lease))
    delegations = _wire_pass(wdog, monkeypatch)

    wdog.staleness_pass()

    assert delegations == []
    assert any("lease" in line for line in lines), lines


def test_due_removal_waives_the_commit_grace(
    wdog: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    now = CHANGE + wdog.MODULE_REMOVAL_GRACE_SECS
    repo = _build_repo(tmp_path)
    _delete(repo, CHANGE)
    _edit(repo, now - 60)
    _point_at(wdog, monkeypatch, tmp_path, repo, now)
    delegations = _wire_pass(wdog, monkeypatch)

    wdog.staleness_pass()

    assert delegations == [_fleet_redeploy_argv(wdog)]


def test_waiver_stops_once_the_redeploy_stamps_the_clock(
    wdog: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    now = CHANGE + wdog.MODULE_REMOVAL_GRACE_SECS
    repo = _build_repo(tmp_path)
    _delete(repo, CHANGE)
    _point_at(wdog, monkeypatch, tmp_path, repo, now)
    delegations = _wire_pass(wdog, monkeypatch)
    wdog.staleness_pass()
    assert len(delegations) == 1

    _write_clock(pathlib.Path(wdog.FLEET_DEPLOY_CLOCK_PATH), now)
    wdog.staleness_pass()

    assert len(delegations) == 1
    assert wdog._due_module_removals() == ()


def test_clock_stamped_mid_pass_cancels_the_waiver_delegation(
    wdog: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    now = CHANGE + wdog.MODULE_REMOVAL_GRACE_SECS
    repo = _build_repo(tmp_path)
    _delete(repo, CHANGE)
    lines = _point_at(wdog, monkeypatch, tmp_path, repo, now)
    reads = iter([DEPLOY, DEPLOY])
    monkeypatch.setattr(wdog, "_read_last_fleet_deploy_epoch", lambda: next(reads, now))
    delegations = _wire_pass(wdog, monkeypatch)

    wdog.staleness_pass()

    assert delegations == []
    assert any("stamped while this pass was probing" in line for line in lines), lines


def test_failed_git_probe_inside_the_min_interval_delegates_nothing(
    wdog: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    repo = _build_repo(tmp_path)
    _delete(repo, CHANGE)
    lines = _point_at(wdog, monkeypatch, tmp_path, repo, CHANGE + wdog.MODULE_REMOVAL_GRACE_SECS)
    delegations = _wire_pass(wdog, monkeypatch)
    _point_repo_dir_at_a_non_repo(wdog, monkeypatch, tmp_path)

    wdog.staleness_pass()

    assert delegations == []
    assert any(line.startswith("WARNING:") for line in lines), lines
