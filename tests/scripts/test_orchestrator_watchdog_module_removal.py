"""Tests for the watchdog's module-removal waiver tier (task 7198).

A watched commit that deletes or renames a Python module arms a
ModuleNotFoundError in every running orchestrator that started before it and
imports it lazily, so scripts/orchestrator-watchdog.py lets such a removal
bypass the fleet-deploy clock gates once a short grace has passed. The git
probe itself lives in scripts/_watched_module_removals.py. OPERATIONS.md
section 8 ("Fleet redeploy & watchdog") describes the behaviour.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import time
from collections.abc import Callable
from types import ModuleType

import pytest
from _watched_module_removals import ModuleRemovals, removals_since

from df_pytest_isolation import synthetic_unit

REPO_ROOT = pathlib.Path(__file__).parents[2]

# Saved before any test wraps subprocess.run, so the fixture's own git calls
# and the pass-through wrappers below always reach the real binary.
_REAL_RUN = subprocess.run

T0 = 1_900_000_000
DEPLOY = T0 + 100
STARTED = DEPLOY + 10
CHANGE = T0 + 3600
STALE = synthetic_unit("stale")

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


def _turn_into_package(repo: pathlib.Path, epoch: int) -> None:
    _git(repo, epoch, "rm", "-q", GONE)
    package_init = repo / GONE.removesuffix(".py") / "__init__.py"
    package_init.parent.mkdir()
    package_init.write_text("# now a package\n")
    _git(repo, epoch, "add", "-A")
    _commit(repo, epoch, "turn gone.py into a package")


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


class _RecordingLog:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, msg: str) -> None:
        self.warnings.append(msg)


def _probe(
    wdog: ModuleType, repo_dir: pathlib.Path, since: float = STARTED, log: _RecordingLog | None = None
) -> ModuleRemovals | None:
    return removals_since(
        str(repo_dir), since, wdog.MODULE_REMOVAL_WATCHED_PATHS, log or _RecordingLog()
    )


@pytest.mark.parametrize(("scenario", "expected"), TRIGGERS)
def test_removed_watched_module_is_reported_with_its_landing_time(
    wdog: ModuleType, tmp_path: pathlib.Path, scenario: Scenario, expected: tuple[str, ...]
) -> None:
    repo = _build_repo(tmp_path)
    scenario(repo, CHANGE)
    assert _probe(wdog, repo) == ModuleRemovals(expected, CHANGE)


@pytest.mark.parametrize("scenario", NON_TRIGGERS)
def test_change_that_removes_no_watched_module_reports_nothing(
    wdog: ModuleType, tmp_path: pathlib.Path, scenario: Scenario
) -> None:
    repo = _build_repo(tmp_path)
    scenario(repo, CHANGE)
    assert _probe(wdog, repo) is None


def test_module_turned_into_a_package_reads_as_a_removal(
    wdog: ModuleType, tmp_path: pathlib.Path
) -> None:
    repo = _build_repo(tmp_path)
    _turn_into_package(repo, CHANGE)
    assert _probe(wdog, repo) == ModuleRemovals((GONE,), CHANGE)


def test_side_branch_removal_is_dated_by_its_merge_onto_main(
    wdog: ModuleType, tmp_path: pathlib.Path
) -> None:
    repo = _build_repo(tmp_path)
    _git(repo, CHANGE - 1000, "checkout", "-q", "-b", "side")
    _delete(repo, CHANGE - 1000)
    _git(repo, CHANGE, "checkout", "-q", "main")
    _git(repo, CHANGE, "merge", "-q", "--no-ff", "-m", "merge side", "side")
    assert _probe(wdog, repo) == ModuleRemovals((GONE,), CHANGE)


def test_removal_before_the_baseline_is_not_reported(
    wdog: ModuleType, tmp_path: pathlib.Path
) -> None:
    repo = _build_repo(tmp_path)
    _delete(repo, T0 + 50)
    assert _probe(wdog, repo) is None


ProbeBreaker = Callable[[pytest.MonkeyPatch, pathlib.Path, pathlib.Path], tuple[pathlib.Path, float]]


def _a_non_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, repo: pathlib.Path
) -> tuple[pathlib.Path, float]:
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()
    return not_a_repo, STARTED


def _git_raising(exc: Exception) -> ProbeBreaker:
    def breaker(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, repo: pathlib.Path
    ) -> tuple[pathlib.Path, float]:
        def run(argv, *args, **kwargs):
            if argv[0] == "git":
                raise exc
            return _REAL_RUN(argv, *args, **kwargs)

        monkeypatch.setattr(subprocess, "run", run)
        return repo, STARTED

    return breaker


def _a_baseline_before_all_history(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, repo: pathlib.Path
) -> tuple[pathlib.Path, float]:
    return repo, T0 - 1000


@pytest.mark.parametrize(
    "breaker",
    [
        pytest.param(_a_non_repo, id="not-a-repo"),
        pytest.param(_git_raising(FileNotFoundError("git")), id="git-missing"),
        pytest.param(_git_raising(subprocess.TimeoutExpired(["git"], 5)), id="git-timeout"),
        pytest.param(_a_baseline_before_all_history, id="no-mainline-base"),
    ],
)
def test_failed_probe_reports_nothing_and_warns(
    wdog: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    breaker: ProbeBreaker,
) -> None:
    repo = _build_repo(tmp_path)
    _delete(repo, CHANGE)
    repo_dir, since = breaker(monkeypatch, tmp_path, repo)
    log = _RecordingLog()
    assert _probe(wdog, repo_dir, since, log) is None
    assert log.warnings


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


def _stub_units(
    wdog: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    starts: dict[str, int],
    enabled: frozenset[str] | None = None,
) -> None:
    """Stub the systemctl-level unit probes: *starts* maps each running unit to its start.

    Read at call time, so a test restarts a unit by assigning to *starts*.
    Every unit is enabled unless *enabled* names the ones that are.
    """
    monkeypatch.setattr(wdog, "_enumerate_running_units", lambda: list(starts))
    monkeypatch.setattr(
        wdog, "is_unit_enabled", lambda unit: enabled is None or unit in enabled
    )
    monkeypatch.setattr(wdog, "_unit_start_elapsed_secs", lambda _unit: 300.0)
    monkeypatch.setattr(wdog, "_unit_start_epoch", lambda unit: starts[unit])


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
    _point_at(wdog, monkeypatch, tmp_path, repo, CHANGE + wdog.MODULE_REMOVAL_GRACE_SECS + offset)
    _stub_units(wdog, monkeypatch, {STALE: STARTED})
    assert wdog._due_module_removals() == expected


def test_later_removal_does_not_defer_the_first(
    wdog: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    grace = wdog.MODULE_REMOVAL_GRACE_SECS
    repo = _build_repo(tmp_path)
    _delete(repo, CHANGE)
    _delete(repo, CHANGE + grace - 30, HELPER)
    _point_at(wdog, monkeypatch, tmp_path, repo, CHANGE + grace)
    _stub_units(wdog, monkeypatch, {STALE: STARTED})
    assert wdog._due_module_removals() == (GONE, HELPER)


@pytest.mark.parametrize(
    ("old_unit_enabled", "expected"), [(True, (GONE,)), (False, ())], ids=["enabled", "disabled"]
)
def test_removal_is_due_only_when_an_enabled_unit_predates_it(
    wdog: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    old_unit_enabled: bool,
    expected: tuple[str, ...],
) -> None:
    old, young = synthetic_unit("old"), synthetic_unit("young")
    repo = _build_repo(tmp_path)
    _delete(repo, CHANGE)
    _point_at(wdog, monkeypatch, tmp_path, repo, CHANGE + wdog.MODULE_REMOVAL_GRACE_SECS)
    enabled = frozenset({old, young} if old_unit_enabled else {young})
    _stub_units(wdog, monkeypatch, {old: STARTED, young: CHANGE + 10}, enabled)
    assert wdog._due_module_removals() == expected


def test_no_running_unit_waives_nothing_without_running_git(
    wdog: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    repo = _build_repo(tmp_path)
    _delete(repo, CHANGE)
    _point_at(wdog, monkeypatch, tmp_path, repo, CHANGE + wdog.MODULE_REMOVAL_GRACE_SECS)
    _stub_units(wdog, monkeypatch, {})
    calls: list[list[str]] = []

    def recording_run(argv, *args, **kwargs):
        calls.append(list(argv))
        return _REAL_RUN(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", recording_run)
    assert wdog._due_module_removals() == ()
    assert calls == []


def _wire_pass(
    wdog: ModuleType, monkeypatch: pytest.MonkeyPatch, starts: dict[str, int] | None = None
) -> list[list[str]]:
    """Stub only the systemctl-level unit probes; return the recorded systemd-run calls.

    git passes through to the real binary against the throwaway repo, so the
    clock and lease gates, the watched-commit probe and _delegate_fleet_restart
    all run for real.
    """
    _stub_units(wdog, monkeypatch, {STALE: STARTED} if starts is None else starts)
    delegations: list[list[str]] = []

    def run(argv, *args, **kwargs):
        if argv[0] == "git":
            return _REAL_RUN(argv, *args, **kwargs)
        if argv[0] == "systemd-run":
            delegations.append(list(argv))
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        pytest.fail(f"unexpected subprocess in staleness_pass: {argv}")

    monkeypatch.setattr(subprocess, "run", run)
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


@pytest.mark.parametrize(
    ("scenario", "delegated"), [(_delete, 1), (_edit, 0)], ids=["removal", "edit-only"]
)
def test_due_removal_delegates_while_the_coordinator_head_start_holds(
    wdog: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    scenario: Scenario,
    delegated: int,
) -> None:
    now = CHANGE + wdog.STALENESS_GRACE_SECS
    repo = _build_repo(tmp_path)
    scenario(repo, CHANGE)
    _point_at(wdog, monkeypatch, tmp_path, repo, now)
    head_start_open_for = 60
    clock_ts = now - wdog.ORCH_RESTART_MIN_INTERVAL_SECS - head_start_open_for
    _write_clock(pathlib.Path(wdog.FLEET_DEPLOY_CLOCK_PATH), clock_ts)
    delegations = _wire_pass(wdog, monkeypatch)

    wdog.staleness_pass()

    assert delegations == [_fleet_redeploy_argv(wdog)] * delegated


def test_removal_landing_before_the_deploy_stamp_redeploys_units_started_before_it(
    wdog: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    repo = _build_repo(tmp_path)
    _delete(repo, CHANGE)
    _point_at(wdog, monkeypatch, tmp_path, repo, CHANGE + wdog.MODULE_REMOVAL_GRACE_SECS)
    _write_clock(pathlib.Path(wdog.FLEET_DEPLOY_CLOCK_PATH), CHANGE + 60)
    delegations = _wire_pass(wdog, monkeypatch, {STALE: STARTED})

    wdog.staleness_pass()

    assert delegations == [_fleet_redeploy_argv(wdog)]


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


def test_waiver_stops_once_the_redeploy_restarts_the_units(
    wdog: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    now = CHANGE + wdog.MODULE_REMOVAL_GRACE_SECS
    repo = _build_repo(tmp_path)
    _delete(repo, CHANGE)
    _point_at(wdog, monkeypatch, tmp_path, repo, now)
    starts = {STALE: STARTED}
    delegations = _wire_pass(wdog, monkeypatch, starts)
    wdog.staleness_pass()
    assert len(delegations) == 1

    starts[STALE] = now
    _write_clock(pathlib.Path(wdog.FLEET_DEPLOY_CLOCK_PATH), now)
    wdog.staleness_pass()

    assert len(delegations) == 1
    assert wdog._due_module_removals() == ()


def test_units_restarted_mid_pass_cancel_the_waiver_delegation(
    wdog: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    now = CHANGE + wdog.MODULE_REMOVAL_GRACE_SECS
    repo = _build_repo(tmp_path)
    _delete(repo, CHANGE)
    lines = _point_at(wdog, monkeypatch, tmp_path, repo, now)
    delegations = _wire_pass(wdog, monkeypatch)
    starts_before_the_recheck = iter([STARTED, STARTED])
    monkeypatch.setattr(
        wdog, "_unit_start_epoch", lambda _unit: next(starts_before_the_recheck, now)
    )

    wdog.staleness_pass()

    assert delegations == []
    assert any("stamped while this pass was probing" in line for line in lines), lines


@pytest.mark.parametrize(
    ("into_the_slot", "warned"), [(0, True), (600, False)], ids=["in-slot", "between-slots"]
)
def test_failed_probe_delegates_nothing_and_warns_only_in_the_skip_log_slot(
    wdog: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    into_the_slot: int,
    warned: bool,
) -> None:
    repo = _build_repo(tmp_path)
    _delete(repo, CHANGE)
    now = _next_log_bucket(wdog, CHANGE + wdog.MODULE_REMOVAL_GRACE_SECS) + into_the_slot
    lines = _point_at(wdog, monkeypatch, tmp_path, repo, now)
    delegations = _wire_pass(wdog, monkeypatch)
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()
    monkeypatch.setattr(wdog, "REPO_DIR", str(not_a_repo))

    wdog.staleness_pass()

    assert delegations == []
    assert any(line.startswith("WARNING:") for line in lines) is warned, lines
