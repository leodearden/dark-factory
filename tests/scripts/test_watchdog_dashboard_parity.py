"""Tests for scripts/_watchdog_dashboard_parity.py, the watchdog's dashboard unit-parity tier (task 4883).

No live systemd: subprocess.run is faked everywhere except
test_unit_parity_verdict_runs_the_real_checker_and_reads_its_tag, which runs
the real checker under a tmp HOME. Every pass and row test hands the tier
tmp_path as its repo_dir, so the clock it reads and stamps lives under tmp_path
by construction rather than by redirecting a module global.

The module is imported by name, so ``parity`` is the sys.modules instance: the
same object scripts/orchestrator-watchdog.py's from-import binds, which is why
the _cli tests below patch it rather than the watchdog.
"""

import importlib.util
import json
import os
import pathlib
import subprocess
import time
import types
from collections.abc import Callable

import _watchdog_dashboard_parity as parity
import pytest
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = pathlib.Path(__file__).parents[2]
WATCHDOG_PATH = REPO_ROOT / "scripts" / "orchestrator-watchdog.py"
PARITY_PATH = REPO_ROOT / "scripts" / "_watchdog_dashboard_parity.py"


@pytest.fixture(autouse=True)
def _no_clock_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ambient ORCH_UNIT_PARITY_CLOCK would move every test's clock out of tmp_path."""
    monkeypatch.delenv("ORCH_UNIT_PARITY_CLOCK", raising=False)


def _discard(_msg: str) -> None:
    pass


def _load_watchdog() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("orchestrator_watchdog", WATCHDOG_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clock(repo_dir: pathlib.Path) -> pathlib.Path:
    """Return *repo_dir*'s parity clock, with its parent directory created."""
    clock = pathlib.Path(parity.unit_parity_clock_path(str(repo_dir)))
    clock.parent.mkdir(parents=True, exist_ok=True)
    return clock


# ---------------------------------------------------------------------------
# Dashboard unit-parity backstop (task 4883)
#
# A read-only, hourly-throttled run of scripts/check_dashboard_unit_parity.py
# on the already-armed watchdog tick, so the parity gate runs without anyone
# having to run setup-host.sh. No live systemd, no real checker, no real clock:
# subprocess.run is faked, and every test that lets unit_parity_pass run for
# real passes tmp_path as its repo_dir.
# ---------------------------------------------------------------------------

_DRIFT_REPORT = (
    "[dashboard_unit_parity] [drift] 1 directive(s) differ between repo and "
    "installed units:\n"
    "[dashboard_unit_parity]   dark-factory-dashboard.service [Service] "
    "SuccessExitStatus\n"
)


def _checker_run(
    calls: list[list[str]],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
    raises: BaseException | None = None,
) -> Callable[..., subprocess.CompletedProcess[str]]:
    """A subprocess.run fake that records every argv and answers like the checker."""

    def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr=stderr)

    return fake_run


@pytest.mark.parametrize(
    ("returncode", "verdict"), [(0, "parity"), (1, "drift"), (2, "absent")]
)
def test_unit_parity_verdict_classifies_the_checkers_exit_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, returncode: int, verdict: str
) -> None:
    """0/1/2 map to the checker's own parity/drift/absent vocabulary.

    The codes map only once the checker's tag shows it actually reported (see
    test_unit_parity_verdict_without_the_checkers_tag_is_unknown); this input
    is tagged. The combined stdout/stderr comes back with the verdict because
    that text is how an operator learns WHICH directive drifted.
    """
    calls: list[list[str]] = []
    monkeypatch.setattr(
        subprocess,
        "run",
        _checker_run(
            calls, returncode=returncode, stdout=_DRIFT_REPORT, stderr="checker stderr line"
        ),
    )

    got_verdict, report = parity.unit_parity_verdict(str(tmp_path))

    assert got_verdict == verdict
    assert isinstance(got_verdict, parity.UnitParityVerdict), (
        f"verdicts are a closed type so a misspelled comparison fails pyright: {got_verdict!r}"
    )
    assert "SuccessExitStatus" in report, f"stdout must reach the report: {report!r}"
    assert "checker stderr line" in report, f"stderr must reach the report: {report!r}"


def test_unit_parity_verdict_runs_the_checker_against_the_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The checker is the repo's own, pointed at the repo, and left to choose its installed dir.

    --installed-dir is deliberately NOT passed: the checker's own default is
    the one setup-host.sh gates against, and re-deriving it here would be a
    second copy of it.
    """
    repo_dir = str(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _checker_run(calls))

    parity.unit_parity_verdict(repo_dir)

    assert len(calls) == 1, calls
    (argv,) = calls
    assert os.path.join(repo_dir, parity.DASHBOARD_PARITY_SCRIPT_RELPATH) in argv, argv
    assert "--repo-root" in argv, argv
    assert argv[argv.index("--repo-root") + 1] == repo_dir, argv
    assert "--installed-dir" not in argv, argv


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError(2, "No such file or directory", "python3"),
        PermissionError(13, "Permission denied"),
        subprocess.TimeoutExpired(cmd="check_dashboard_unit_parity.py", timeout=30),
    ],
    ids=["not-found", "permission", "timeout"],
)
def test_unit_parity_verdict_tooling_failure_is_unknown_and_never_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, error: BaseException
) -> None:
    """A checker that cannot run is 'unknown', never an exception.

    Same never-raises contract as the watchdog's probe_port and log(): a
    tooling failure in an observational pass must not abort the tick it runs
    on.
    """
    monkeypatch.setattr(subprocess, "run", _checker_run([], raises=error))

    verdict, _report = parity.unit_parity_verdict(str(tmp_path))

    assert verdict == "unknown"


def test_unit_parity_verdict_unexpected_exit_code_is_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A TAGGED exit code outside 0/1/2 is not a parity claim, and says which code it was."""
    monkeypatch.setattr(
        subprocess, "run", _checker_run([], returncode=7, stdout="[dashboard_unit_parity] boom")
    )

    verdict, report = parity.unit_parity_verdict(str(tmp_path))

    assert verdict == "unknown"
    assert "7" in report, f"the report must name the unexpected exit code: {report!r}"


# Real failure shapes that carry NO [dashboard_unit_parity] tag, so the checker
# never reported. Exit 2 is also python3's can't-open-file status and
# argparse's usage-error status, and an uncaught traceback exits 1.
_UNTAGGED_CHECKER_OUTPUTS = [
    pytest.param(
        2,
        "",
        "python3: can't open file '/nonexistent/check_dashboard_unit_parity.py': "
        "[Errno 2] No such file or directory",
        id="cant-open",
    ),
    pytest.param(
        2,
        "",
        "usage: check_dashboard_unit_parity.py [-h] [--installed-dir INSTALLED_DIR]\n"
        "check_dashboard_unit_parity.py: error: unrecognized arguments: --repo-root /x",
        id="argparse-reject",
    ),
    pytest.param(
        1,
        "",
        "Traceback (most recent call last):\n"
        '  File "check_dashboard_unit_parity.py", line 1, in <module>\n'
        "ValueError: boom",
        id="traceback",
    ),
    pytest.param(0, "", "", id="silent-zero"),
]


@pytest.mark.parametrize(("returncode", "stdout", "stderr"), _UNTAGGED_CHECKER_OUTPUTS)
def test_unit_parity_verdict_without_the_checkers_tag_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    returncode: int,
    stdout: str,
    stderr: str,
) -> None:
    """No tag, no verdict — whatever the exit code says.

    Trusting the code alone turns a moved checker or a renamed --repo-root
    into a permanently silent 'absent', and a crash into a WARNING drift.
    'silent-zero' proves the tag outranks even a parity-looking 0, which is
    scripts/setup-host.sh::_parity_verdict's own rule.
    """
    monkeypatch.setattr(
        subprocess,
        "run",
        _checker_run([], returncode=returncode, stdout=stdout, stderr=stderr),
    )

    verdict, report = parity.unit_parity_verdict(str(tmp_path))

    assert verdict == "unknown", report
    assert stderr in report, f"the captured stderr must reach the report: {report!r}"
    assert f"exit {returncode}" in report, f"the report must name the exit status: {report!r}"


def test_unit_parity_verdict_runs_the_real_checker_and_reads_its_tag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """CONTRACT LOCK: the real checker, run the way the watchdog runs it, reports.

    The one test here that does not fake subprocess.run. Every other one
    would stay green through a renamed or moved checker, a renamed
    --repo-root, or a changed tag, while the hourly pass went quiet.

    This checkout is the repo_dir, so the checker that runs is the code under
    test, found at DASHBOARD_PARITY_SCRIPT_RELPATH, so a rename still fails.
    HOME points at an empty tmp dir, so the checker's default installed dir
    holds no units and the real ones are never read, while the production
    argv stays intact. 'absent' is then the only correct answer, and getting
    it proves the script runs at that path, accepts --repo-root, and prints
    the tag the watchdog searches for.
    """
    assert (REPO_ROOT / parity.DASHBOARD_PARITY_SCRIPT_RELPATH).is_file(), (
        f"DASHBOARD_PARITY_SCRIPT_RELPATH names {parity.DASHBOARD_PARITY_SCRIPT_RELPATH}, "
        "which this checkout does not have"
    )
    monkeypatch.setenv("HOME", str(tmp_path))

    verdict, report = parity.unit_parity_verdict(str(REPO_ROOT))

    assert verdict == "absent", f"expected 'absent' from an empty installed dir: {report!r}"


def _pass_log_lines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    verdict: str,
    report: str = "REPORT-TEXT naming [Service] SuccessExitStatus",
) -> list[str]:
    """Run unit_parity_pass once against a stubbed *verdict*; return what it logged."""
    lines: list[str] = []
    monkeypatch.setattr(parity, "unit_parity_verdict", lambda _repo_dir: (verdict, report))
    parity.unit_parity_pass(str(tmp_path), lines.append)
    return lines


def test_unit_parity_pass_warns_once_on_drift_with_the_checkers_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Drift is the one verdict an operator must act on, so it is the one WARNING."""
    lines = _pass_log_lines(monkeypatch, tmp_path, "drift")

    assert len(lines) == 1, lines
    (line,) = lines
    assert "WARNING" in line, line
    assert "unit parity" in line, line
    assert "REPORT-TEXT naming [Service] SuccessExitStatus" in line, line


@pytest.mark.parametrize("tag", ["[drift]", "[override]", "[vanished]"])
def test_unit_parity_pass_warning_headline_names_every_exit_1_case(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, tag: str
) -> None:
    """The checker exits 1 for three findings, so the headline must not name only one.

    A drop-in override or a vanished committed unit is not a directive diff,
    and a headline asserting drift would send the operator to the wrong fix.
    End to end through a faked checker, so it is the real verdict that reaches
    the headline.
    """
    lines: list[str] = []
    report = f"[dashboard_unit_parity] {tag} finding\n"
    monkeypatch.setattr(subprocess, "run", _checker_run([], returncode=1, stdout=report))

    parity.unit_parity_pass(str(tmp_path), lines.append)

    assert len(lines) == 1, lines
    headline = lines[0].split("\n", 1)[0]
    assert "WARNING" in headline, headline
    assert tag in headline, f"the headline must name the {tag} case: {headline!r}"


@pytest.mark.parametrize("verdict", ["parity", "absent"])
def test_unit_parity_pass_is_silent_on_parity_and_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, verdict: str
) -> None:
    """Parity needs no line, and neither does 'absent'.

    Exit 2 means the dashboard units are not installed on this host, which
    setup-host.sh already treats as benign. Warning on it would make every
    checkout on a host without the dashboard permanently noisy.
    """
    assert _pass_log_lines(monkeypatch, tmp_path, verdict) == []


def test_unit_parity_pass_reports_a_broken_checker_without_a_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """'unknown' gets one line, worded apart from drift.

    A gate that could not run must be distinguishable from a green one, but it
    is a tooling problem rather than a parity claim, so it is not a WARNING.
    """
    lines = _pass_log_lines(monkeypatch, tmp_path, "unknown", report="PermissionError")

    assert len(lines) == 1, lines
    (line,) = lines
    assert "WARNING" not in line, line
    assert "unit parity" in line, line
    assert "PermissionError" in line, line


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr"),
    [p for p in _UNTAGGED_CHECKER_OUTPUTS if p.id in {"cant-open", "traceback"}],
)
def test_unit_parity_pass_reports_an_untagged_exit_instead_of_trusting_its_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    returncode: int,
    stdout: str,
    stderr: str,
) -> None:
    """End to end through the pass: a checker that never reported is one plain line.

    Not a stubbed verdict — subprocess.run is faked, so the regression is
    locked where the hourly pass actually reads it. Trusting the code alone,
    a moved checker (rc=2) logged nothing at all, and a crash (rc=1) logged a
    WARNING that the installed units had drifted.
    """
    lines: list[str] = []
    monkeypatch.setattr(
        subprocess,
        "run",
        _checker_run([], returncode=returncode, stdout=stdout, stderr=stderr),
    )

    parity.unit_parity_pass(str(tmp_path), lines.append)

    assert len(lines) == 1, lines
    (line,) = lines
    assert "WARNING" not in line, line
    assert "unit parity" in line, line
    assert stderr in line, line


def test_unit_parity_pass_never_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """An unexpected failure inside the pass is logged, never raised out of the tick."""
    lines: list[str] = []

    def boom(_repo_dir: str) -> tuple[str, str]:
        raise RuntimeError("unexpected parity failure")

    monkeypatch.setattr(parity, "unit_parity_verdict", boom)

    parity.unit_parity_pass(str(tmp_path), lines.append)

    assert any("unexpected parity failure" in line for line in lines), lines


def test_unit_parity_pass_issues_no_mutating_systemd_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """READ-ONLY: the pass detects drift and never acts on it.

    The journal sink is injected, so the checker is the only subprocess the
    pass itself can spawn, and every one it does spawn reaches the recorder.
    """
    calls: list[list[str]] = []
    monkeypatch.setattr(
        subprocess, "run", _checker_run(calls, returncode=1, stdout=_DRIFT_REPORT)
    )

    parity.unit_parity_pass(str(tmp_path), _discard)

    assert any(
        arg.endswith("check_dashboard_unit_parity.py") for argv in calls for arg in argv
    ), f"sanity: the checker must have run: {calls}"
    mutating = [argv for argv in calls if argv and argv[0] in {"systemctl", "systemd-run"}]
    assert mutating == [], mutating


def _count_checker_runs(monkeypatch: pytest.MonkeyPatch, verdict: str = "parity") -> list[str]:
    """Stub the verdict with a recorder; return the list each run appends to."""
    runs: list[str] = []

    def fake_verdict(_repo_dir: str) -> tuple[str, str]:
        runs.append(verdict)
        return verdict, ""

    monkeypatch.setattr(parity, "unit_parity_verdict", fake_verdict)
    return runs


def test_unit_parity_pass_skips_inside_the_min_interval(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A fresh stamp holds the checker off: ~1440 ticks/day must not mean ~1440 runs."""
    _clock(tmp_path).write_text(json.dumps({"ts": int(time.time())}))
    runs = _count_checker_runs(monkeypatch)

    parity.unit_parity_pass(str(tmp_path), _discard)

    assert runs == []


def test_unit_parity_pass_runs_and_stamps_when_the_clock_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """No clock yet means the check is due: it runs once and stamps the clock."""
    clock = pathlib.Path(parity.unit_parity_clock_path(str(tmp_path)))
    runs = _count_checker_runs(monkeypatch)

    parity.unit_parity_pass(str(tmp_path), _discard)

    assert len(runs) == 1, runs
    assert clock.exists(), "the pass must stamp its clock"
    assert abs(json.loads(clock.read_text())["ts"] - time.time()) < 60


def test_unit_parity_pass_stamps_on_attempt_even_when_the_checker_cannot_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The stamp is written on ATTEMPT, not on success.

    Otherwise a persistently broken checker would re-run, and re-log, on every
    one of the ~1440 ticks a day.
    """
    clock = pathlib.Path(parity.unit_parity_clock_path(str(tmp_path)))
    runs = _count_checker_runs(monkeypatch, verdict="unknown")

    parity.unit_parity_pass(str(tmp_path), _discard)
    parity.unit_parity_pass(str(tmp_path), _discard)

    assert clock.exists(), "an 'unknown' verdict must still stamp the clock"
    assert len(runs) == 1, f"the second tick must be held off by the stamp: {runs}"


def test_unit_parity_pass_fails_open_on_a_corrupt_clock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A corrupt clock body lets the check RUN — the read_clock_epoch contract.

    Failing the other way would let one unreadable file silence the gate for
    good.
    """
    _clock(tmp_path).write_text("{not json")
    runs = _count_checker_runs(monkeypatch)

    parity.unit_parity_pass(str(tmp_path), _discard)

    assert len(runs) == 1, runs


def test_unit_parity_min_interval_of_zero_disables_the_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """<=0 disables the throttle, the shared within_min_interval contract."""
    _clock(tmp_path).write_text(json.dumps({"ts": int(time.time())}))
    monkeypatch.setattr(parity, "UNIT_PARITY_MIN_INTERVAL_SECS", 0)
    runs = _count_checker_runs(monkeypatch)

    parity.unit_parity_pass(str(tmp_path), _discard)

    assert len(runs) == 1, runs


def test_unit_parity_clock_path_default() -> None:
    """The parity clock is its OWN file under <repo_dir>/data/orchestrator/."""
    assert parity.unit_parity_clock_path("/r") == os.path.join(
        "/r", "data", "orchestrator", "last_unit_parity_check.json"
    )


def test_unit_parity_clock_is_none_of_the_watchdogs_clocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinct from every deploy and restart clock under the watchdog's own REPO_DIR.

    So a redeploy never resets the parity cadence and a parity check never
    moves a deploy gate.
    """
    for env in ("ORCH_FLEET_DEPLOY_CLOCK", "FM_DEPLOY_CLOCK", "FM_LIVENESS_RESTART_CLOCK"):
        monkeypatch.delenv(env, raising=False)
    wdog = _load_watchdog()

    assert parity.unit_parity_clock_path(wdog.REPO_DIR) not in {
        wdog.FLEET_DEPLOY_CLOCK_PATH,
        wdog.FM_DEPLOY_CLOCK_PATH,
        wdog.FM_LIVENESS_RESTART_CLOCK_PATH,
    }


def test_unit_parity_clock_path_honors_its_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ORCH_UNIT_PARITY_CLOCK, when set, is the clock, whatever the repo_dir."""
    monkeypatch.setenv("ORCH_UNIT_PARITY_CLOCK", "/tmp/custom_unit_parity_clock.json")

    assert parity.unit_parity_clock_path("/r") == "/tmp/custom_unit_parity_clock.json"


def _fresh_parity_module() -> types.ModuleType:
    """Exec a FRESH copy of the tier, so its import-time env reads run again.

    Under a name never inserted into sys.modules, so the instance the
    watchdog's from-import binds is left untouched.
    """
    spec = importlib.util.spec_from_file_location("_watchdog_dashboard_parity_fresh", PARITY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unit_parity_min_interval_honors_its_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ORCH_UNIT_PARITY_MIN_INTERVAL_SECS is read at import."""
    monkeypatch.setenv("ORCH_UNIT_PARITY_MIN_INTERVAL_SECS", "120")

    assert _fresh_parity_module().UNIT_PARITY_MIN_INTERVAL_SECS == 120


def test_unit_parity_min_interval_falls_back_on_a_malformed_env_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo'd interval falls back to the hourly default rather than crashing the oneshot."""
    monkeypatch.setenv("ORCH_UNIT_PARITY_MIN_INTERVAL_SECS", "hourly")

    assert _fresh_parity_module().UNIT_PARITY_MIN_INTERVAL_SECS == 3600


# ---------------------------------------------------------------------------
# The dashboard unit-parity --report row (task 4883)
#
# The operator's "right now" view of the same check unit_parity_pass runs
# hourly: never clock-gated, never writes a clock, never alters --report's
# exit code. The checker itself is always faked.
# ---------------------------------------------------------------------------


def _parity_row(out: str) -> str:
    """Return the single dashboard unit-parity row from --report output."""
    rows = [line for line in out.splitlines() if line.startswith("dashboard unit parity")]
    assert len(rows) == 1, f"expected exactly one parity row, got {rows!r} in {out!r}"
    return rows[0]


def _row_verdict(row: str) -> str:
    """Return the verdict field of a parity row: between the first ':' and the first '|'.

    Asserting ``"parity" in row`` is vacuous, because the row's label,
    ``dashboard unit parity``, always contains the word.
    """
    return row.split(":", 1)[1].split("|", 1)[0].strip()


@pytest.mark.parametrize("verdict", ["parity", "drift", "absent", "unknown"])
def test_print_unit_parity_prints_one_labelled_row(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: pathlib.Path,
    verdict: str,
) -> None:
    """One row, in the fm row's ``LABEL: value | LABEL: value`` shape.

    The CHECKER field names the checker relative to the repo, whatever
    repo_dir is.
    """
    monkeypatch.setattr(parity, "unit_parity_verdict", lambda _repo_dir: (verdict, ""))

    parity.print_unit_parity(str(tmp_path), _discard)

    row = _parity_row(capsys.readouterr().out)
    assert _row_verdict(row) == verdict, row
    assert row.endswith(f" | CHECKER: {parity.DASHBOARD_PARITY_SCRIPT_RELPATH}"), (
        f"the row must keep --report's LABEL: value | LABEL: value shape: {row!r}"
    )


def test_print_unit_parity_shows_the_drifted_directive(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: pathlib.Path
) -> None:
    """On drift the checker's report follows the row, naming the directive.

    The operator must see WHICH directive drifted without a second command.
    """
    monkeypatch.setattr(parity, "unit_parity_verdict", lambda _repo_dir: ("drift", _DRIFT_REPORT))

    parity.print_unit_parity(str(tmp_path), _discard)

    out = capsys.readouterr().out
    assert "drift" in _parity_row(out)
    assert "SuccessExitStatus" in out, out


def test_print_unit_parity_is_not_clock_gated_and_writes_no_clock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--report answers RIGHT NOW, and leaves the timer path's cadence alone.

    A fresh stamp would hold unit_parity_pass off; it must not hold the row
    off. And a read-only doctor invocation must not move the clock, or running
    --report would itself change when the next hourly check happens.
    """
    clock = _clock(tmp_path)
    clock.write_text(json.dumps({"ts": int(time.time()), "iso": "fresh"}) + "\n")
    before = clock.read_bytes()
    calls: list[list[str]] = []
    monkeypatch.setattr(
        subprocess,
        "run",
        _checker_run(
            calls,
            returncode=0,
            stdout="[dashboard_unit_parity] [ok] parity — 3 unit(s) match their committed copies.",
        ),
    )

    parity.print_unit_parity(str(tmp_path), _discard)

    checker_runs = [
        argv for argv in calls if any(a.endswith("check_dashboard_unit_parity.py") for a in argv)
    ]
    assert len(checker_runs) == 1, f"the row must run the checker despite a fresh stamp: {calls}"
    assert clock.read_bytes() == before, "--report must never write the parity clock"
    assert _row_verdict(_parity_row(capsys.readouterr().out)) == "parity"


@pytest.mark.parametrize("report_rc", [0, 1])
def test_cli_report_exit_code_is_unchanged_by_a_drifted_parity_row(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], report_rc: int
) -> None:
    """The parity row is informational: --report returns report()'s OWN code.

    A drifted dashboard unit must not silently change what --report's exit
    code means to its existing callers.
    """
    wdog = _load_watchdog()
    monkeypatch.setattr(wdog, "report", lambda: report_rc)
    monkeypatch.setattr(wdog, "_print_fused_memory_liveness", lambda: None)
    monkeypatch.setattr(parity, "unit_parity_verdict", lambda _repo_dir: ("drift", _DRIFT_REPORT))

    assert wdog._cli(["--report"]) == report_rc
    assert "drift" in _parity_row(capsys.readouterr().out)


def test_cli_report_runs_no_mutating_pass_including_unit_parity(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--report never runs a timer-path pass — unit_parity_pass included.

    That pass STAMPS a clock, so running it under --report would break the
    read-only doctor contract even though the check itself is read-only.
    """
    wdog = _load_watchdog()
    monkeypatch.setattr(wdog, "report", lambda: 0)
    monkeypatch.setattr(wdog, "_print_fused_memory_liveness", lambda: None)
    monkeypatch.setattr(parity, "unit_parity_verdict", lambda _repo_dir: ("parity", ""))
    for name in (
        "main",
        "fused_memory_liveness_pass",
        "staleness_pass",
        "fused_memory_staleness_pass",
        "unit_parity_pass",
    ):
        monkeypatch.setattr(
            wdog,
            name,
            lambda *_, name=name: pytest.fail(f"{name}() must not run under --report"),
        )

    assert wdog._cli(["--report"]) == 0
    assert _row_verdict(_parity_row(capsys.readouterr().out)) == "parity"


def test_print_unit_parity_degrades_to_unknown_on_an_unexpected_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failure inside the verdict costs this row its verdict, not --report its run.

    Driven through _cli with the watchdog's own log patched, so the error line
    reaching it proves _cli hands the tier the watchdog's journal sink.
    """
    wdog = _load_watchdog()
    logged: list[str] = []
    monkeypatch.setattr(wdog, "log", logged.append)
    monkeypatch.setattr(wdog, "report", lambda: 0)
    monkeypatch.setattr(wdog, "_print_fused_memory_liveness", lambda: None)

    def boom(_repo_dir: str) -> tuple[str, str]:
        raise RuntimeError("parity checker exploded")

    monkeypatch.setattr(parity, "unit_parity_verdict", boom)

    assert wdog._cli(["--report"]) == 0
    assert "unknown" in _parity_row(capsys.readouterr().out)
    assert any("parity checker exploded" in line for line in logged), logged
