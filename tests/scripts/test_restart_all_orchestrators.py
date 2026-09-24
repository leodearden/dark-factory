"""I2 recorder tests for scripts/restart-all-orchestrators.sh's fleet-deploy
clock stamp (task 2396, fleet-redeploy β).

Drives the script via subprocess against a fake `systemctl` shell script
shimmed onto PATH — the fake-binary-on-PATH harness pattern from
test_spawn_claude.py (tmp bin dir, chmod 0o755, PATH-prepended env,
subprocess.run([script], env=..., capture_output=True)).

Only the clock-stamp contract is covered here: a verified-fresh restart
(script exit 0) stamps ORCH_FLEET_DEPLOY_CLOCK; a failed verify (script
exit 1) must leave it untouched. The `--drain` merge-drain gate has its
own dedicated suite at scripts/tests/test_restart_all_orchestrators.py
(task 2397) — not duplicated here.

Also covers the VERIFY_TIMEOUT grace re-probe (task 2961): a unit whose
own stop/start is still in flight when VERIFY_TIMEOUT expires (its restart
job superseded/canceled and re-run by systemd's own supervision) must NOT
be declared failed until a further RESTART_VERIFY_GRACE_SECS re-probe
window also elapses with no fresh reading. The "delayed-fresh" scenario
below simulates that: the fake systemctl reports stale for the first
FAKE_SYSTEMCTL_FRESH_AFTER_CALLS `show` calls after `restart`, then fresh.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
# APPEND, never insert(0, ...): the repo root must stay LAST on sys.path or the
# subproject directories resolve as namespace packages shadowing their own
# src/<pkg>/ — the failure the root conftest.py docstring exists to prevent.
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from df_pytest_isolation import (  # noqa: E402
    CLOCK_PROVENANCE_SESSION_KEY,
    CLOCK_PROVENANCE_SOURCE_KEY,
    PYTEST_SESSION_TOKEN_ENV,
    assert_synthetic_units,
)

SCRIPT = REPO_ROOT / "scripts" / "restart-all-orchestrators.sh"
# The ORIGINAL synthetic literal, and the precedent task 3799's allowlist prefix
# was chosen around: `orchestrator-fake` with no stem is a legal fixture name, so
# this file needed no rename. Left as a literal rather than routed through
# synthetic_unit() because it has no stem to name; _run_script below still puts
# it through the same checker every other fixture name goes through.
UNIT_NAME = "orchestrator-fake.service"

# Stateful fake `systemctl`: `list-units` reports one fake orchestrator unit;
# `show -p <fields>` reports a baseline MainPID/ActiveState/
# ActiveEnterTimestamp(Monotonic) until the marker file `restart` touches
# exists, at which point -- scenario "fresh" only -- it reports a fresh,
# higher monotonic timestamp (a verified restart). Scenario "stale" never
# advances the monotonic timestamp even after `restart`, simulating a
# restart that never actually came back up fresh (the NEGATIVE/I2 case).
# Scenario "delayed-fresh" (task 2961) reports stale for the first
# FAKE_SYSTEMCTL_FRESH_AFTER_CALLS post-restart `show` calls, then flips to
# fresh -- simulating a slow-draining unit whose actual start lands AFTER
# VERIFY_TIMEOUT already expired, only within the grace re-probe window.
_FAKE_SYSTEMCTL = textwrap.dedent("""\
    #!/usr/bin/env bash
    set -euo pipefail

    MARKER="${FAKE_SYSTEMCTL_MARKER:?FAKE_SYSTEMCTL_MARKER not set}"
    SCENARIO="${FAKE_SYSTEMCTL_SCENARIO:-fresh}"
    UNIT_NAME="${FAKE_SYSTEMCTL_UNIT:?FAKE_SYSTEMCTL_UNIT not set}"
    COUNTER="${MARKER}.count"
    FRESH_AFTER="${FAKE_SYSTEMCTL_FRESH_AFTER_CALLS:-0}"

    args=()
    for a in "$@"; do
        [[ "$a" == "--user" ]] || args+=("$a")
    done
    verb="${args[0]:-}"

    # Mid-sweep lease observation (task 4755). This fake runs as a descendant
    # of the sweep, so it is the only vantage point from which the in-flight
    # lease can be seen WHILE it is held; the test process only ever sees the
    # before and the after, and "the file is absent afterwards" is equally
    # true of a script that never wrote one.
    LEASE_LOG="${FAKE_SYSTEMCTL_LEASE_LOG:-}"
    LEASE_PATH="${ORCH_FLEET_LEASE:-}"
    if [[ -n "$LEASE_LOG" && -n "$LEASE_PATH" && -f "$LEASE_PATH" ]]; then
        printf '%s\\t%s\\n' "$verb" "$(tr -d '\\n' < "$LEASE_PATH")" >> "$LEASE_LOG"
    fi

    case "$verb" in
        list-units)
            # FAKE_SYSTEMCTL_NO_UNITS (task 4755) reports an EMPTY fleet, the
            # one input that reaches the script's early "nothing to restart"
            # exit-0 -- an exit path that must still release the lease even
            # though it never stamps the clock.
            if [[ "${FAKE_SYSTEMCTL_NO_UNITS:-0}" != "1" ]]; then
                echo "${UNIT_NAME} loaded active running Orchestrator"
            fi
            ;;
        restart)
            touch "$MARKER"
            rm -f "$COUNTER"
            # FAKE_SYSTEMCTL_LEASE_SWAP (task 4755) stands in for a SECOND
            # sweep that started while this one was mid-flight and took the
            # lease for itself. Written from here because "mid-flight" is the
            # only moment at which it is a hand-over rather than litter.
            if [[ -n "${FAKE_SYSTEMCTL_LEASE_SWAP:-}" && -n "$LEASE_PATH" ]]; then
                printf '%s' "$FAKE_SYSTEMCTL_LEASE_SWAP" > "$LEASE_PATH"
            fi
            ;;
        show)
            is_fresh=0
            if [[ -f "$MARKER" ]]; then
                if [[ "$SCENARIO" == "fresh" ]]; then
                    is_fresh=1
                elif [[ "$SCENARIO" == "delayed-fresh" ]]; then
                    count=0
                    [[ -f "$COUNTER" ]] && count="$(cat "$COUNTER")"
                    count=$((count + 1))
                    echo "$count" > "$COUNTER"
                    if [[ "$count" -gt "$FRESH_AFTER" ]]; then
                        is_fresh=1
                    fi
                fi
            fi
            if [[ "$is_fresh" == "1" ]]; then
                pid=1001
                mono=2000000
                ts=restarted
            else
                pid=1000
                mono=1000000
                ts=baseline
            fi
            printf 'MainPID=%s\\n' "$pid"
            printf 'ActiveState=active\\n'
            printf 'ActiveEnterTimestamp=%s\\n' "$ts"
            printf 'ActiveEnterTimestampMonotonic=%s\\n' "$mono"
            ;;
        *)
            exit 1
            ;;
    esac
""")


def _make_fake_systemctl(bin_dir: Path) -> None:
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "systemctl"
    fake.write_text(_FAKE_SYSTEMCTL)
    fake.chmod(0o755)


def _run_script(
    tmp_path: Path,
    *,
    scenario: str,
    clock_file: Path,
    verify_timeout: str = "2",
    verify_grace: str = "1",
    fresh_after_calls: str | None = None,
    lease_file: Path | None = None,
    lease_swap: dict | None = None,
    no_units: bool = False,
    extra_args: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[bytes]:
    bin_dir = tmp_path / "bin"
    _make_fake_systemctl(bin_dir)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["FAKE_SYSTEMCTL_MARKER"] = str(tmp_path / "restarted.marker")
    env["FAKE_SYSTEMCTL_SCENARIO"] = scenario
    # The PATH-shimming seam for this file (task 3799): the unit name reaches the
    # fake through the environment rather than through _make_fake_systemctl, so
    # the check belongs here. Same hazard as the sibling factories -- the fake
    # shadows `systemctl` only while its tmpdir lives on PATH.
    assert_synthetic_units(
        [UNIT_NAME],
        where="tests/scripts/test_restart_all_orchestrators.py::_run_script",
    )
    env["FAKE_SYSTEMCTL_UNIT"] = UNIT_NAME
    env["ORCH_FLEET_DEPLOY_CLOCK"] = str(clock_file)
    env["RESTART_VERIFY_TIMEOUT"] = verify_timeout
    # RESTART_VERIFY_GRACE_SECS (task 2961): default kept small (1s) here so
    # tests that don't care about the grace re-probe (e.g. the happy-path
    # fresh-on-first-check case) stay fast; tests exercising the grace
    # window override verify_timeout/verify_grace/fresh_after_calls
    # explicitly.
    env["RESTART_VERIFY_GRACE_SECS"] = verify_grace
    if fresh_after_calls is not None:
        env["FAKE_SYSTEMCTL_FRESH_AFTER_CALLS"] = fresh_after_calls
    # Per-test, always (task 4755). The session-wide redirect in
    # df_pytest_isolation keeps a forgetful spawner off the LIVE lease path,
    # but it is ONE path shared by the whole session -- a test that asserts on
    # lease contents must own its own file, exactly as the clock tests above do.
    if lease_file is not None:
        env["ORCH_FLEET_LEASE"] = str(lease_file)
        env["FAKE_SYSTEMCTL_LEASE_LOG"] = str(_lease_log_path(tmp_path))
    if lease_swap is not None:
        env["FAKE_SYSTEMCTL_LEASE_SWAP"] = json.dumps(lease_swap)
    if no_units:
        env["FAKE_SYSTEMCTL_NO_UNITS"] = "1"

    return subprocess.run(
        [str(SCRIPT), *extra_args],
        env=env,
        capture_output=True,
        timeout=30,
    )


def test_verified_fresh_restart_stamps_the_fleet_deploy_clock(tmp_path: Path) -> None:
    """HAPPY/I2-positive: a verified-fresh restart stamps the clock file.

    The script must exit 0 and the clock file must exist afterward with a
    numeric `ts` -- the shape both the coordinator and the watchdog read.
    """
    clock_file = tmp_path / "last_redeploy_orchestrator.json"
    assert not clock_file.exists()

    result = _run_script(tmp_path, scenario="fresh", clock_file=clock_file)

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert clock_file.exists(), "clock file must be stamped on verified-fresh exit-0"
    stamped = json.loads(clock_file.read_text())
    assert isinstance(stamped["ts"], (int, float)), f"ts must be numeric; got {stamped!r}"


def test_failed_verify_leaves_the_fleet_deploy_clock_unchanged(tmp_path: Path) -> None:
    """NEGATIVE/I2: a failed verify (mono never advances) must NOT stamp.

    The clock file is pre-seeded with a sentinel value; the script must
    exit 1 and the file must be byte-identical afterward -- a failed
    detached/backstop restart must never silence the other tier for a
    full min_interval window.
    """
    clock_file = tmp_path / "last_redeploy_orchestrator.json"
    sentinel = '{"ts": 1.0}'
    clock_file.write_text(sentinel)

    result = _run_script(tmp_path, scenario="stale", clock_file=clock_file)

    assert result.returncode == 1, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert clock_file.read_text() == sentinel, (
        f"clock file must be byte-identical after a failed verify; got {clock_file.read_text()!r}"
    )


# ---------------------------------------------------------------------------
# task 2961: VERIFY_TIMEOUT grace re-probe
# ---------------------------------------------------------------------------

def test_unit_fresh_only_during_grace_still_verifies_and_stamps(tmp_path: Path) -> None:
    """POSITIVE: a unit that is still stale when VERIFY_TIMEOUT expires but
    turns fresh during the grace re-probe window must be treated as a
    verified restart -- exit 0, clock stamped, no FAILED declaration -- not
    a false failure/escalation (the reify incident this task fixes).

    VERIFY_TIMEOUT=1 gives exactly one `show` call before the initial
    window expires (still stale, since FRESH_AFTER=2). The grace window
    (5s) then gets several more 1/sec polls, flipping fresh on the third
    call.
    """
    clock_file = tmp_path / "last_redeploy_orchestrator.json"

    result = _run_script(
        tmp_path,
        scenario="delayed-fresh",
        clock_file=clock_file,
        verify_timeout="1",
        verify_grace="5",
        fresh_after_calls="2",
    )

    stdout = result.stdout.decode(errors="replace")
    assert result.returncode == 0, f"stdout={stdout!r} stderr={result.stderr!r}"
    assert "FAILED" not in stdout, f"must not declare FAILED; got stdout={stdout!r}"
    assert "re-probing" in stdout, (
        f"expected the grace re-probe line; got stdout={stdout!r}"
    )
    assert clock_file.exists(), "clock file must be stamped once the grace re-probe verifies fresh"
    stamped = json.loads(clock_file.read_text())
    assert isinstance(stamped["ts"], (int, float)), f"ts must be numeric; got {stamped!r}"


def test_unit_never_fresh_through_grace_still_fails(tmp_path: Path) -> None:
    """NEGATIVE: a genuinely dead unit (never turns fresh) still exits 1
    once BOTH the initial VERIFY_TIMEOUT and the grace window elapse -- the
    grace re-probe must not turn a real failure into a false success."""
    clock_file = tmp_path / "last_redeploy_orchestrator.json"
    sentinel = '{"ts": 1.0}'
    clock_file.write_text(sentinel)

    result = _run_script(
        tmp_path,
        scenario="stale",
        clock_file=clock_file,
        verify_timeout="1",
        verify_grace="1",
    )

    stdout = result.stdout.decode(errors="replace")
    assert result.returncode == 1, f"stdout={stdout!r} stderr={result.stderr!r}"
    assert "FAILED" in stdout, f"expected an eventual FAILED line; got stdout={stdout!r}"
    assert clock_file.read_text() == sentinel, (
        f"clock file must be byte-identical after a failed verify; got {clock_file.read_text()!r}"
    )


# ---------------------------------------------------------------------------
# task 4823: the stamp carries its own provenance
#
# The WRITE side of the fix, pinned end-to-end through the real script;
# df_pytest_isolation.py::deploy_clock_change_report states what the provenance
# is for and what it buys.
#
# Every assertion below names the key through the df_pytest_isolation constants
# rather than through a string of its own. That is what makes this a drift pin
# rather than a tautology: the pytest guard reads the keys it defines, so if
# this script's literals ever diverged from them, the guard would stop
# recognising production writes — with every test here still green if they
# compared literals to literals.
# ---------------------------------------------------------------------------

# The `source` value this script must claim: its own filename, which is what an
# operator reading the clock file by hand needs in order to know which of the
# two writers stamped it.
_EXPECTED_SOURCE = "restart-all-orchestrators.sh"


def test_the_stamp_carries_its_writer_and_this_sessions_token(tmp_path: Path) -> None:
    """A test-spawned stamp is TAGGED as test-spawned — the whole discriminator.

    The script inherits the ambient token because _run_script builds its child
    env from dict(os.environ), which is the same free-tagging property the
    drain-leak guard relies on. This is the write that must keep failing a run
    (task 3797's exact defect: a test driving the REAL script against a fake
    systemctl), and it is now self-evidently that rather than an inference from
    "the bytes moved".
    """
    clock_file = tmp_path / "last_redeploy_orchestrator.json"

    result = _run_script(tmp_path, scenario="fresh", clock_file=clock_file)

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    stamped = json.loads(clock_file.read_text())
    assert stamped[CLOCK_PROVENANCE_SOURCE_KEY] == _EXPECTED_SOURCE
    assert stamped[CLOCK_PROVENANCE_SESSION_KEY] == os.environ[PYTEST_SESSION_TOKEN_ENV]


def test_a_production_stamp_carries_an_empty_session_token(
    tmp_path: Path, monkeypatch
) -> None:
    """The shape every GENUINE machine-operated redeploy writes.

    An empty token is a positive statement — "no pytest session was an ancestor
    of this write" — not an omission, which is why the key is always present.
    An omitted key would be indistinguishable from a pre-4823 writer and would
    (correctly, but uselessly) keep failing. This is the exact input to the
    guard's EXTERNAL_REDEPLOY verdict.
    """
    monkeypatch.delenv(PYTEST_SESSION_TOKEN_ENV, raising=False)
    clock_file = tmp_path / "last_redeploy_orchestrator.json"

    result = _run_script(tmp_path, scenario="fresh", clock_file=clock_file)

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    stamped = json.loads(clock_file.read_text())
    assert CLOCK_PROVENANCE_SESSION_KEY in stamped, (
        f"the key must be PRESENT and empty, not omitted; got {stamped!r}"
    )
    assert stamped[CLOCK_PROVENANCE_SESSION_KEY] == ""
    assert stamped[CLOCK_PROVENANCE_SOURCE_KEY] == _EXPECTED_SOURCE


def test_the_uuid_token_survives_the_stamp_byte_for_byte(tmp_path: Path) -> None:
    """The sanitiser must be a NO-OP for a real token.

    The script sanitises before interpolating (printf cannot escape JSON), and a
    sanitiser that mangled the ordinary case would break attribution silently:
    the guard compares tokens for equality, so a single dropped character turns
    "this run wrote it" into "some other session wrote it" — still a failure, but
    the wrong one, with a misleading message.
    """
    clock_file = tmp_path / "last_redeploy_orchestrator.json"
    ambient = os.environ[PYTEST_SESSION_TOKEN_ENV]
    assert re.fullmatch(r"[0-9a-f]{32}", ambient), (
        f"the session token should be a uuid4().hex; got {ambient!r}"
    )

    result = _run_script(tmp_path, scenario="fresh", clock_file=clock_file)

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    stamped = json.loads(clock_file.read_text())
    assert stamped[CLOCK_PROVENANCE_SESSION_KEY] == ambient


def test_the_stamp_is_still_well_formed_json_for_a_hostile_token(
    tmp_path: Path, monkeypatch
) -> None:
    """A corrupt clock body would be strictly WORSE than the bug being fixed.

    printf cannot escape JSON, so an env value carrying a quote, a backslash, a
    newline or a brace could otherwise produce a syntactically broken file --
    and _read_clock_epoch fails OPEN on a corrupt body, which would disarm the
    very min-interval cap this stamp exists to arm. The token's exact value is
    not asserted (it is unrepresentable here by construction), but its
    EMPTINESS is: sanitising away a token that was genuinely present would
    forgive the very write it proves, since empty is what
    df_pytest_isolation.py::deploy_clock_change_report reads as "no pytest
    session was an ancestor of this write". So the sanitiser drops to a
    non-empty sentinel rather than to nothing -- fail-closed in the same
    direction as every other unattributable shape.
    """
    monkeypatch.setenv(PYTEST_SESSION_TOKEN_ENV, 'ab"cd\\ef\ngh}ij')
    clock_file = tmp_path / "last_redeploy_orchestrator.json"

    result = _run_script(tmp_path, scenario="fresh", clock_file=clock_file)

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    stamped = json.loads(clock_file.read_text())
    assert isinstance(stamped["ts"], (int, float)), f"ts must stay numeric; got {stamped!r}"
    assert isinstance(stamped[CLOCK_PROVENANCE_SESSION_KEY], str)
    assert stamped[CLOCK_PROVENANCE_SESSION_KEY] != "", (
        "a non-empty ambient token sanitised down to the empty string, which "
        "the guard reads as a POSITIVE 'no pytest session wrote this' and "
        f"forgives; got {stamped!r}"
    )



# ---------------------------------------------------------------------------
# The in-flight fleet-redeploy lease (task 4755), EXIT-PATH lifecycle.
#
# The lease's whole value to its three readers -- the watchdog's staleness
# backstop, the merge-landed coordinator, and the watchdog's liveness probe --
# is that its ABSENCE is trustworthy: each stands down while it is there, so a
# lease that outlives its sweep suppresses every redeploy tier for up to the
# reader-side max-age bound.
#
# Every test here asserts the PAIR -- held mid-sweep, gone afterwards -- and
# never the second half alone. "The file is absent afterwards" is equally true
# of a script that never wrote one, so an absence-only test is green against
# the very code it is meant to be RED against. The mid-sweep half comes from
# the fake systemctl's lease log, which is written from inside the sweep.
#
# Paired with the clock assertions above rather than kept in the --drain suite,
# because the two files are written by the same exit paths in a FIXED order --
# clock first and only on verified success (I2), lease second and always --
# and that ordering is only visible when both are asserted on one run.
# ---------------------------------------------------------------------------


def _lease_log_path(tmp_path: Path) -> Path:
    return tmp_path / "lease_observations.log"


def _lease_observations(tmp_path: Path) -> list[tuple[str, dict]]:
    """Every (verb, lease body) the fake systemctl saw on disk, in call order."""
    log = _lease_log_path(tmp_path)
    if not log.exists():
        return []
    observations = []
    for line in log.read_text().splitlines():
        verb, _, body = line.partition("\t")
        observations.append((verb, json.loads(body)))
    return observations


def _assert_held_mid_sweep(tmp_path: Path, result) -> list[tuple[str, dict]]:
    """The sweep must have been holding a lease of its own while it ran."""
    observations = _lease_observations(tmp_path)
    assert observations, (
        f"no systemctl call observed a lease on disk -- the sweep must hold one "
        f"for its whole run, or nothing below distinguishes 'released' from "
        f"'never acquired'. stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    for verb, lease in observations:
        assert set(lease) == {"pid", "started_ts", "current_unit"}, (
            f"lease body seen at `{verb}` must carry exactly "
            f"pid/started_ts/current_unit; got {lease!r}"
        )
        assert isinstance(lease["pid"], int) and lease["pid"] > 0, (
            f"lease seen at `{verb}` must name a positive pid; got {lease!r}"
        )
    return observations


def test_verified_fresh_exit_stamps_the_clock_and_then_releases_the_lease(
    tmp_path: Path,
) -> None:
    """HAPPY PATH: lease held throughout, clock stamped, lease gone.

    The ordering is automatic and easy to misread, which is why it is pinned:
    stamp_fleet_deploy_clock runs, then `exit 0`, and only then does the EXIT
    trap release. Inverting it -- releasing before stamping -- would open a
    window in which no lease is held and no clock has been stamped yet, i.e.
    precisely the gap the two mechanisms exist between them to close.
    """
    clock_file = tmp_path / "last_redeploy_orchestrator.json"
    lease_file = tmp_path / "fleet_redeploy_lease.json"

    result = _run_script(
        tmp_path, scenario="fresh", clock_file=clock_file, lease_file=lease_file,
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    _assert_held_mid_sweep(tmp_path, result)
    assert clock_file.exists(), "the verified-fresh path must still stamp the clock"
    assert not lease_file.exists(), (
        f"the lease must be released on exit-0; it still holds "
        f"{lease_file.read_text()!r}"
    )


def test_failed_verify_exit_releases_the_lease_and_leaves_the_clock_alone(
    tmp_path: Path,
) -> None:
    """I2 UNCHANGED, lease still released: the two fail in opposite directions.

    A failed sweep must NOT stamp the clock -- stamping would silence the
    backstop for a full min-interval after a restart that never verified --
    and must nonetheless release the lease, because there is no longer a sweep
    in flight for anyone to stand down for. Asserting both on one run is what
    stops a later "release next to the stamp" tidy-up from coupling them.
    """
    clock_file = tmp_path / "last_redeploy_orchestrator.json"
    lease_file = tmp_path / "fleet_redeploy_lease.json"
    sentinel = '{"ts": 1.0}'
    clock_file.write_text(sentinel)

    result = _run_script(
        tmp_path, scenario="stale", clock_file=clock_file, lease_file=lease_file,
    )

    assert result.returncode == 1, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    _assert_held_mid_sweep(tmp_path, result)
    assert clock_file.read_text() == sentinel, "I2: a failed verify must not stamp"
    assert not lease_file.exists(), (
        f"the lease must be released on the verify-failure exit-1; it still "
        f"holds {lease_file.read_text()!r}"
    )


def test_no_running_units_exit_releases_the_lease(tmp_path: Path) -> None:
    """The early "nothing to restart" exit-0 must release too.

    Reached before the unit loop and before the stamp, so it is the path a
    hand-placed `lease_release` next to the other two would most plausibly
    miss -- and the sweep it ends did hold a lease for its whole (very short)
    life, which the single `list-units` observation proves.
    """
    clock_file = tmp_path / "last_redeploy_orchestrator.json"
    lease_file = tmp_path / "fleet_redeploy_lease.json"

    result = _run_script(
        tmp_path, scenario="fresh", clock_file=clock_file, lease_file=lease_file,
        no_units=True,
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert b"nothing to restart" in result.stdout, (
        f"expected the early no-units exit; got stdout={result.stdout!r}"
    )
    observations = _assert_held_mid_sweep(tmp_path, result)
    assert [verb for verb, _ in observations] == ["list-units"], (
        f"the only call on this path is the enumeration, and it must already "
        f"see the lease -- acquisition precedes enumeration; got {observations!r}"
    )
    assert not clock_file.exists(), (
        "I2: an empty fleet is not a verified redeploy and must not stamp"
    )
    assert not lease_file.exists(), (
        f"the lease must be released on the no-units exit-0; it still holds "
        f"{lease_file.read_text()!r}"
    )


def test_unexpected_argument_exit_writes_no_lease_at_all(tmp_path: Path) -> None:
    """A rejected argument must not even ACQUIRE -- acquisition follows parsing.

    A REGRESSION PIN, not a RED test, and deliberately so: "never acquired"
    and "acquired then released" are indistinguishable from out here, and both
    are indistinguishable from today's no-lease-at-all script. What it guards
    is the ORDER, which a concurrent reader CAN tell apart -- it would see a
    live lease for a sweep that never ran. The order itself is pinned
    structurally by scripts/tests/test_restart_all_orchestrators.py::
    test_every_exit_after_acquisition_is_covered_by_the_release_trap; this is
    the behavioural half of that pair.
    """
    clock_file = tmp_path / "last_redeploy_orchestrator.json"
    lease_file = tmp_path / "fleet_redeploy_lease.json"

    result = _run_script(
        tmp_path, scenario="fresh", clock_file=clock_file, lease_file=lease_file,
        extra_args=("--no-such-flag",),
    )

    assert result.returncode == 1, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert b"unexpected argument" in result.stderr, (
        f"expected the argument rejection; got stderr={result.stderr!r}"
    )
    assert not lease_file.exists(), (
        f"a rejected argument must leave no lease behind; got "
        f"{lease_file.read_text()!r}"
    )


def test_release_refuses_to_delete_a_lease_recorded_under_another_pid(
    tmp_path: Path,
) -> None:
    """A sweep must release only its OWN lease, never whoever holds it now.

    THE OVERLAP THIS CLOSES, which is why the task exists: the fixed
    transient-unit-name guard stops the staleness backstop running two sweeps
    at once, but the merge-landed coordinator uses a DIFFERENT transient unit
    name and is not covered by it. With an unconditional `rm -f` in the EXIT
    trap, the shorter of two overlapping sweeps deletes the longer one's lease
    on its way out -- re-arming all three readers while units are still being
    restarted, which is exactly the collision being prevented.

    The hand-over is staged mid-sweep rather than pre-seeded, because
    acquisition is unconditional: a lease seeded BEFORE the run is clobbered by
    this sweep's own `lease_acquire` and so proves nothing about the release.
    os.getpid() is a live pid that is definitively not the script's, so the
    handed-over lease also stays LIVE by the readers' pid-alive test.
    """
    clock_file = tmp_path / "last_redeploy_orchestrator.json"
    lease_file = tmp_path / "fleet_redeploy_lease.json"
    successor = {
        "pid": os.getpid(),
        "started_ts": 1.0,
        "current_unit": "orchestrator-fake-successor.service",
    }

    result = _run_script(
        tmp_path, scenario="fresh", clock_file=clock_file, lease_file=lease_file,
        lease_swap=successor,
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    observations = _assert_held_mid_sweep(tmp_path, result)
    own_pids = {lease["pid"] for _, lease in observations} - {successor["pid"]}
    assert own_pids, (
        f"this sweep must have held its OWN lease before the hand-over, or the "
        f"survival below is vacuous; got {observations!r}"
    )
    assert lease_file.exists(), (
        "the successor's lease must survive this sweep's release"
    )
    assert json.loads(lease_file.read_text()) == successor, (
        f"the successor's lease must be byte-equal to what it wrote; got "
        f"{lease_file.read_text()!r}"
    )
