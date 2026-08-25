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

from df_pytest_isolation import assert_synthetic_units  # noqa: E402

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

    case "$verb" in
        list-units)
            echo "${UNIT_NAME} loaded active running Orchestrator"
            ;;
        restart)
            touch "$MARKER"
            rm -f "$COUNTER"
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

    return subprocess.run(
        [str(SCRIPT)],
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
