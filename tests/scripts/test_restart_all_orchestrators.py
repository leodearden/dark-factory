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
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
SCRIPT = REPO_ROOT / "scripts" / "restart-all-orchestrators.sh"
UNIT_NAME = "orchestrator-fake.service"

# Stateful fake `systemctl`: `list-units` reports one fake orchestrator unit;
# `show -p <fields>` reports a baseline MainPID/ActiveState/
# ActiveEnterTimestamp(Monotonic) until the marker file `restart` touches
# exists, at which point -- scenario "fresh" only -- it reports a fresh,
# higher monotonic timestamp (a verified restart). Scenario "stale" never
# advances the monotonic timestamp even after `restart`, simulating a
# restart that never actually came back up fresh (the NEGATIVE/I2 case).
_FAKE_SYSTEMCTL = textwrap.dedent("""\
    #!/usr/bin/env bash
    set -euo pipefail

    MARKER="${FAKE_SYSTEMCTL_MARKER:?FAKE_SYSTEMCTL_MARKER not set}"
    SCENARIO="${FAKE_SYSTEMCTL_SCENARIO:-fresh}"
    UNIT_NAME="${FAKE_SYSTEMCTL_UNIT:?FAKE_SYSTEMCTL_UNIT not set}"

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
            ;;
        show)
            if [[ -f "$MARKER" && "$SCENARIO" == "fresh" ]]; then
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
) -> subprocess.CompletedProcess[bytes]:
    bin_dir = tmp_path / "bin"
    _make_fake_systemctl(bin_dir)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["FAKE_SYSTEMCTL_MARKER"] = str(tmp_path / "restarted.marker")
    env["FAKE_SYSTEMCTL_SCENARIO"] = scenario
    env["FAKE_SYSTEMCTL_UNIT"] = UNIT_NAME
    env["ORCH_FLEET_DEPLOY_CLOCK"] = str(clock_file)
    env["RESTART_VERIFY_TIMEOUT"] = verify_timeout

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
