"""Tests for scripts/check_memory_metadata_census_reporting.py, the predicate
that tells "the 05:00 census is producing readings" apart from every way it
can fail to (never installed, never ran, stale or failed last run, history
row stale behind a healthy unit).

The probe is driven through its real subprocess boundary. A fake
`systemctl` on PATH answers `show -p ...` from a JSON state file and
records every call, so which unit was probed is asserted, not assumed. The
history artifact lives under tmp_path, reached through the same REPO env
var the census wrapper reads. No test touches the live host or checkout.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "check_memory_metadata_census_reporting.py"
UNIT = "memory-metadata-coverage-census.service"
HISTORY_RELPATH = Path("plans") / "memory-metadata-coverage-history.json"
INSTALLER = "scripts/install-memory-metadata-coverage-census-timer.sh"


_FAKE_SYSTEMCTL_SRC = '''#!/usr/bin/env python3
"""Fake `systemctl`: records every call (minus `--user`) into
$FAKE_SYSTEMCTL_STATE and answers `show -p F1,F2 <unit>` from that file's
"fields" mapping. A field absent from the mapping answers as the empty
string, as real systemd does for a unit that never ran."""
import json
import os
import sys

STATE_PATH = os.environ["FAKE_SYSTEMCTL_STATE"]


def main(argv):
    args = [a for a in argv[1:] if a != "--user"]
    with open(STATE_PATH) as f:
        state = json.load(f)
    state["calls"].append(args)
    state["call_tz"].append(os.environ.get("TZ"))
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)

    if state.get("exit_code", 0) != 0:
        return state["exit_code"]
    if not args or args[0] != "show":
        return 1
    fields = ""
    rest = args[1:]
    for i, tok in enumerate(rest):
        if tok == "-p":
            fields = rest[i + 1]
        elif tok.startswith("--property="):
            fields = tok.split("=", 1)[1]
    current = state["fields"]
    for key in fields.split(","):
        print(f"{key}={current.get(key, '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''


def _fake_bins(tmp_path, *, fields, exit_code=0):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake_systemctl = bin_dir / "systemctl"
    fake_systemctl.write_text(_FAKE_SYSTEMCTL_SRC)
    fake_systemctl.chmod(0o755)
    state_path = tmp_path / "systemctl_state.json"
    state_path.write_text(json.dumps(
        {"calls": [], "call_tz": [], "fields": fields, "exit_code": exit_code}
    ))
    return bin_dir, state_path


def _systemctl_state(tmp_path):
    return json.loads((tmp_path / "systemctl_state.json").read_text())


def _systemd_timestamp(dt):
    return dt.strftime("%a %Y-%m-%d %H:%M:%S UTC")


def _hours_ago(hours):
    return datetime.now(UTC) - timedelta(hours=hours)


def _history_with_newest_stamp(dt):
    return {
        "schema_version": 1,
        "runs": [{"stamp": dt.replace(microsecond=0).isoformat()}],
    }


def _healthy_fields(**overrides):
    fields = {
        "LoadState": "loaded",
        "Result": "success",
        "ExecMainStatus": "0",
        "ExecMainExitTimestamp": _systemd_timestamp(_hours_ago(2)),
    }
    fields.update(overrides)
    return fields


def _write_history(tmp_path, history):
    if history is None:
        return
    path = tmp_path / HISTORY_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(history if isinstance(history, str) else json.dumps(history))


def _run(tmp_path, *argv, fields, history, exit_code=0, path_dirs=None):
    bin_dir, state_path = _fake_bins(tmp_path, fields=fields, exit_code=exit_code)
    _write_history(tmp_path, history)
    env = dict(os.environ)
    env["PATH"] = (
        os.pathsep.join(str(d) for d in path_dirs)
        if path_dirs is not None
        else f"{bin_dir}{os.pathsep}{env['PATH']}"
    )
    env["FAKE_SYSTEMCTL_STATE"] = str(state_path)
    env["REPO"] = str(tmp_path)
    return subprocess.run(
        [str(SCRIPT), *argv],
        env=env, capture_output=True, text=True, timeout=30,
    )


def _last_line(text):
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} must exist and be executable"


def test_healthy_census_exits_zero_and_names_unit_and_age(tmp_path):
    result = _run(
        tmp_path, "48",
        fields=_healthy_fields(),
        history=_history_with_newest_stamp(_hours_ago(2)),
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    calls = _systemctl_state(tmp_path)["calls"]
    assert any(c[0] == "show" and UNIT in c for c in calls), calls
    last = _last_line(result.stdout)
    assert UNIT in last, last
    assert re.search(r"\b2\.0h\b", last), last


def _never_ran_fields():
    fields = _healthy_fields()
    del fields["ExecMainExitTimestamp"]
    return fields


def test_not_found_load_state_is_not_installed_and_names_installer(tmp_path):
    result = _run(
        tmp_path, "48",
        fields=_healthy_fields(LoadState="not-found"),
        history=_history_with_newest_stamp(_hours_ago(2)),
    )

    assert result.returncode != 0, result.stdout
    assert "not installed" in result.stderr.lower(), result.stderr
    assert "not-found" in result.stderr, result.stderr
    assert INSTALLER in result.stderr, result.stderr


def test_absent_load_state_is_not_installed(tmp_path):
    fields = _healthy_fields()
    del fields["LoadState"]
    result = _run(
        tmp_path, "48",
        fields=fields,
        history=_history_with_newest_stamp(_hours_ago(2)),
    )

    assert result.returncode != 0, result.stdout
    assert "not installed" in result.stderr.lower(), result.stderr


def test_not_installed_verdict_is_reached_without_reading_history(tmp_path):
    result = _run(
        tmp_path, "48",
        fields=_healthy_fields(LoadState="not-found"),
        history=None,
    )

    assert result.returncode != 0, result.stdout
    assert "not installed" in result.stderr.lower(), result.stderr
    assert HISTORY_RELPATH.name not in result.stderr, result.stderr


def test_not_installed_and_never_ran_are_told_apart(tmp_path):
    fresh_history = _history_with_newest_stamp(_hours_ago(2))
    not_installed_dir = tmp_path / "not_installed"
    never_ran_dir = tmp_path / "never_ran"
    not_installed_dir.mkdir()
    never_ran_dir.mkdir()

    not_installed = _run(
        not_installed_dir, "48",
        fields=_healthy_fields(LoadState="not-found"), history=fresh_history,
    )
    never_ran = _run(
        never_ran_dir, "48", fields=_never_ran_fields(), history=fresh_history,
    )

    assert "not installed" in not_installed.stderr.lower(), not_installed.stderr
    assert not_installed.stderr != never_ran.stderr


def _assert_named_failure(result, *fragments):
    assert result.returncode != 0, result.stdout
    assert "Traceback" not in result.stderr, result.stderr
    for fragment in fragments:
        assert fragment in result.stderr, (fragment, result.stderr)


def test_empty_exit_timestamp_is_never_ran(tmp_path):
    result = _run(
        tmp_path, "48",
        fields=_never_ran_fields(),
        history=_history_with_newest_stamp(_hours_ago(2)),
    )

    _assert_named_failure(result, "never run")
    assert "not installed" not in result.stderr.lower(), result.stderr


def test_na_exit_timestamp_is_never_ran(tmp_path):
    result = _run(
        tmp_path, "48",
        fields=_healthy_fields(ExecMainExitTimestamp="n/a"),
        history=_history_with_newest_stamp(_hours_ago(2)),
    )

    _assert_named_failure(result, "never run")


def test_last_run_older_than_window_is_stale(tmp_path):
    result = _run(
        tmp_path, "48",
        fields=_healthy_fields(
            ExecMainExitTimestamp=_systemd_timestamp(_hours_ago(72))
        ),
        history=_history_with_newest_stamp(_hours_ago(2)),
    )

    _assert_named_failure(result, "72.0h", "48h")


def test_last_run_just_inside_window_passes(tmp_path):
    result = _run(
        tmp_path, "48",
        fields=_healthy_fields(
            ExecMainExitTimestamp=_systemd_timestamp(_hours_ago(47))
        ),
        history=_history_with_newest_stamp(_hours_ago(2)),
    )

    assert result.returncode == 0, result.stderr


def test_last_run_just_outside_window_fails(tmp_path):
    result = _run(
        tmp_path, "48",
        fields=_healthy_fields(
            ExecMainExitTimestamp=_systemd_timestamp(_hours_ago(49))
        ),
        history=_history_with_newest_stamp(_hours_ago(2)),
    )

    _assert_named_failure(result, "49.0h", "48h")


def test_unparsable_exit_timestamp_fails_loud(tmp_path):
    result = _run(
        tmp_path, "48",
        fields=_healthy_fields(ExecMainExitTimestamp="garbage"),
        history=_history_with_newest_stamp(_hours_ago(2)),
    )

    _assert_named_failure(result, "'garbage'")
    assert "never run" not in result.stderr, result.stderr


def test_local_zone_exit_timestamp_fails_loud_rather_than_misdating(tmp_path):
    local = _hours_ago(2).strftime("%a %Y-%m-%d %H:%M:%S BST")
    result = _run(
        tmp_path, "48",
        fields=_healthy_fields(ExecMainExitTimestamp=local),
        history=_history_with_newest_stamp(_hours_ago(2)),
    )

    _assert_named_failure(result, "BST")


def test_unit_state_is_read_with_timestamps_pinned_to_utc(tmp_path):
    _run(
        tmp_path, "48",
        fields=_healthy_fields(),
        history=_history_with_newest_stamp(_hours_ago(2)),
    )

    assert _systemctl_state(tmp_path)["call_tz"] == ["UTC"]


def test_stale_history_row_behind_healthy_unit_fails(tmp_path):
    result = _run(
        tmp_path, "48",
        fields=_healthy_fields(),
        history=_history_with_newest_stamp(_hours_ago(25 * 24)),
    )

    _assert_named_failure(result, "600.0h", "48h", str(tmp_path / HISTORY_RELPATH))


@pytest.mark.parametrize(
    ("history", "fragment"),
    [
        (None, "does not exist"),
        ("{not json", "not valid JSON"),
        ({"schema_version": 1}, "no 'runs' list"),
        ({"schema_version": 1, "runs": {}}, "no 'runs' list"),
        ({"schema_version": 1, "runs": []}, "'runs' is empty"),
        ({"schema_version": 1, "runs": [{"census": 1}]}, "no 'stamp'"),
        ({"schema_version": 1, "runs": [{"stamp": "garbage"}]}, "'garbage'"),
        ({"schema_version": 1, "runs": [{"stamp": 123}]}, "123"),
        (
            {"schema_version": 1, "runs": [{"stamp": "2026-09-24T05:00:00"}]},
            "no UTC offset",
        ),
    ],
    ids=[
        "absent", "invalid-json", "runs-missing", "runs-not-a-list",
        "runs-empty", "stamp-missing", "stamp-unparsable", "stamp-not-a-string",
        "stamp-naive",
    ],
)
def test_unreadable_history_fails_loud_naming_path(tmp_path, history, fragment):
    result = _run(tmp_path, "48", fields=_healthy_fields(), history=history)

    _assert_named_failure(result, fragment, str(tmp_path / HISTORY_RELPATH))


def test_not_installed_outranks_a_stale_history_row(tmp_path):
    result = _run(
        tmp_path, "48",
        fields=_healthy_fields(LoadState="not-found"),
        history=_history_with_newest_stamp(_hours_ago(25 * 24)),
    )

    _assert_named_failure(result, INSTALLER)
    assert "not installed" in result.stderr.lower(), result.stderr
    assert HISTORY_RELPATH.name not in result.stderr, result.stderr
