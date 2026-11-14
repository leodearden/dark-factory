"""PATH-shimmed fake `systemctl` and `git` for the systemd unit-liveness probes
(scripts/legibility/check_*_liveness.sh).

The fake `systemctl` answers `show -p FIELD1,FIELD2,... <unit>` from a JSON
state file and records every call; the fake `git` only drops a marker, so
"never consults git" is an assertion rather than an assumption.
ExecMainExitTimestamp values come from :func:`systemd_timestamp`, in
systemd's own format, which the probe's real `date -d` parses natively, so
the age math runs for real without a faked `date`.
"""
from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_FAKE_SYSTEMCTL_SRC = '''#!/usr/bin/env python3
"""Fake `systemctl` for the systemd unit-liveness probes.

Records every invocation (minus `--user`) into a JSON state file at
$FAKE_SYSTEMCTL_STATE, and answers `show -p FIELD1,FIELD2,...  <unit>`
(or `--property=FIELD1,...`; only the last such flag counts) from that
file's "fields" mapping. A requested field absent from "fields" answers
as an empty string. Real systemd answers a never-run unit differently:
Result=success and ExecMainStatus=0, with only ExecMainExitTimestamp
empty -- the shape the never-ran guard's comment in
scripts/legibility/check_trickle_health_liveness.sh records -- so a
never-ran fixture must spell that shape out rather than omit fields.
"""
import json
import os
import sys

STATE_PATH = os.environ["FAKE_SYSTEMCTL_STATE"]


def _load():
    with open(STATE_PATH) as f:
        return json.load(f)


def _save(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


def main(argv):
    args = [a for a in argv[1:] if a != "--user"]
    if not args:
        return 1
    verb, rest = args[0], args[1:]

    state = _load()
    state.setdefault("calls", []).append(args)
    _save(state)

    if verb == "show":
        fields = None
        i = 0
        while i < len(rest):
            tok = rest[i]
            if tok == "-p":
                fields = rest[i + 1]
                i += 2
            elif tok.startswith("--property="):
                fields = tok.split("=", 1)[1]
                i += 1
            else:
                i += 1  # unit name or unrecognized flag

        current = state.get("fields", {})
        keys = fields.split(",") if fields else list(current.keys())
        for k in keys:
            print(f"{k}={current.get(k, '')}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''

_FAKE_GIT_SRC = '''#!/usr/bin/env bash
# Fake `git` for the systemd unit-liveness probes -- decision 7 forbids
# consulting git for liveness (quiet nights commit nothing, so a git-history
# probe would false-alarm on a healthy but quiet timer). Any invocation drops
# a marker file so the test can assert git was NEVER called.
set -euo pipefail
: > "$FAKE_GIT_CALLED_MARKER"
exit 0
'''


def systemd_timestamp(dt: datetime) -> str:
    """Format a UTC datetime the way `systemctl show` emits a timestamp
    property, e.g. 'Tue 2026-07-14 03:00:12 UTC'."""
    return dt.strftime('%a %Y-%m-%d %H:%M:%S UTC')


@dataclass(frozen=True)
class UnitProbeRun:
    """One probe run: its process result, every `systemctl` call it made
    (minus `--user`), and whether it ever invoked `git`."""

    result: subprocess.CompletedProcess[str]
    systemctl_calls: tuple[tuple[str, ...], ...]
    git_called: bool


def _install_fake(bin_dir: Path, name: str, source: str) -> None:
    fake = bin_dir / name
    fake.write_text(source)
    fake.chmod(0o755)


def run_unit_probe(
    script: Path, tmp_path: Path, *args: str, fields: Mapping[str, str],
) -> UnitProbeRun:
    """Run ``bash <script> <args...>`` with the fakes first on PATH, the fake
    `systemctl show` answering from *fields*."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    _install_fake(bin_dir, 'systemctl', _FAKE_SYSTEMCTL_SRC)
    _install_fake(bin_dir, 'git', _FAKE_GIT_SRC)

    state_path = tmp_path / 'systemctl_state.json'
    state_path.write_text(json.dumps({'calls': [], 'fields': dict(fields)}))
    git_marker_path = tmp_path / 'git_was_called'

    env = dict(os.environ)
    env['PATH'] = f'{bin_dir}{os.pathsep}{env["PATH"]}'
    env['FAKE_SYSTEMCTL_STATE'] = str(state_path)
    env['FAKE_GIT_CALLED_MARKER'] = str(git_marker_path)
    result = subprocess.run(
        ['bash', str(script), *args],
        env=env, capture_output=True, text=True, timeout=30,
    )
    calls = json.loads(state_path.read_text())['calls']
    return UnitProbeRun(
        result=result,
        systemctl_calls=tuple(tuple(call) for call in calls),
        git_called=git_marker_path.exists(),
    )
