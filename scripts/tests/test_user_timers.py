"""scripts/user_timers.py, driven through its CLI against a PATH-shimmed fake `systemctl`."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest
from cli_subprocess_timeout import cli_timeout_from_env
from expected_user_timers import EXPECTED_TIMERS

SCRIPT = Path(__file__).resolve().parents[1] / 'user_timers.py'
CLI_TIMEOUT_SECS = cli_timeout_from_env('USER_TIMERS_CLI_TIMEOUT_SECS')
TAG = '[user_timers] '
PROJECT = 'dark_factory'
ARMED = {'LoadState': 'loaded', 'UnitFileState': 'enabled', 'ActiveState': 'active'}
ARMING_VERDICTS = frozenset({
    'armed', 'not-installed', 'not-enabled', 'not-active', 'masked', 'not-loaded', 'unreadable',
})
SHOW_FIELDS = ('Id', 'LoadState', 'UnitFileState', 'ActiveState')
SYSTEMCTL_ERROR = 'Failed to connect to user scope bus via local transport: No such file or directory'

CENSUS = 'memory-metadata-coverage-census.timer'
TRICKLE = f'legibility-trickle@{PROJECT}.timer'
RETURN_BRIEF = 'return-brief.timer'

_FAKE_SYSTEMCTL_SRC = r'''#!/usr/bin/env python3
"""Fake `systemctl --user` answering `show` per unit and flipping state on `enable --now`.

Every call minus `--user` is recorded into $FAKE_SYSTEMCTL_STATE. A unit absent
from state["units"] answers the shape a missing unit has on a real host.
"""
import json
import os
import sys

STATE_PATH = os.environ["FAKE_SYSTEMCTL_STATE"]
NOT_FOUND = {"LoadState": "not-found", "ActiveState": "inactive", "UnitFileState": ""}
ARMED = {"LoadState": "loaded", "UnitFileState": "enabled", "ActiveState": "active"}


def _show(state, rest):
    if state.get("show_exit"):
        print(state["show_error"], file=sys.stderr)
        return state["show_exit"]
    flag = rest.index("-p")
    fields = rest[flag + 1].split(",")
    units = [arg for i, arg in enumerate(rest) if i not in (flag, flag + 1)]
    blocks = []
    for unit in units:
        values = state["units"].get(unit, NOT_FOUND)
        lines = [f"Id={unit}"] + [f"{field}={values.get(field, '')}" for field in fields if field != "Id"]
        blocks.append("\n".join(lines))
    print("\n\n".join(blocks))
    return 0


def _enable_now(state, rest):
    if rest[:1] != ["--now"]:
        return 1
    for unit in rest[1:]:
        state["units"][unit] = dict(ARMED)
    return 0


def main(argv):
    if "--user" not in argv:
        print("fake systemctl: refusing a call without --user", file=sys.stderr)
        return 1
    args = [a for a in argv[1:] if a != "--user"]
    with open(STATE_PATH) as f:
        state = json.load(f)
    state["calls"].append(args)
    verbs = {"show": _show, "enable": _enable_now}
    rc = verbs[args[0]](state, args[1:]) if args and args[0] in verbs else 1
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''


@dataclass(frozen=True)
class TimersRun:
    result: subprocess.CompletedProcess[str]
    systemctl_calls: tuple[tuple[str, ...], ...]

    @property
    def output(self) -> str:
        return self.result.stdout + self.result.stderr

    @property
    def show_calls(self) -> list[tuple[str, ...]]:
        return [call for call in self.systemctl_calls if call[:1] == ('show',)]


def resolved_units(project_id: str = PROJECT) -> list[str]:
    return [timer.unit_name(project_id) for timer in EXPECTED_TIMERS]


def host_units(
    overrides: Mapping[str, Mapping[str, str] | None] | None = None, project_id: str = PROJECT,
) -> dict[str, dict[str, str]]:
    """Every expected unit armed, then *overrides* applied; a None override removes the unit."""
    units = {name: dict(ARMED) for name in resolved_units(project_id)}
    for name, fields in (overrides or {}).items():
        if fields is None:
            units.pop(name)
        else:
            units[name] = dict(fields)
    return units


def install_fake_systemctl(tmp_path: Path, units: Mapping[str, Mapping[str, str]], show_exit: int = 0) -> dict[str, str]:
    """Write the fake onto a tmp bin/ and return the environment that puts it first on PATH."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / 'systemctl'
    fake.write_text(_FAKE_SYSTEMCTL_SRC)
    fake.chmod(0o755)
    state_path = tmp_path / 'systemctl_state.json'
    state_path.write_text(json.dumps({
        'calls': [], 'units': dict(units), 'show_exit': show_exit, 'show_error': SYSTEMCTL_ERROR,
    }))
    return {
        **os.environ,
        'PATH': f'{bin_dir}{os.pathsep}{os.environ["PATH"]}',
        'FAKE_SYSTEMCTL_STATE': str(state_path),
    }


def run_timers(tmp_path: Path, env: Mapping[str, str], *args: str) -> TimersRun:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        env=dict(env), capture_output=True, text=True, timeout=CLI_TIMEOUT_SECS,
    )
    calls = json.loads((tmp_path / 'systemctl_state.json').read_text())['calls']
    return TimersRun(result=result, systemctl_calls=tuple(tuple(call) for call in calls))


def check(
    tmp_path: Path,
    overrides: Mapping[str, Mapping[str, str] | None] | None = None,
    *,
    root_name: str = 'dark-factory',
    project_id: str = PROJECT,
    show_exit: int = 0,
) -> TimersRun:
    root = tmp_path / root_name
    root.mkdir(exist_ok=True)
    env = install_fake_systemctl(tmp_path, host_units(overrides, project_id), show_exit)
    return run_timers(tmp_path, env, 'check', '--repo-root', str(root))


def untagged_lines(output: str) -> list[str]:
    """Each line with its tag removed, asserting every non-empty line carries the tag."""
    lines = [line for line in output.splitlines() if line]
    stray = [line for line in lines if not line.startswith(TAG)]
    assert not stray, f'lines without the {TAG!r} tag: {stray}'
    return [line.removeprefix(TAG) for line in lines]


def final_verdicts(output: str) -> dict[str, str]:
    """unit -> its report line, for every line opening with an arming verdict."""
    verdicts = {}
    for line in untagged_lines(output):
        verdict, _, rest = line.partition('] ')
        if verdict.startswith('[') and verdict[1:] in ARMING_VERDICTS and rest:
            verdicts[rest.split()[0].rstrip(':')] = line
    return verdicts


def assert_verdict(output: str, unit: str, verdict: str) -> str:
    line = final_verdicts(output)[unit]
    assert line.startswith(f'[{verdict}] {unit}'), line
    return line


def assert_others_armed(output: str, *non_armed: str) -> None:
    verdicts = final_verdicts(output)
    assert set(verdicts) == set(resolved_units()), sorted(set(verdicts) ^ set(resolved_units()))
    for unit in set(verdicts) - set(non_armed):
        assert verdicts[unit] == f'[armed] {unit}'


def test_all_armed_exits_zero_listing_each_unit_and_a_count(tmp_path: Path) -> None:
    run = check(tmp_path)
    assert run.result.returncode == 0, run.output
    assert_others_armed(run.output)
    count = len(EXPECTED_TIMERS)
    assert f'{count} of {count} expected user timers armed' in untagged_lines(run.output)[-1]


def test_one_show_call_reads_every_resolved_unit_in_registry_order(tmp_path: Path) -> None:
    run = check(tmp_path)
    assert len(run.show_calls) == 1, run.systemctl_calls
    (call,) = run.show_calls
    flag = call.index('-p')
    assert set(SHOW_FIELDS) <= set(call[flag + 1].split(','))
    units = [arg for i, arg in enumerate(call[1:], start=1) if i not in (flag, flag + 1)]
    assert units == resolved_units()
    assert not [unit for unit in units if unit.endswith('@.timer')]
    untagged_lines(run.output)


def test_template_instances_derive_from_the_repo_roots_basename(tmp_path: Path) -> None:
    run = check(tmp_path, root_name='reify', project_id='reify')
    assert run.result.returncode == 0, run.output
    (call,) = run.show_calls
    assert 'legibility-trickle@reify.timer' in call
    assert 'quality-snapshot@reify.timer' in call
    assert TRICKLE not in call
    untagged_lines(run.output)


def test_a_missing_unit_is_not_installed_and_names_its_installer(tmp_path: Path) -> None:
    run = check(tmp_path, {CENSUS: None})
    assert run.result.returncode == 1, run.output
    line = assert_verdict(run.output, CENSUS, 'not-installed')
    assert 'scripts/install-memory-metadata-coverage-census-timer.sh' in line
    assert_others_armed(run.output, CENSUS)
    count = len(EXPECTED_TIMERS)
    summary = untagged_lines(run.output)[-1]
    assert f'{count - 1} of {count} expected user timers armed' in summary
    assert 'scripts/user_timers.py arm' in summary


def test_an_installed_but_disabled_instance_is_not_enabled(tmp_path: Path) -> None:
    run = check(tmp_path, {TRICKLE: {'LoadState': 'loaded', 'UnitFileState': 'disabled', 'ActiveState': 'inactive'}})
    assert run.result.returncode == 1, run.output
    line = assert_verdict(run.output, TRICKLE, 'not-enabled')
    assert f'scripts/legibility/install-trickle-timer.sh {PROJECT}' in line
    assert_others_armed(run.output, TRICKLE)


def test_an_enabled_but_inactive_timer_is_not_active(tmp_path: Path) -> None:
    run = check(tmp_path, {RETURN_BRIEF: {'LoadState': 'loaded', 'UnitFileState': 'enabled', 'ActiveState': 'inactive'}})
    assert run.result.returncode == 1, run.output
    line = assert_verdict(run.output, RETURN_BRIEF, 'not-active')
    assert 'scripts/install-return-brief-timer.sh' in line
    assert_others_armed(run.output, RETURN_BRIEF)


def test_a_masked_timer_names_the_unmask(tmp_path: Path) -> None:
    run = check(tmp_path, {CENSUS: {'LoadState': 'masked', 'UnitFileState': 'masked', 'ActiveState': 'inactive'}})
    assert run.result.returncode == 1, run.output
    line = assert_verdict(run.output, CENSUS, 'masked')
    assert f'systemctl --user unmask {CENSUS}' in line
    assert_others_armed(run.output, CENSUS)


def test_a_rejected_unit_file_is_not_loaded_and_names_status(tmp_path: Path) -> None:
    run = check(tmp_path, {CENSUS: {'LoadState': 'bad-setting', 'UnitFileState': 'enabled', 'ActiveState': 'inactive'}})
    assert run.result.returncode == 1, run.output
    line = assert_verdict(run.output, CENSUS, 'not-loaded')
    assert f'systemctl --user status {CENSUS}' in line
    assert_others_armed(run.output, CENSUS)


def test_a_failed_show_makes_every_unit_unreadable_and_reports_why(tmp_path: Path) -> None:
    run = check(tmp_path, show_exit=1)
    assert run.result.returncode == 1, run.output
    verdicts = final_verdicts(run.output)
    assert set(verdicts) == set(resolved_units())
    for unit in verdicts:
        assert_verdict(run.output, unit, 'unreadable')
    assert SYSTEMCTL_ERROR in run.output


@pytest.mark.parametrize('args', [(), ('check', '--bogus')], ids=['no-subcommand', 'unknown-flag'])
def test_a_usage_error_exits_two_untagged(tmp_path: Path, args: tuple[str, ...]) -> None:
    env = install_fake_systemctl(tmp_path, host_units())
    run = run_timers(tmp_path, env, *args)
    assert run.result.returncode == 2, run.output
    assert '[user_timers]' not in run.output


def test_the_script_is_executable() -> None:
    assert os.access(SCRIPT, os.X_OK)

