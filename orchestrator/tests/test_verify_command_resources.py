"""Tests for orchestrator.verify_command_resources — the C4 per-run verify record.

PRD ``plans/per-consumer-resource-recording-prd.md`` C4 and leaf 4. The pure
surface (record, invariants, the reading map, the capture decision) is tested
over hand-filled capture directories holding kernel-format fixture files.
The in-scope script (``CommandCapture.wrap``) is run for real, unscoped, against
a fake cgroup root that mirrors this process's own unified cgroup path. The PRD
section 7 boundary rows run ``verify.py::_run_cmd`` under real
``systemd-run --user --scope``, and skip on a host without it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from _fake_systemd_run import fake_systemd_run as fake_systemd_run
from _fake_systemd_run import recorded_scope_spawns
from _real_project_slice import RealProjectSlice
from _real_project_slice import real_project_slice as real_project_slice
from shared.cgroup_stats import (
    CPU_STAT,
    MEMORY_EVENTS,
    MEMORY_PEAK,
    SWAP_PEAK,
    CgroupReading,
    PressureResource,
    read_cgroup,
)
from shared.proc_group import unified_cgroup_path
from shared.verify_scope_tag import parse_verify_scope_unit, scope_tag_for

from orchestrator.config import FusedMemoryConfig, ModuleConfig, OrchestratorConfig
from orchestrator.scope_launch_handshake import ScopeLaunchHandshake
from orchestrator.verify import _run_cmd, run_verification
from orchestrator.verify_command_resources import (
    Capture,
    CommandCapture,
    CommandResources,
    allocate_command_capture,
    resources_from_reading,
)

UNREACHED_COMMAND_TIMEOUT_SECS = 120
SYSTEMD_CALL_TIMEOUT_SECS = 60

LAUNCHED = 'df-verify-proj-0123abcd-0123456789ab.scope'
LAUNCH_CGROUP = f'/user.slice/user-1000.slice/user@1000.service/df.slice/{LAUNCHED}'

CPU_STAT_TEXT = (
    'usage_usec 4200000\n'
    'user_usec 3100000\n'
    'system_usec 1100000\n'
    'nr_periods 0\n'
    'nr_throttled 0\n'
    'throttled_usec 0\n'
)
MEMORY_EVENTS_TEXT = 'low 0\nhigh 7\nmax 2\noom 1\noom_kill 1\noom_group_kill 0\n'
CPU_PRESSURE_TEXT = (
    'some avg10=0.00 avg60=0.00 avg300=0.00 total=11\n'
    'full avg10=0.00 avg60=0.00 avg300=0.00 total=12\n'
)
MEMORY_PRESSURE_TEXT = (
    'some avg10=0.00 avg60=0.00 avg300=0.00 total=21\n'
    'full avg10=0.00 avg60=0.00 avg300=0.00 total=22\n'
)
IO_PRESSURE_TEXT = (
    'some avg10=0.00 avg60=0.00 avg300=0.00 total=31\n'
    'full avg10=0.00 avg60=0.00 avg300=0.00 total=32\n'
)

SCOPE_FILES = {
    'cpu.stat': CPU_STAT_TEXT,
    'memory.peak': '209715200\n',
    'memory.swap.peak': '4096\n',
    'memory.events': MEMORY_EVENTS_TEXT,
    'cpu.pressure': CPU_PRESSURE_TEXT,
    'memory.pressure': MEMORY_PRESSURE_TEXT,
    'io.pressure': IO_PRESSURE_TEXT,
}

IDENTITY = {
    'project': 'dark_factory',
    'role': 'task',
    'task_id': '6355',
    'attempt': 2,
    'module': 'orchestrator',
    'leg': 'test',
    'segment': None,
    'plan_id': 'f' * 32,
    'started_at': '2026-11-27T05:00:00+00:00',
    'ended_at': '2026-11-27T05:01:00+00:00',
}

MEASUREMENT_FIELDS = (
    'cpu_usec',
    'user_usec',
    'system_usec',
    'memory_peak_bytes',
    'swap_peak_bytes',
    'memory_high_events',
    'memory_max_events',
    'oom_kills',
    'cpu_stall_some_usec',
    'memory_stall_some_usec',
    'memory_stall_full_usec',
    'io_stall_some_usec',
)


def _write(directory: Path, files: dict[str, str]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (directory / name).write_text(text)
    return directory


def _exit_capture_dir(tmp_path: Path, files: dict[str, str] | None = None) -> Path:
    return _write(
        tmp_path / 'capture',
        {
            **(SCOPE_FILES if files is None else files),
            'cgroup': f'0::{LAUNCH_CGROUP}\n',
            'complete': '',
        },
    )


def _launched_capture(directory: Path, **kwargs) -> CommandCapture:
    capture = CommandCapture(directory, **kwargs)
    capture.launched(LAUNCHED)
    return capture


def _exit_record(**overrides) -> CommandResources:
    fields = {
        'capture': Capture.EXIT,
        'capture_reason': 'read at exit',
        'relocated_scopes': frozenset(),
        'cpu_usec': 1,
        'user_usec': 1,
        'system_usec': 0,
        'memory_peak_bytes': 4096,
        'swap_peak_bytes': 0,
        'memory_high_events': 0,
        'memory_max_events': 0,
        'oom_kills': 0,
        'cpu_stall_some_usec': 0,
        'memory_stall_some_usec': 0,
        'memory_stall_full_usec': 0,
        'io_stall_some_usec': 0,
        'scope_unit': LAUNCHED,
        'host': 'host',
        **IDENTITY,
    }
    return CommandResources(**{**fields, **overrides})


class TestCapture:
    def test_the_four_states(self):
        assert {c.value for c in Capture} == {
            'exit',
            'before_kill',
            'relocated',
            'unavailable',
        }
        assert Capture('before_kill') is Capture.BEFORE_KILL

    def test_the_census_copy_of_the_states_matches(self):
        """scripts/verify_budget_census.py imports no orchestrator code, so it
        restates the states; a state it does not know reads as malformed."""
        census = _load_script('verify_budget_census')

        assert set(census.C4_CAPTURE_STATES) == {c.value for c in Capture}


def _load_script(name: str) -> Any:
    """A scripts/ module by path; registered so its dataclasses resolve."""
    path = Path(__file__).resolve().parents[2] / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(f'_scripts_{name}', path)
    assert spec is not None and spec.loader is not None, path
    module: Any = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
    return module


class TestCommandResourcesRecord:
    def test_to_dict_is_json_native_and_round_trips(self):
        record = _exit_record(
            capture=Capture.RELOCATED,
            relocated_scopes=frozenset({'run-b.scope', 'run-a.scope'}),
        )
        as_dict = record.to_dict()

        assert json.loads(json.dumps(as_dict)) == as_dict
        assert as_dict['capture'] == 'relocated'
        assert as_dict['relocated_scopes'] == ['run-a.scope', 'run-b.scope']
        assert CommandResources.from_dict(as_dict) == record

    def test_to_dict_carries_every_c4_field_flat(self):
        assert set(_exit_record().to_dict()) == {
            'capture',
            'capture_reason',
            'relocated_scopes',
            *MEASUREMENT_FIELDS,
            'scope_unit',
            'host',
            'project',
            'role',
            'task_id',
            'attempt',
            'module',
            'leg',
            'segment',
            'plan_id',
            'started_at',
            'ended_at',
        }

    def test_an_unavailable_record_round_trips(self):
        record = CommandResources(
            capture=Capture.UNAVAILABLE, capture_reason='no cgroup scope'
        )
        assert CommandResources.from_dict(record.to_dict()) == record

    def test_the_record_is_immutable(self):
        record = _exit_record()
        field = 'cpu_usec'
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(record, field, 2)


class TestCommandResourcesInvariants:
    def test_an_empty_reason_is_refused(self):
        with pytest.raises(ValueError, match='capture_reason'):
            _exit_record(capture_reason='')

    @pytest.mark.parametrize(
        'capture', [Capture.EXIT, Capture.BEFORE_KILL, Capture.RELOCATED]
    )
    def test_a_measured_capture_needs_cpu(self, capture):
        relocated = frozenset({'run-u1.scope'}) if capture is Capture.RELOCATED else frozenset()
        with pytest.raises(ValueError, match='cpu_usec'):
            _exit_record(capture=capture, relocated_scopes=relocated, cpu_usec=None)

    @pytest.mark.parametrize('field', MEASUREMENT_FIELDS)
    def test_an_unavailable_capture_carries_no_measurement(self, field):
        unavailable = CommandResources(
            capture=Capture.UNAVAILABLE, capture_reason='no cgroup scope'
        )
        with pytest.raises(ValueError, match=field):
            CommandResources.from_dict({**unavailable.to_dict(), field: 5})

    def test_relocated_names_its_scopes(self):
        with pytest.raises(ValueError, match='relocated_scopes'):
            _exit_record(capture=Capture.RELOCATED, relocated_scopes=frozenset())

    @pytest.mark.parametrize('capture', [Capture.EXIT, Capture.BEFORE_KILL])
    def test_a_launch_scope_capture_names_no_relocated_scope(self, capture):
        with pytest.raises(ValueError, match='run-u1.scope'):
            _exit_record(capture=capture, relocated_scopes=frozenset({'run-u1.scope'}))

    @pytest.mark.parametrize('value', [-1, True, 1.5, '7'])
    def test_a_counter_must_be_a_non_negative_int(self, value):
        with pytest.raises(ValueError, match='memory_peak_bytes') as raised:
            _exit_record(memory_peak_bytes=value)
        assert repr(value) in str(raised.value)

    def test_from_dict_refuses_an_unknown_capture(self):
        as_dict = {**_exit_record().to_dict(), 'capture': 'sampled'}
        with pytest.raises(ValueError, match='sampled'):
            CommandResources.from_dict(as_dict)

    def test_from_dict_refuses_a_missing_key(self):
        as_dict = _exit_record().to_dict()
        del as_dict['oom_kills']
        with pytest.raises(ValueError, match='oom_kills'):
            CommandResources.from_dict(as_dict)

    def test_from_dict_enforces_the_constructor_invariants(self):
        as_dict = {**_exit_record().to_dict(), 'cpu_usec': -4}
        with pytest.raises(ValueError, match='cpu_usec'):
            CommandResources.from_dict(as_dict)


def _reading(tmp_path: Path, files: dict[str, str]) -> CgroupReading:
    directory = _write(tmp_path / 'reading', files)
    reading = read_cgroup(directory, present=frozenset(files))
    assert isinstance(reading, CgroupReading)
    return reading


class TestResourcesFromReading:
    def test_maps_every_file_to_its_c4_field(self, tmp_path):
        record = resources_from_reading(
            _reading(tmp_path, SCOPE_FILES),
            capture=Capture.EXIT,
            reason='read at exit',
            relocated=frozenset(),
        )

        assert record.capture is Capture.EXIT
        assert record.capture_reason == 'read at exit'
        assert (record.cpu_usec, record.user_usec, record.system_usec) == (
            4200000,
            3100000,
            1100000,
        )
        assert record.memory_peak_bytes == 209715200
        assert record.swap_peak_bytes == 4096
        assert (record.memory_high_events, record.memory_max_events, record.oom_kills) == (
            7,
            2,
            1,
        )
        assert record.cpu_stall_some_usec == 11
        assert record.memory_stall_some_usec == 21
        assert record.memory_stall_full_usec == 22
        assert record.io_stall_some_usec == 31

    def test_a_file_not_in_the_reading_is_none_never_zero(self, tmp_path):
        record = resources_from_reading(
            _reading(tmp_path, {'cpu.stat': CPU_STAT_TEXT}),
            capture=Capture.EXIT,
            reason='read at exit',
            relocated=frozenset(),
        )

        assert record.cpu_usec == 4200000
        for field in MEASUREMENT_FIELDS[3:]:
            assert getattr(record, field) is None, field

    def test_relocated_scopes_ride_the_record(self, tmp_path):
        record = resources_from_reading(
            _reading(tmp_path, SCOPE_FILES),
            capture=Capture.RELOCATED,
            reason='work ran in other scopes',
            relocated=frozenset({'run-u1.scope'}),
        )
        assert record.relocated_scopes == frozenset({'run-u1.scope'})


class TestRecordAtExit:
    def test_never_launched_is_unavailable_naming_no_scope(self, tmp_path):
        capture = CommandCapture(_exit_capture_dir(tmp_path))

        record = capture.record(**IDENTITY)

        assert record.capture is Capture.UNAVAILABLE
        assert 'no cgroup scope' in record.capture_reason
        assert record.scope_unit is None

    def test_no_recorded_cgroup_is_unavailable(self, tmp_path):
        directory = _write(tmp_path / 'capture', {**SCOPE_FILES, 'complete': ''})

        record = _launched_capture(directory).record(**IDENTITY)

        assert record.capture is Capture.UNAVAILABLE
        assert 'never recorded its cgroup' in record.capture_reason
        assert record.scope_unit == LAUNCHED

    def test_a_cgroup_other_than_the_launched_scope_is_unavailable(self, tmp_path):
        directory = _exit_capture_dir(tmp_path)
        (directory / 'cgroup').write_text('0::/user.slice/other.scope\n')

        record = _launched_capture(directory).record(**IDENTITY)

        assert record.capture is Capture.UNAVAILABLE
        assert 'other.scope' in record.capture_reason
        assert LAUNCHED in record.capture_reason

    def test_no_complete_marker_is_unavailable(self, tmp_path):
        directory = _exit_capture_dir(tmp_path)
        (directory / 'complete').unlink()

        record = _launched_capture(directory).record(**IDENTITY)

        assert record.capture is Capture.UNAVAILABLE
        assert 'ended before its capture completed' in record.capture_reason

    def test_a_complete_capture_reads_exit(self, tmp_path):
        record = _launched_capture(_exit_capture_dir(tmp_path)).record(**IDENTITY)

        assert record.capture is Capture.EXIT
        assert record.capture_reason
        assert record.cpu_usec == 4200000
        assert record.memory_peak_bytes == 209715200
        assert record.oom_kills == 1
        assert record.scope_unit == LAUNCHED
        assert record.relocated_scopes == frozenset()

    def test_a_noted_relocation_reads_relocated_keeping_the_numbers(self, tmp_path):
        capture = _launched_capture(_exit_capture_dir(tmp_path))
        capture.note_relocated({'run-u1.scope'})

        record = capture.record(**IDENTITY)

        assert record.capture is Capture.RELOCATED
        assert record.relocated_scopes == frozenset({'run-u1.scope'})
        assert 'run-u1.scope' in record.capture_reason
        assert record.cpu_usec == 4200000
        assert record.memory_peak_bytes == 209715200

    def test_relocations_accumulate_across_notes(self, tmp_path):
        capture = _launched_capture(_exit_capture_dir(tmp_path))
        capture.note_relocated({'run-u1.scope'})
        capture.note_relocated(frozenset())
        capture.note_relocated({'run-u2.scope'})

        record = capture.record(**IDENTITY)

        assert record.relocated_scopes == frozenset({'run-u1.scope', 'run-u2.scope'})

    def test_complete_without_cpu_stat_is_unavailable_naming_it(self, tmp_path):
        files = {k: v for k, v in SCOPE_FILES.items() if k != 'cpu.stat'}

        record = _launched_capture(_exit_capture_dir(tmp_path, files)).record(**IDENTITY)

        assert record.capture is Capture.UNAVAILABLE
        assert 'cpu.stat' in record.capture_reason

    def test_a_garbage_peak_is_unavailable_naming_the_file_and_reason(self, tmp_path):
        files = {**SCOPE_FILES, 'memory.peak': 'lots\n'}

        record = _launched_capture(_exit_capture_dir(tmp_path, files)).record(**IDENTITY)

        assert record.capture is Capture.UNAVAILABLE
        assert 'memory.peak' in record.capture_reason
        assert "'lots" in record.capture_reason

    def test_identity_is_stamped_verbatim(self, tmp_path):
        record = _launched_capture(_exit_capture_dir(tmp_path)).record(**IDENTITY)

        for key, value in IDENTITY.items():
            assert getattr(record, key) == value, key
        assert record.host == socket.gethostname()

    def test_an_unavailable_record_is_stamped_too(self, tmp_path):
        record = CommandCapture(tmp_path).record(**IDENTITY)

        assert record.capture is Capture.UNAVAILABLE
        for key, value in IDENTITY.items():
            assert getattr(record, key) == value, key
        assert record.host == socket.gethostname()


def _live_scope(cgroup_root: Path, files: dict[str, str] = SCOPE_FILES) -> Path:
    return _write(cgroup_root / LAUNCH_CGROUP.lstrip('/'), files)


def _launched_with_cgroup(tmp_path: Path, cgroup_root: Path) -> CommandCapture:
    directory = _write(tmp_path / 'capture', {'cgroup': f'0::{LAUNCH_CGROUP}\n'})
    return _launched_capture(directory, cgroup_root=cgroup_root)


class TestReadBeforeKill:
    def test_reads_the_live_scope_directory(self, tmp_path):
        cgroup_root = tmp_path / 'sysfs'
        _live_scope(cgroup_root)
        capture = _launched_with_cgroup(tmp_path, cgroup_root)

        capture.read_before_kill(frozenset())
        record = capture.record(**IDENTITY)

        assert record.capture is Capture.BEFORE_KILL
        assert record.capture_reason
        assert record.cpu_usec == 4200000
        assert record.memory_peak_bytes == 209715200
        assert record.scope_unit == LAUNCHED

    def test_relocated_scopes_found_at_the_kill_make_it_relocated(self, tmp_path):
        cgroup_root = tmp_path / 'sysfs'
        _live_scope(cgroup_root)
        capture = _launched_with_cgroup(tmp_path, cgroup_root)

        capture.read_before_kill(frozenset({'run-u7.scope'}))
        record = capture.record(**IDENTITY)

        assert record.capture is Capture.RELOCATED
        assert record.relocated_scopes == frozenset({'run-u7.scope'})
        assert record.cpu_usec == 4200000

    def test_a_missing_live_directory_is_unavailable_and_never_raises(self, tmp_path):
        capture = _launched_with_cgroup(tmp_path, tmp_path / 'sysfs')

        capture.read_before_kill(frozenset())
        record = capture.record(**IDENTITY)

        assert record.capture is Capture.UNAVAILABLE
        assert record.capture_reason

    def test_an_unlaunched_capture_never_raises(self, tmp_path):
        capture = CommandCapture(tmp_path / 'absent')

        capture.read_before_kill(frozenset())

        assert capture.record(**IDENTITY).capture is Capture.UNAVAILABLE

    def test_a_before_kill_reading_wins_over_a_complete_exit_capture(self, tmp_path):
        cgroup_root = tmp_path / 'sysfs'
        _live_scope(cgroup_root, {**SCOPE_FILES, 'cpu.stat': 'usage_usec 99\n'})
        capture = _launched_capture(_exit_capture_dir(tmp_path), cgroup_root=cgroup_root)

        capture.read_before_kill(frozenset())
        record = capture.record(**IDENTITY)

        assert record.capture is Capture.BEFORE_KILL
        assert record.cpu_usec == 99


class TestAllocateCommandCapture:
    def test_the_directory_exists_inside_the_block_and_is_removed_after(self):
        with allocate_command_capture() as capture:
            directory = capture.directory
            assert directory is not None
            assert directory.is_dir()
            (directory / 'cpu.stat').write_text(CPU_STAT_TEXT)
        assert not directory.exists()

    def test_the_directory_is_removed_when_the_block_raises(self):
        with pytest.raises(RuntimeError), allocate_command_capture() as capture:
            directory = capture.directory
            raise RuntimeError('boom')
        assert directory is not None
        assert not directory.exists()

    def test_an_allocation_failure_degrades_to_an_unavailable_capture(self, monkeypatch):
        def refuse(*args, **kwargs):
            raise OSError(28, 'No space left on device')

        monkeypatch.setattr(tempfile, 'mkdtemp', refuse)

        with allocate_command_capture() as capture:
            assert capture.directory is None
            assert capture.wrap('echo hi') == 'echo hi'
            capture.launched(LAUNCHED)
            record = capture.record(**IDENTITY)

        assert record.capture is Capture.UNAVAILABLE
        assert 'No space left on device' in record.capture_reason


CAPTURED_FILES = frozenset(
    {CPU_STAT, MEMORY_PEAK, SWAP_PEAK, MEMORY_EVENTS, *(r.file_name for r in PressureResource)}
)


def _own_unified_line() -> str:
    lines = Path('/proc/self/cgroup').read_text().splitlines()
    return next(line for line in lines if line.startswith('0::'))


class _FakeScope:
    """A fake cgroup root whose directory at this process's own cgroup holds fixture files."""

    def __init__(self, tmp_path: Path, files: dict[str, str] | None = None) -> None:
        self.root = tmp_path / 'sysfs'
        unified = unified_cgroup_path(_own_unified_line())
        assert unified is not None
        self.unit = unified.rsplit('/', 1)[-1]
        self.directory = _write(
            self.root / unified.lstrip('/'), SCOPE_FILES if files is None else files
        )

    def capture(self, tmp_path: Path) -> CommandCapture:
        directory = tmp_path / 'capture'
        directory.mkdir()
        capture = CommandCapture(directory, cgroup_root=self.root)
        capture.launched(self.unit)
        return capture


def _run_payload(cmd: str) -> tuple[subprocess.CompletedProcess[str], bool]:
    with ScopeLaunchHandshake() as handshake:
        try:
            proc = subprocess.run(
                ['/bin/bash', '-c', handshake.payload(cmd)],
                pass_fds=handshake.pass_fds,
                capture_output=True,
                text=True,
                check=False,
                timeout=UNREACHED_COMMAND_TIMEOUT_SECS,
            )
        finally:
            handshake.release_writer()
        return proc, handshake.launcher_exited_before_exec()


def _outcome(proc: subprocess.CompletedProcess[str]) -> tuple[int, str, str]:
    return proc.returncode, proc.stdout, proc.stderr


HEREDOC_CMD = "cat <<'EOF'\nline one\n  $NOT_EXPANDED\nEOF\necho after"


class TestWrapRunsTheCommandUnchanged:
    @pytest.mark.parametrize(
        'cmd',
        ['echo out; echo err >&2; exit 3', 'echo x; exec true', 'exit 0', HEREDOC_CMD],
    )
    def test_output_and_rc_are_those_of_the_unwrapped_payload(self, tmp_path, cmd):
        capture = _FakeScope(tmp_path).capture(tmp_path)

        wrapped, launcher_exited_before_exec = _run_payload(capture.wrap(cmd))
        bare, _ = _run_payload(cmd)

        assert _outcome(wrapped) == _outcome(bare)
        assert launcher_exited_before_exec is False

    @pytest.mark.parametrize('cmd', ['echo bye; exit 5', 'exec true'])
    def test_exit_and_exec_cannot_skip_the_capture(self, tmp_path, cmd):
        scope = _FakeScope(tmp_path)
        capture = scope.capture(tmp_path)
        assert capture.directory is not None

        proc, _ = _run_payload(capture.wrap(cmd))

        assert proc.returncode == (5 if cmd.startswith('echo') else 0)
        assert (capture.directory / 'complete').exists()
        assert (capture.directory / CPU_STAT).read_bytes() == (
            scope.directory / CPU_STAT
        ).read_bytes()
        assert (capture.directory / 'cgroup').read_text() == _own_unified_line() + '\n'
        assert capture.record(**IDENTITY).capture is Capture.EXIT

    def test_a_syntax_error_still_proves_the_launch_and_reads_unavailable(self, tmp_path):
        capture = _FakeScope(tmp_path).capture(tmp_path)
        assert capture.directory is not None

        proc, launcher_exited_before_exec = _run_payload(capture.wrap('if then fi'))

        assert proc.returncode == 2
        assert 'syntax error' in proc.stderr
        assert launcher_exited_before_exec is False
        assert not (capture.directory / 'complete').exists()
        assert capture.record(**IDENTITY).capture is Capture.UNAVAILABLE

    def test_an_unreadable_sysfs_keeps_the_command_and_reads_unavailable(self, tmp_path):
        scope = _FakeScope(tmp_path, files={})
        capture = scope.capture(tmp_path)
        assert capture.directory is not None
        cmd = 'echo out; echo err >&2; exit 4'

        wrapped, _ = _run_payload(capture.wrap(cmd))
        bare, _ = _run_payload(cmd)

        assert _outcome(wrapped) == _outcome(bare)
        assert (capture.directory / 'complete').exists()
        record = capture.record(**IDENTITY)
        assert record.capture is Capture.UNAVAILABLE
        assert CPU_STAT in record.capture_reason


class TestWrapScriptText:
    def test_it_starts_no_python(self):
        capture = CommandCapture(Path('/run/capture'), cgroup_root=Path('/sys/fs/cgroup'))
        assert 'python' not in capture.wrap('true')

    def test_it_copies_exactly_the_files_named_by_the_shared_constants(self, tmp_path):
        scope = _FakeScope(tmp_path, files={**SCOPE_FILES, 'memory.stat': 'anon 1\n'})
        capture = scope.capture(tmp_path)
        assert capture.directory is not None

        _run_payload(capture.wrap('true'))

        assert frozenset(p.name for p in capture.directory.iterdir()) == (
            CAPTURED_FILES | {'cgroup', 'complete'}
        )

    def test_the_command_stays_on_payload_line_two(self, tmp_path):
        capture = _FakeScope(tmp_path).capture(tmp_path)

        wrapped, _ = _run_payload(capture.wrap('nosuchcommand_xyz'))
        bare, _ = _run_payload('nosuchcommand_xyz')

        assert 'line 2:' in bare.stderr
        assert wrapped.stderr == bare.stderr
        assert wrapped.returncode == bare.returncode == 127


MIB = 1024 * 1024


@pytest.fixture(scope='module')
def real_user_scopes() -> None:
    try:
        probe = subprocess.run(
            ['systemd-run', '--user', '--scope', '--quiet', '--collect', '/bin/true'],
            capture_output=True,
            timeout=SYSTEMD_CALL_TIMEOUT_SECS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f'no working systemd-run --user --scope: {exc}')
    if probe.returncode != 0:
        pytest.skip(f'systemd-run --user --scope refused: {probe.stderr!r}')


@pytest.fixture
def inner_unit() -> Iterator[str]:
    """A unique unit name for a scope the command moves itself into; stopped at teardown."""
    unit = f'df6355reloc{uuid.uuid4().hex[:10]}.scope'
    yield unit
    with contextlib.suppress(Exception):
        subprocess.run(
            ['systemctl', '--user', 'stop', unit],
            capture_output=True,
            timeout=SYSTEMD_CALL_TIMEOUT_SECS,
        )


async def _scoped(
    cmd: str,
    tmp_path: Path,
    slice_name: str,
    *,
    timeout: float = 60,
    capture: CommandCapture | None = None,
) -> tuple[int, str, bool]:
    return await _run_cmd(
        cmd,
        tmp_path,
        timeout=timeout,
        use_cgroup_scope=True,
        scope_tag=scope_tag_for(tmp_path),
        slice_name=slice_name,
        capture=capture,
    )


def _relocating(unit: str, seconds: int) -> str:
    return f'systemd-run --user --scope --quiet --collect --unit={unit} sleep {seconds}; true'


ALLOCATE_200_MIB = (
    "python3 -c \"x = b'x' * (200 * 1024 * 1024); print('allocated', len(x))\"; exec true"
)


@pytest.mark.usefixtures('real_user_scopes')
class TestRunCmdCapturesInItsRealScope:
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_capture_at_exit(self, tmp_path, real_project_slice: RealProjectSlice):
        with allocate_command_capture() as capture:
            captured = await _scoped(
                ALLOCATE_200_MIB, tmp_path, real_project_slice.slice_name, capture=capture
            )
            record = capture.record(**IDENTITY)
        bare = await _scoped(ALLOCATE_200_MIB, tmp_path, real_project_slice.slice_name)

        assert captured == bare
        assert captured[0] == 0
        assert record.capture is Capture.EXIT, record.capture_reason
        assert record.memory_peak_bytes is not None
        assert record.memory_peak_bytes >= 200 * MIB
        assert record.cpu_usec is not None
        assert record.cpu_usec > 0
        assert record.scope_unit is not None
        assert parse_verify_scope_unit(record.scope_unit) is not None

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_capture_at_exit_keeps_a_failing_rc(
        self, tmp_path, real_project_slice: RealProjectSlice
    ):
        with allocate_command_capture() as capture:
            rc, _output, timed_out = await _scoped(
                'echo failing; exit 3', tmp_path, real_project_slice.slice_name, capture=capture
            )
            record = capture.record(**IDENTITY)

        assert (rc, timed_out) == (3, False)
        assert record.capture is Capture.EXIT, record.capture_reason

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    @pytest.mark.timing(reason='the command must outlive its verify timeout to reach the kill path')
    async def test_capture_on_timeout_is_read_before_the_kill(
        self, tmp_path, real_project_slice: RealProjectSlice
    ):
        with allocate_command_capture() as capture:
            _rc, _output, timed_out = await _scoped(
                'sleep 30', tmp_path, real_project_slice.slice_name, timeout=5, capture=capture
            )
            record = capture.record(**IDENTITY)

        assert timed_out is True
        assert record.capture is Capture.BEFORE_KILL, record.capture_reason
        assert record.cpu_usec is not None
        assert record.memory_peak_bytes is not None
        assert record.memory_peak_bytes > 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_a_relocating_command_that_exits_reads_relocated(
        self, tmp_path, inner_unit, real_project_slice: RealProjectSlice
    ):
        with allocate_command_capture() as capture:
            rc, _output, _ = await _scoped(
                _relocating(inner_unit, 4),
                tmp_path,
                real_project_slice.slice_name,
                capture=capture,
            )
            record = capture.record(**IDENTITY)

        assert rc == 0
        assert record.capture is Capture.RELOCATED, record.capture_reason
        assert inner_unit in record.relocated_scopes

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    @pytest.mark.timing(reason='the command must outlive its verify timeout to reach the kill path')
    async def test_a_relocating_command_that_is_killed_reads_relocated(
        self, tmp_path, inner_unit, real_project_slice: RealProjectSlice
    ):
        with allocate_command_capture() as capture:
            _rc, _output, timed_out = await _scoped(
                _relocating(inner_unit, 30),
                tmp_path,
                real_project_slice.slice_name,
                timeout=5,
                capture=capture,
            )
            record = capture.record(**IDENTITY)

        assert timed_out is True
        assert record.capture is Capture.RELOCATED, record.capture_reason
        assert inner_unit in record.relocated_scopes


class TestRunCmdWithoutAScope:
    @pytest.mark.asyncio
    async def test_scope_off_runs_unchanged_and_reads_unavailable(self, tmp_path):
        cmd = 'echo out; echo err >&2; exit 4'
        with allocate_command_capture() as capture:
            captured = await _run_cmd(cmd, tmp_path, timeout=UNREACHED_COMMAND_TIMEOUT_SECS, capture=capture)
            record = capture.record(**IDENTITY)
        bare = await _run_cmd(cmd, tmp_path, timeout=UNREACHED_COMMAND_TIMEOUT_SECS)

        assert captured == bare
        assert record.capture is Capture.UNAVAILABLE
        assert 'no cgroup scope' in record.capture_reason

    @pytest.mark.asyncio
    async def test_no_systemd_run_runs_unchanged_and_reads_unavailable(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(shutil, 'which', lambda _name: None)
        cmd = 'echo out; exit 2'
        with allocate_command_capture() as capture:
            captured = await _run_cmd(
                cmd,
                tmp_path,
                timeout=UNREACHED_COMMAND_TIMEOUT_SECS,
                use_cgroup_scope=True,
                scope_tag=scope_tag_for(tmp_path),
                capture=capture,
            )
            record = capture.record(**IDENTITY)
        bare = await _run_cmd(cmd, tmp_path, timeout=UNREACHED_COMMAND_TIMEOUT_SECS)

        assert captured == bare
        assert record.capture is Capture.UNAVAILABLE
        assert 'no cgroup scope' in record.capture_reason


class TestRunCmdDefaultSpawn:
    @pytest.mark.asyncio
    async def test_no_capture_spawns_the_bare_payload(self, tmp_path, fake_systemd_run):
        cmd = 'echo hello; exit 0'

        rc, output, _ = await _run_cmd(
            cmd,
            tmp_path,
            timeout=UNREACHED_COMMAND_TIMEOUT_SECS,
            use_cgroup_scope=True,
            scope_tag=scope_tag_for(tmp_path),
        )

        assert (rc, output) == (0, 'hello\n')
        (spawn,) = recorded_scope_spawns(fake_systemd_run)
        _handshake, newline, command = spawn[-1].partition('\n')
        assert newline
        assert command == cmd


ALLOCATE_50_MIB = (
    "python3 -c \"x = b'x' * (50 * 1024 * 1024); print('allocated', len(x))\"; exec true"
)


@pytest.mark.usefixtures('real_user_scopes', 'code_default_config')
class TestRunVerificationRecordsTheRealScope:
    """The task's observable signal in miniature: a real scoped verify's written summary."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_the_summary_records_cpu_and_peak_captured_at_exit(
        self, tmp_path, real_project_slice: RealProjectSlice
    ):
        worktree = tmp_path / 'wt'
        (worktree / '.task').mkdir(parents=True)
        config = OrchestratorConfig(
            project_root=worktree,
            verify_use_cgroup_scope=True,
            verify_admission_enabled=False,
            fused_memory=FusedMemoryConfig(project_id=real_project_slice.project_id),
        )
        module = ModuleConfig(
            prefix='pkg',
            test_command=ALLOCATE_50_MIB,
            lint_command=None,
            type_check_command=None,
        )

        result = await run_verification(
            worktree, config, module, attempt_id=1, task_id='6355', max_retries=0, role='task'
        )

        assert result.passed, result.summary
        summary = json.loads(
            (worktree / '.task' / 'verify' / 'attempt-1.pkg.summary.json').read_text()
        )
        (test,) = [entry for entry in summary['commands'] if entry['label'] == 'test']
        resources = test['resources']
        assert resources['capture'] == 'exit', resources['capture_reason']
        assert resources['cpu_usec'] > 0
        assert resources['memory_peak_bytes'] >= 50 * MIB
        assert resources['project'] == real_project_slice.project_id
