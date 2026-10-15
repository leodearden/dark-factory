"""Run-record schema, store, retention and read side of the offline lane."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shlex
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from test_offline_lane import _child_writing, _make_config, _make_git_ops, _make_worker

from orchestrator.config import LaneCommand
from orchestrator.offline_lane import OfflineLaneWorker
from orchestrator.offline_lane_records import (
    FIRST_LINE_MAX_CHARS,
    LEGACY_COMMANDS,
    LEGACY_INFRA_COMMAND,
    LEGACY_NUMERIC_COMMAND,
    RECORD_RETENTION,
    EndedBy,
    FailureRecord,
    Leg,
    PassRecord,
    Phase,
    ProbeResult,
    RunRecord,
    RunRecordStore,
    classify_exit,
    command_key,
    is_well_formed,
    pass_id_for,
    records_root,
)
from orchestrator.offline_lane_runs import (
    RUN_ALL_INFRA_SCRIPT,
    RUN_OFFLINE_DEEP_SCRIPT,
    require_current_pass,
)

T0 = datetime(2026, 10, 1, 12, 30, 5, tzinfo=UTC)
HEAD = 'abcdef0123456789'


def _record(
    lane_pass_command: str = 'unit',
    phase: Phase = Phase.DETECT,
    *,
    returncode: int = 1,
    failures: tuple[FailureRecord, ...] | None = None,
) -> RunRecord:
    return RunRecord(
        command=lane_pass_command,
        leg=Leg.GENERIC.value,
        head=HEAD,
        phase=phase.value,
        started_at=T0,
        ended_at=T0 + timedelta(seconds=3),
        returncode=returncode,
        ended_by=classify_exit(returncode).value,
        output_path=f'/records/{phase.value}.log',
        junit_path=None,
        failures=(
            (FailureRecord('tests/test_q.py::test_a', 'failed', 'AssertionError: boom'),)
            if failures is None
            else failures
        ),
        ids_available=True,
        outcomes=None,
        session_complete=True,
        probe=None,
    )


class _Clock:
    def __init__(self, start: datetime) -> None:
        self.current = start

    def __call__(self) -> datetime:
        return self.current


@pytest.fixture
def clock() -> _Clock:
    return _Clock(T0)


@pytest.fixture
def store(tmp_path: Path, clock: _Clock) -> RunRecordStore:
    return RunRecordStore(tmp_path, clock=clock)


_needs_file_permissions = pytest.mark.skipif(
    os.geteuid() == 0, reason='root removes a read-only directory regardless'
)


@contextlib.contextmanager
def _locked_subdirectory(pass_dir: Path) -> Iterator[None]:
    """A non-empty read-only subdirectory of *pass_dir*: ``rmtree`` fails with PermissionError."""
    locked = pass_dir / 'locked'
    locked.mkdir()
    (locked / 'output').write_text('x')
    locked.chmod(0o500)
    try:
        yield
    finally:
        locked.chmod(0o700)


class TestClassifyExit:
    @pytest.mark.parametrize('returncode', [-9, -15, 137, 143, 129, 130])
    def test_external_kill_forms_are_signal(self, returncode: int) -> None:
        assert classify_exit(returncode) is EndedBy.SIGNAL

    @pytest.mark.parametrize('returncode', [0, 1, 2, 139, 255, 128, 193])
    def test_other_codes_are_exit(self, returncode: int) -> None:
        assert classify_exit(returncode) is EndedBy.EXIT

    def test_any_negative_code_is_signal(self) -> None:
        assert classify_exit(-11) is EndedBy.SIGNAL


class TestCommandKey:
    def test_legacy_names_pass_through(self) -> None:
        assert command_key(LEGACY_NUMERIC_COMMAND) == 'legacy_numeric'
        assert command_key(LEGACY_INFRA_COMMAND) == 'legacy_infra'

    @pytest.mark.parametrize(
        ('name', 'key'),
        [
            ('Qdrant Integration', 'qdrant-integration'),
            ('a__b', 'a-b'),
            ('--x--', 'x'),
            ('', 'command'),
            ('___', 'command'),
            ('Ünïcode λ', 'n-code'),
        ],
    )
    def test_generic_names_are_slugged(self, name: str, key: str) -> None:
        assert command_key(name) == key

    @pytest.mark.parametrize(
        'name', ['LEGACY_NUMERIC', 'legacy_numeric ', 'legacy-numeric', 'legacy_infra!', 'Legacy_Infra']
    )
    def test_no_generic_name_collides_with_a_legacy_key(self, name: str) -> None:
        assert command_key(name) not in LEGACY_COMMANDS
        assert '_' not in command_key(name)

    def test_a_legacy_name_in_another_case_is_slugged_like_any_other(self) -> None:
        assert command_key('LEGACY_NUMERIC') == 'legacy-numeric'


class TestRecordsRoot:
    def test_real_root_gets_the_data_layout(self, tmp_path: Path) -> None:
        assert records_root(tmp_path) == tmp_path / 'data' / 'orchestrator' / 'offline-lane'

    @pytest.mark.parametrize('unusable', [None, '', 'None'])
    def test_unusable_root_falls_back_to_a_temp_directory_with_a_warning(
        self,
        unusable: str | None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with caplog.at_level(logging.WARNING):
            root = records_root(unusable)
        try:
            assert root.is_dir()
            assert root.name.startswith('offline-lane-records-')
            assert any(r.levelno == logging.WARNING for r in caplog.records)
            assert not (tmp_path / 'None').exists()
            assert list(tmp_path.iterdir()) == []
        finally:
            root.rmdir()


class TestRunRecordJson:
    def test_round_trip_preserves_every_field(self) -> None:
        probe = ProbeResult('outage', None, 'qdrant', 'connection refused', 1.5)
        record = RunRecord(**{**_record().__dict__, 'probe': probe, 'outcomes': {'a': 'passed'}})
        assert RunRecord.from_json(json.loads(json.dumps(record.to_json()))) == record

    def test_missing_key_is_a_value_error(self) -> None:
        payload = _record().to_json()
        del payload['returncode']
        with pytest.raises(ValueError):
            RunRecord.from_json(payload)

    @pytest.mark.parametrize(
        ('key', 'value'),
        [
            ('phase', 'bogus'),
            ('leg', 'bogus'),
            ('ended_by', 'bogus'),
            ('returncode', 'one'),
            ('returncode', True),
            ('started_at', '2026-10-01T12:30:05'),
            ('started_at', 'yesterday'),
            ('ids_available', 1),
            ('failures', 'none'),
            ('failures', [{'node': 'n', 'kind': 'bogus', 'first_line': ''}]),
            ('probe', {'result': 'sideways'}),
            ('outcomes', ['tests/test_q.py::test_a']),
            ('outcomes', {'tests/test_q.py::test_a': 1}),
            ('started_at', '0001-01-01T00:00:00+01:00'),
        ],
    )
    def test_malformed_value_is_a_value_error(self, key: str, value: Any) -> None:
        payload = _record().to_json()
        payload[key] = value
        with pytest.raises(ValueError):
            RunRecord.from_json(payload)

    def test_first_line_cap_is_the_documented_constant(self) -> None:
        assert FIRST_LINE_MAX_CHARS == 300


class TestAllocatePass:
    def test_pass_id_is_utc_start_plus_short_head(self) -> None:
        assert pass_id_for(T0, HEAD) == '20261001T123005-abcdef'

    def test_allocation_writes_the_header_before_any_run(self, store: RunRecordStore) -> None:
        lane_pass = store.allocate_pass('Qdrant Integration', Leg.GENERIC, HEAD)

        assert lane_pass.directory == store.root / 'qdrant-integration' / '20261001T123005-abcdef'
        assert lane_pass.output_path(Phase.CONFIRM) == lane_pass.directory / 'confirm.log'
        assert lane_pass.junit_path('detect') == lane_pass.directory / 'detect.junit.xml'
        payload = json.loads(lane_pass.record_path.read_text())
        assert payload == {
            'command': 'Qdrant Integration',
            'leg': 'generic',
            'head': HEAD,
            'pass_id': '20261001T123005-abcdef',
            'started_at': T0.isoformat(),
            'detect': None,
            'confirm': None,
            'outcome': None,
            'effect': None,
        }
        assert store.current_pass('Qdrant Integration') == lane_pass

    def test_unknown_leg_is_rejected(self, store: RunRecordStore) -> None:
        with pytest.raises(ValueError):
            store.allocate_pass('unit', 'sideways', HEAD)

    @pytest.mark.parametrize('legacy_name', sorted(LEGACY_COMMANDS))
    def test_a_generic_pass_cannot_take_a_legacy_name(
        self, store: RunRecordStore, legacy_name: str
    ) -> None:
        with pytest.raises(ValueError, match=legacy_name):
            store.allocate_pass(legacy_name, Leg.GENERIC, HEAD)
        assert store.current_pass(legacy_name) is None

    def test_require_current_pass_is_the_allocated_pass(self, store: RunRecordStore) -> None:
        lane_pass = store.allocate_pass('unit', Leg.GENERIC, HEAD)
        assert require_current_pass(store, 'unit') == lane_pass

    def test_require_current_pass_without_one_names_the_command(self, store: RunRecordStore) -> None:
        with pytest.raises(RuntimeError, match='never-allocated'):
            require_current_pass(store, 'never-allocated')

    def test_second_allocation_replaces_the_pass_and_forgets_last_records(
        self, store: RunRecordStore, clock: _Clock
    ) -> None:
        first = store.allocate_pass('unit', Leg.GENERIC, HEAD)
        store.write_run(first, _record())
        other = store.allocate_pass('other', Leg.GENERIC, HEAD)
        store.write_run(other, _record('other'))

        clock.current = T0 + timedelta(minutes=5)
        second = store.allocate_pass('unit', Leg.GENERIC, HEAD)

        assert store.current_pass('unit') == second != first
        assert store.last_record('unit', Phase.DETECT) is None
        assert store.last_record('other', Phase.DETECT) is not None
        assert store.current_pass('never-allocated') is None

    def test_header_only_pass_is_well_formed(self, store: RunRecordStore) -> None:
        lane_pass = store.allocate_pass('unit', Leg.NUMERIC, HEAD)
        loaded = store.load_pass('unit', lane_pass.pass_id)
        assert loaded is not None
        assert (loaded.detect, loaded.confirm, loaded.outcome, loaded.effect) == (None,) * 4


class TestWriteRun:
    def test_detect_record_round_trips_through_disk(self, store: RunRecordStore) -> None:
        lane_pass = store.allocate_pass('unit', Leg.GENERIC, HEAD)
        record = _record(failures=(
            FailureRecord('t::a', 'failed', 'x'),
            FailureRecord('t::b', 'error', ''),
        ))

        store.write_run(lane_pass, record)

        loaded = store.load_pass('unit', lane_pass.pass_id)
        assert loaded is not None
        assert loaded.detect is not None
        assert loaded.detect == record
        assert loaded.detect.junit_path is None
        assert loaded.detect.outcomes is None
        assert loaded.detect.probe is None
        assert loaded.confirm is None
        assert store.last_record('unit', Phase.DETECT) == record
        assert store.last_record('unit', 'detect') == record
        assert store.last_record('unit', Phase.CONFIRM) is None

    def test_phases_are_kept_side_by_side(self, store: RunRecordStore) -> None:
        lane_pass = store.allocate_pass('unit', Leg.GENERIC, HEAD)
        detect, confirm = _record(), _record(phase=Phase.CONFIRM, returncode=0, failures=())
        store.write_run(lane_pass, detect)
        store.write_run(lane_pass, confirm)

        loaded = store.load_pass('unit', lane_pass.pass_id)
        assert loaded is not None
        assert (loaded.detect, loaded.confirm) == (detect, confirm)

    def test_write_leaves_no_temp_residue(self, store: RunRecordStore) -> None:
        lane_pass = store.allocate_pass('unit', Leg.GENERIC, HEAD)
        store.write_run(lane_pass, _record())
        assert [p.name for p in lane_pass.directory.iterdir()] == ['record.json']

    def test_missing_header_is_a_lookup_error_naming_the_path(self, store: RunRecordStore) -> None:
        lane_pass = store.allocate_pass('unit', Leg.GENERIC, HEAD)
        lane_pass.record_path.unlink()
        with pytest.raises(LookupError, match='record.json'):
            store.write_run(lane_pass, _record())

    def test_unreadable_header_is_a_lookup_error(self, store: RunRecordStore) -> None:
        lane_pass = store.allocate_pass('unit', Leg.GENERIC, HEAD)
        lane_pass.record_path.write_text('{not json')
        with pytest.raises(LookupError):
            store.write_run(lane_pass, _record())


class TestPassBlocks:
    def test_only_the_given_block_is_set(self, store: RunRecordStore) -> None:
        lane_pass = store.allocate_pass('unit', Leg.GENERIC, HEAD)

        store.set_pass_blocks(lane_pass, outcome={'arm': 'intersected'})
        loaded = store.load_pass('unit', lane_pass.pass_id)
        assert loaded is not None
        assert (loaded.outcome, loaded.effect) == ({'arm': 'intersected'}, None)

        store.set_pass_blocks(lane_pass, effect=['filed 7001'])
        loaded = store.load_pass('unit', lane_pass.pass_id)
        assert loaded is not None
        assert (loaded.outcome, loaded.effect) == ({'arm': 'intersected'}, ['filed 7001'])

    def test_an_explicit_none_clears_a_block(self, store: RunRecordStore) -> None:
        lane_pass = store.allocate_pass('unit', Leg.GENERIC, HEAD)
        store.set_pass_blocks(lane_pass, outcome='x', effect='y')
        store.set_pass_blocks(lane_pass, outcome=None)
        loaded = store.load_pass('unit', lane_pass.pass_id)
        assert loaded is not None
        assert (loaded.outcome, loaded.effect) == (None, 'y')

    def test_blocks_survive_a_later_run_write(self, store: RunRecordStore) -> None:
        lane_pass = store.allocate_pass('unit', Leg.GENERIC, HEAD)
        store.set_pass_blocks(lane_pass, outcome='kept')
        store.write_run(lane_pass, _record())
        loaded = store.load_pass('unit', lane_pass.pass_id)
        assert loaded is not None
        assert loaded.outcome == 'kept'


class TestSetProbe:
    def test_probe_is_written_into_the_detect_record(self, store: RunRecordStore) -> None:
        lane_pass = store.allocate_pass('unit', Leg.GENERIC, HEAD)
        detect = _record()
        store.write_run(lane_pass, detect)
        probe = ProbeResult('outage', 7, 'qdrant', 'connection refused', 2.25)

        store.set_probe(lane_pass, probe)

        loaded = store.load_pass('unit', lane_pass.pass_id)
        assert loaded is not None
        assert loaded.detect == RunRecord(**{**detect.__dict__, 'probe': probe})
        assert store.last_record('unit', Phase.DETECT) == loaded.detect

    def test_probe_before_any_detect_record_is_a_lookup_error(self, store: RunRecordStore) -> None:
        lane_pass = store.allocate_pass('unit', Leg.GENERIC, HEAD)
        with pytest.raises(LookupError):
            store.set_probe(lane_pass, ProbeResult('up', 0, None, '', 0.1))


class TestWellFormedness:
    def _header(self, store: RunRecordStore) -> dict[str, Any]:
        lane_pass = store.allocate_pass('unit', Leg.GENERIC, HEAD)
        store.write_run(lane_pass, _record())
        return json.loads(lane_pass.record_path.read_text())

    def test_a_written_pass_is_well_formed(self, store: RunRecordStore) -> None:
        assert is_well_formed(self._header(store))

    @pytest.mark.parametrize('payload', [None, [], 'record', 3])
    def test_a_non_object_is_not_well_formed(self, payload: object) -> None:
        assert not is_well_formed(payload)

    @pytest.mark.parametrize('key', ['command', 'leg', 'head', 'pass_id', 'started_at', 'detect', 'confirm', 'outcome', 'effect'])
    def test_a_header_missing_a_key_is_not_well_formed(self, store: RunRecordStore, key: str) -> None:
        payload = self._header(store)
        del payload[key]
        assert not is_well_formed(payload)

    def test_a_detect_block_with_an_unknown_failure_kind_is_not_well_formed(
        self, store: RunRecordStore
    ) -> None:
        payload = self._header(store)
        payload['detect']['failures'][0]['kind'] = 'exploded'
        assert not is_well_formed(payload)

    def test_a_detect_block_that_is_not_an_object_is_not_well_formed(self, store: RunRecordStore) -> None:
        payload = self._header(store)
        payload['detect'] = 'broken'
        assert not is_well_formed(payload)

    def test_load_of_a_malformed_file_is_none_with_a_warning(
        self, store: RunRecordStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        lane_pass = store.allocate_pass('unit', Leg.GENERIC, HEAD)
        lane_pass.record_path.write_text(json.dumps({'command': 'unit'}))
        with caplog.at_level(logging.WARNING):
            assert store.load_pass('unit', lane_pass.pass_id) is None
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_load_of_unparseable_json_is_none_with_a_warning(
        self, store: RunRecordStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        lane_pass = store.allocate_pass('unit', Leg.GENERIC, HEAD)
        lane_pass.record_path.write_text('{truncated')
        with caplog.at_level(logging.WARNING):
            assert store.load_pass('unit', lane_pass.pass_id) is None
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_a_start_out_of_range_in_utc_is_skipped_with_a_warning(
        self, store: RunRecordStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        lane_pass = store.allocate_pass('unit', Leg.GENERIC, HEAD)
        payload = json.loads(lane_pass.record_path.read_text())
        payload['started_at'] = '0001-01-01T00:00:00+01:00'
        lane_pass.record_path.write_text(json.dumps(payload))

        assert not is_well_formed(payload)
        with caplog.at_level(logging.WARNING):
            assert store.load_pass('unit', lane_pass.pass_id) is None
            assert list(store.passes_since('unit', T0)) == []
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2

    def test_load_of_an_absent_pass_is_none_without_a_warning(
        self, store: RunRecordStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            assert store.load_pass('unit', '20260101T000000-abcdef') is None
        assert caplog.records == []


class TestPassesSince:
    def _allocate_at(self, store: RunRecordStore, clock: _Clock, when: datetime, command: str = 'unit') -> str:
        clock.current = when
        return store.allocate_pass(command, Leg.GENERIC, HEAD).pass_id

    def test_passes_come_back_ascending_from_the_cut(self, store: RunRecordStore, clock: _Clock) -> None:
        late = self._allocate_at(store, clock, T0 + timedelta(hours=2))
        early = self._allocate_at(store, clock, T0)
        middle = self._allocate_at(store, clock, T0 + timedelta(hours=1))
        self._allocate_at(store, clock, T0 + timedelta(hours=1), command='other')

        everything = [p.pass_id for p in store.passes_since('unit', T0)]
        cut = [p.pass_id for p in store.passes_since('unit', T0 + timedelta(hours=1))]

        assert everything == [early, middle, late]
        assert cut == [middle, late]

    def test_a_stray_directory_is_ignored(self, store: RunRecordStore, clock: _Clock) -> None:
        real = self._allocate_at(store, clock, T0)
        (store.root / 'unit' / 'scratch').mkdir()
        (store.root / 'unit' / 'notes.txt').write_text('x')

        assert [p.pass_id for p in store.passes_since('unit', T0)] == [real]

    def test_a_malformed_pass_is_skipped_with_a_warning(
        self, store: RunRecordStore, clock: _Clock, caplog: pytest.LogCaptureFixture
    ) -> None:
        broken = store.allocate_pass('unit', Leg.GENERIC, HEAD)
        broken.record_path.write_text('{}')
        good = self._allocate_at(store, clock, T0 + timedelta(hours=1))

        with caplog.at_level(logging.WARNING):
            passes = [p.pass_id for p in store.passes_since('unit', T0)]

        assert passes == [good]
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_an_unknown_command_has_no_passes(self, store: RunRecordStore) -> None:
        assert list(store.passes_since('never-ran', T0)) == []

    def test_names_that_slug_alike_read_back_only_their_own_passes(
        self, store: RunRecordStore, clock: _Clock
    ) -> None:
        spaced = self._allocate_at(store, clock, T0, command='Unit Tests')
        dashed = self._allocate_at(store, clock, T0 + timedelta(hours=1), command='unit-tests')

        assert command_key('Unit Tests') == command_key('unit-tests')
        assert store.load_pass('Unit Tests', dashed) is None
        assert [p.pass_id for p in store.passes_since('unit-tests', T0)] == [dashed]
        assert [p.pass_id for p in store.passes_since('Unit Tests', T0)] == [spaced]


class TestRetention:
    def _allocate_aged(self, store: RunRecordStore, clock: _Clock, now: datetime, days: int) -> str:
        clock.current = now - timedelta(days=days)
        lane_pass = store.allocate_pass('unit', Leg.GENERIC, HEAD)
        store.write_run(lane_pass, _record())
        return lane_pass.pass_id

    def test_retention_is_fourteen_days(self) -> None:
        assert timedelta(days=14) == RECORD_RETENTION

    def test_a_pass_older_than_retention_is_removed_and_a_newer_one_kept(
        self, store: RunRecordStore, clock: _Clock
    ) -> None:
        now = T0 + timedelta(days=30)
        old = self._allocate_aged(store, clock, now, 15)
        recent = self._allocate_aged(store, clock, now, 13)

        assert store.prune(now) == [old]

        assert not (store.root / 'unit' / old).exists()
        assert (store.root / 'unit' / recent / 'record.json').is_file()

    def test_a_pinned_pass_survives(self, store: RunRecordStore, clock: _Clock) -> None:
        now = T0 + timedelta(days=30)
        old = self._allocate_aged(store, clock, now, 15)

        assert store.prune(now, pinned_pass_ids={old}) == []

        assert (store.root / 'unit' / old / 'record.json').is_file()

    def test_pruning_covers_every_command(self, store: RunRecordStore, clock: _Clock) -> None:
        now = T0 + timedelta(days=30)
        clock.current = now - timedelta(days=20)
        first = store.allocate_pass('one', Leg.GENERIC, HEAD).pass_id
        second = store.allocate_pass(LEGACY_INFRA_COMMAND, Leg.INFRA, 'fedcba9876').pass_id

        assert sorted(store.prune(now)) == sorted([first, second])

    def test_unparsable_directories_are_left_alone(self, store: RunRecordStore) -> None:
        stray = store.root / 'unit' / 'not-a-pass'
        stray.mkdir(parents=True)

        assert store.prune(T0 + timedelta(days=365)) == []
        assert stray.is_dir()

    def test_age_comes_from_the_pass_id_not_the_file_contents(
        self, store: RunRecordStore, clock: _Clock
    ) -> None:
        now = T0 + timedelta(days=30)
        old = self._allocate_aged(store, clock, now, 15)
        (store.root / 'unit' / old / 'record.json').write_text('garbage')

        assert store.prune(now) == [old]

    def test_a_missing_root_prunes_nothing(self, tmp_path: Path) -> None:
        store = RunRecordStore(tmp_path / 'never-created')
        assert store.prune(T0) == []

    @_needs_file_permissions
    def test_a_pass_that_cannot_be_removed_is_skipped_with_a_warning(
        self, store: RunRecordStore, clock: _Clock, caplog: pytest.LogCaptureFixture
    ) -> None:
        now = T0 + timedelta(days=30)
        stuck = self._allocate_aged(store, clock, now, 15)
        removable = self._allocate_aged(store, clock, now, 16)

        with _locked_subdirectory(store.root / 'unit' / stuck), caplog.at_level(logging.WARNING):
            assert store.prune(now) == [removable]

        assert (store.root / 'unit' / stuck).is_dir()
        assert any(
            r.levelno == logging.WARNING and str(store.root / 'unit' / stuck) in r.getMessage()
            for r in caplog.records
        )


_LANE_COMMAND = 'unit'
_PASS_DEADLINE_SECS = 120.0


def _lane_worker(tmp_path: Path, **git_overrides: Any) -> OfflineLaneWorker:
    """A worker whose worktree is ``tmp_path / 'wt'``, with every leg off unless *git_overrides* enables it."""
    worktree = tmp_path / 'wt'
    worktree.mkdir(exist_ok=True)
    config = _make_config(
        tmp_path,
        **{
            'offline_lane_legacy_numeric_enabled': False,
            'offline_lane_poll_interval_secs': 3600.0,
            **git_overrides,
        },
    )
    return _make_worker(
        tmp_path, git_ops=_make_git_ops(head=HEAD, worktree_path=worktree), config=config,
    )


def _generic_worker(tmp_path: Path, command: str) -> OfflineLaneWorker:
    return _lane_worker(
        tmp_path, offline_lane_commands=[LaneCommand(name=_LANE_COMMAND, command=command, cwd='.')],
    )


def _add_lane_script(tmp_path: Path, relative_path: str, body: str) -> None:
    """Put an executable ``sh`` script at *relative_path* in the lane worktree, for a default seam to run."""
    script = tmp_path / 'wt' / relative_path
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(f'#!/bin/sh\n{body}\n')
    script.chmod(0o755)


async def _until(done: Callable[[], bool]) -> None:
    deadline = time.monotonic() + _PASS_DEADLINE_SECS
    while not done():
        assert time.monotonic() < deadline, 'the lane pass left no record in time'
        await asyncio.sleep(0.02)


async def _drive_one_pass(worker: OfflineLaneWorker, command: str, until_phase: Phase) -> None:
    """Drive one lane pass through ``on_post_merge`` + ``run()`` until *command*'s *until_phase* is recorded."""
    task = asyncio.create_task(worker.run())
    try:
        await worker.on_post_merge('t', 'base', 'head')
        await _until(lambda: worker.records.last_record(command, until_phase) is not None)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def _run_one_pass(tmp_path: Path, command: str, *, until_phase: Phase) -> OfflineLaneWorker:
    worker = _generic_worker(tmp_path, command)
    await _drive_one_pass(worker, _LANE_COMMAND, until_phase)
    return worker


def _current_pass_record(worker: OfflineLaneWorker, command: str = _LANE_COMMAND) -> PassRecord:
    lane_pass = worker.records.current_pass(command)
    assert lane_pass is not None
    record = worker.records.load_pass(command, lane_pass.pass_id)
    assert record is not None
    return record


def _sub_run_logs(caplog: pytest.LogCaptureFixture, command: str) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if r.getMessage().startswith(f'offline-lane: {command} sub-run')
    ]


def _seed_pass(tmp_path: Path, age: timedelta) -> Path:
    """Allocate a pass of the lane's generic command that started *age* ago; its directory."""
    started_at = datetime.now(UTC) - age
    return RunRecordStore(tmp_path, clock=lambda: started_at).allocate_pass(
        _LANE_COMMAND, Leg.GENERIC, HEAD
    ).directory


class TestLanePassRecords:
    """Scenario 1 and the record half of scenario 2: every lane sub-run leaves a record."""

    @pytest.mark.asyncio
    async def test_a_green_pass_records_its_detect_run_and_output(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger='orchestrator.offline_lane'):
            worker = await _run_one_pass(
                tmp_path, "printf '=== 3 passed in 0.10s ===\\n'; exit 0", until_phase=Phase.DETECT,
            )

        detect = _current_pass_record(worker).detect
        assert detect is not None
        assert detect.returncode == 0
        assert detect.ended_by == EndedBy.EXIT
        assert detect.phase == Phase.DETECT
        assert detect.junit_path is None
        assert detect.failures == ()
        assert detect.ids_available is False
        assert detect.session_complete is True
        assert Path(detect.output_path).read_text() == '=== 3 passed in 0.10s ===\n'
        assert ['ended_by=exit' in line for line in _sub_run_logs(caplog, _LANE_COMMAND)] == [True]

    @pytest.mark.asyncio
    async def test_a_red_pass_names_the_failing_test_with_kind_and_first_line(
        self, tmp_path: Path
    ) -> None:
        worker = await _run_one_pass(
            tmp_path,
            "printf 'FAILED tests/test_q.py::test_a - AssertionError: boom\\n'; exit 1",
            until_phase=Phase.CONFIRM,
        )

        record = _current_pass_record(worker)
        assert record.detect is not None
        assert record.detect.returncode == 1
        assert record.detect.ended_by == EndedBy.EXIT
        assert record.detect.failures == (
            FailureRecord('tests/test_q.py::test_a', 'failed', 'AssertionError: boom'),
        )
        assert record.detect.ids_available is True
        assert record.confirm is not None
        assert record.confirm.phase == Phase.CONFIRM
        assert worker.records.last_record(_LANE_COMMAND, Phase.DETECT) == record.detect

    @pytest.mark.asyncio
    async def test_a_failure_reason_is_stored_capped(self, tmp_path: Path) -> None:
        reason = 'x' * (FIRST_LINE_MAX_CHARS + 100)
        worker = await _run_one_pass(
            tmp_path,
            f"printf 'FAILED tests/test_q.py::test_a - {reason}\\n'; exit 1",
            until_phase=Phase.CONFIRM,
        )

        detect = _current_pass_record(worker).detect
        assert detect is not None
        assert [failure.first_line for failure in detect.failures] == [reason[:FIRST_LINE_MAX_CHARS]]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('summary', 'session_complete'),
        [('=== 1 failed in 0.10s ===\\n', True), ('', False)],
        ids=['final-summary', 'no-final-summary'],
    )
    async def test_a_generic_run_is_session_complete_only_with_pytests_final_summary(
        self, tmp_path: Path, summary: str, session_complete: bool
    ) -> None:
        worker = await _run_one_pass(
            tmp_path,
            f"printf 'FAILED tests/test_q.py::test_a - AssertionError: boom\\n{summary}'; exit 1",
            until_phase=Phase.CONFIRM,
        )

        detect = _current_pass_record(worker).detect
        assert detect is not None
        assert detect.session_complete is session_complete

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('command', 'returncode'),
        [('kill -9 $$', -9), ('sh -c "kill -9 \\$\\$"', 137)],
        ids=['negative-returncode', 'sh-c-128-plus-n'],
    )
    async def test_a_signal_ended_run_records_ended_by_signal(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, command: str, returncode: int
    ) -> None:
        with caplog.at_level(logging.INFO, logger='orchestrator.offline_lane'):
            worker = await _run_one_pass(tmp_path, command, until_phase=Phase.CONFIRM)

        detect = _current_pass_record(worker).detect
        assert detect is not None
        assert detect.returncode == returncode
        assert detect.ended_by == EndedBy.SIGNAL
        assert ['ended_by=signal' in line for line in _sub_run_logs(caplog, _LANE_COMMAND)] == [True]

    def test_an_allocated_pass_has_its_header_before_any_sub_run(self, tmp_path: Path) -> None:
        worker = _generic_worker(tmp_path, 'exit 0')
        lane_pass = worker.records.allocate_pass(_LANE_COMMAND, Leg.GENERIC, HEAD)

        record = worker.records.load_pass(_LANE_COMMAND, lane_pass.pass_id)

        assert record is not None
        assert (record.command, record.leg, record.head) == (_LANE_COMMAND, Leg.GENERIC, HEAD)
        assert record.pass_id == lane_pass.pass_id
        assert (record.detect, record.confirm, record.outcome, record.effect) == (
            None, None, None, None,
        )

    @pytest.mark.asyncio
    async def test_a_structured_pytest_command_writes_junit_into_the_pass(
        self, tmp_path: Path
    ) -> None:
        worker = _generic_worker(tmp_path, 'pytest -q tests')
        cmd = LaneCommand(name=_LANE_COMMAND, command='pytest -q tests', cwd='.')
        lane_pass = worker.records.allocate_pass(_LANE_COMMAND, Leg.GENERIC, HEAD)

        with patch(
            'orchestrator.offline_lane_runs.asyncio.create_subprocess_exec',
            side_effect=_child_writing(b'1 passed in 0.01s\n', returncode=0),
        ) as child:
            await worker.command_runner(cmd, tmp_path / 'wt', HEAD)

        junit_path = lane_pass.junit_path(Phase.DETECT)
        assert child.call_args.args[-3:] == (
            'sh', '-c', f'pytest -q --junitxml {shlex.quote(str(junit_path))} tests',
        )
        detect = worker.records.last_record(_LANE_COMMAND, Phase.DETECT)
        assert detect is not None
        assert detect.junit_path == str(junit_path)
        assert detect.session_complete is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'command', ['make check', 'pytest -q tests && echo done'], ids=['opaque', 'raw-retained-pytest'],
    )
    async def test_a_command_with_no_structured_pytest_runs_unchanged_without_junit(
        self, tmp_path: Path, command: str
    ) -> None:
        worker = _generic_worker(tmp_path, command)
        cmd = LaneCommand(name=_LANE_COMMAND, command=command, cwd='.')
        worker.records.allocate_pass(_LANE_COMMAND, Leg.GENERIC, HEAD)

        with patch(
            'orchestrator.offline_lane_runs.asyncio.create_subprocess_exec',
            side_effect=_child_writing(b'make: Nothing to be done.\n', returncode=0),
        ) as child:
            await worker.command_runner(cmd, tmp_path / 'wt', HEAD)

        assert child.call_args.args[-3:] == ('sh', '-c', command)
        detect = worker.records.last_record(_LANE_COMMAND, Phase.DETECT)
        assert detect is not None
        assert detect.junit_path is None


class TestLegacyLegRecords:
    """The numeric and infra legs' default seams, run against stub scripts in the lane worktree."""

    @pytest.mark.asyncio
    async def test_the_infra_leg_records_each_failing_test_and_its_confirm_run(
        self, tmp_path: Path
    ) -> None:
        worker = _lane_worker(tmp_path, offline_lane_infra_enabled=True)
        _add_lane_script(tmp_path, RUN_ALL_INFRA_SCRIPT, "echo '  RESULT: FAIL (test_x.sh)'\nexit 1")

        await _drive_one_pass(worker, LEGACY_INFRA_COMMAND, Phase.CONFIRM)

        record = _current_pass_record(worker, LEGACY_INFRA_COMMAND)
        assert record.leg == Leg.INFRA
        assert record.detect is not None
        assert record.detect.failures == (
            FailureRecord('test_x.sh', 'failed', 'RESULT: FAIL (test_x.sh)'),
        )
        assert record.detect.ids_available is True
        assert record.confirm is not None
        assert record.confirm.phase == Phase.CONFIRM

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('body', 'until_phase', 'session_complete'),
        [('exit 0', Phase.DETECT, True), ('kill -9 $$', Phase.CONFIRM, False)],
        ids=['exit', 'signal'],
    )
    async def test_the_numeric_leg_records_an_opaque_detect_run(
        self, tmp_path: Path, body: str, until_phase: Phase, session_complete: bool
    ) -> None:
        worker = _lane_worker(tmp_path, offline_lane_legacy_numeric_enabled=True)
        _add_lane_script(tmp_path, RUN_OFFLINE_DEEP_SCRIPT, body)

        await _drive_one_pass(worker, LEGACY_NUMERIC_COMMAND, until_phase)

        record = _current_pass_record(worker, LEGACY_NUMERIC_COMMAND)
        assert record.leg == Leg.NUMERIC
        assert record.detect is not None
        assert record.detect.failures == ()
        assert record.detect.ids_available is False
        assert record.detect.session_complete is session_complete


class TestLaneRetention:
    @pytest.mark.asyncio
    async def test_a_lane_pass_prunes_expired_records_and_keeps_recent_ones(
        self, tmp_path: Path
    ) -> None:
        expired = _seed_pass(tmp_path, RECORD_RETENTION + timedelta(days=6))
        recent = _seed_pass(tmp_path, timedelta(days=1))

        await _run_one_pass(tmp_path, 'exit 0', until_phase=Phase.DETECT)

        assert not expired.exists()
        assert (recent / 'record.json').is_file()

    @_needs_file_permissions
    @pytest.mark.asyncio
    async def test_an_expired_pass_that_cannot_be_removed_does_not_stop_the_lane(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        stuck = _seed_pass(tmp_path, RECORD_RETENTION + timedelta(days=6))

        with _locked_subdirectory(stuck), caplog.at_level(logging.WARNING):
            worker = await _run_one_pass(tmp_path, 'exit 0', until_phase=Phase.DETECT)

        assert _current_pass_record(worker).detect is not None
        assert any(
            r.levelno == logging.WARNING and str(stuck) in r.getMessage() for r in caplog.records
        )
