"""MarkupResidueSpool — the durable floor under the markup-residue queue (task 4809).

Driven through the public surface only: ``preserve`` and ``status``. Every
envelope literal is BUILT from ``shared.toolcall_markup``'s constants, never
spelled raw (``tests/scripts/test_no_raw_envelope_literal.py``).
"""

from __future__ import annotations

import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import pytest
from shared.toolcall_markup import CANONICAL_OPENER_PREFIX, closer_for

from fused_memory.server.markup_residue_spool import MarkupResidueSpool

_LOGGER = 'fused_memory.server.markup_residue_spool'
_LOCATOR_TIME_FORMAT = '%Y%m%dT%H%M%S.%fZ'

_RAW_VALUE = (
    'a memory about café — 日本語'
    + closer_for('content')
    + CANONICAL_OPENER_PREFIX
    + '"project_id">reify'
    + closer_for('parameter')
)


def _record(raw_value: object = _RAW_VALUE, **extra: object) -> dict[str, object]:
    return {
        'error_type': 'mcp_markup_unrepairable',
        'tool': 'add_memory',
        'field': 'content',
        'project': None,
        'raw_value': raw_value,
        **extra,
    }


def _unwritable_spool(tmp_path) -> MarkupResidueSpool:
    """A spool whose parent is a regular FILE: root-safe, unlike chmod."""
    blocker = tmp_path / 'not-a-directory'
    blocker.write_text('occupied')
    return MarkupResidueSpool(blocker / 'spool')


def _stored(locator: str | None) -> dict:
    assert locator is not None
    return json.loads(Path(locator).read_text(encoding='utf-8'))


class _Fsync(NamedTuple):
    """One ``os.fsync``: what it synced, and what the spool dir held right then."""

    inode: int
    is_dir: bool
    spool_names: list[str]


def _record_fsyncs(monkeypatch, spool_dir: Path) -> list[_Fsync]:
    """Every ``os.fsync`` from here on, in order, each still really performed.

    The spool's names at each sync are what a crash at that instant leaves.
    """
    calls: list[_Fsync] = []
    real_fsync = os.fsync

    def _recording_fsync(fd: int) -> None:
        synced = os.fstat(fd)
        names = sorted(path.name for path in spool_dir.iterdir()) if spool_dir.is_dir() else []
        calls.append(_Fsync(synced.st_ino, stat.S_ISDIR(synced.st_mode), names))
        real_fsync(fd)

    monkeypatch.setattr(os, 'fsync', _recording_fsync)
    return calls


def _record_listings(monkeypatch) -> list[Path]:
    """Every dir ``os.scandir`` lists from here on."""
    listed: list[Path] = []
    real_scandir = os.scandir

    def _recording_scandir(path):
        listed.append(Path(path))
        return real_scandir(path)

    monkeypatch.setattr(os, 'scandir', _recording_scandir)
    return listed


class TestPreserve:
    def test_it_creates_the_dir_and_returns_the_absolute_path_of_one_file(self, tmp_path):
        spool_dir = tmp_path / 'data' / 'markup_residue'
        spool = MarkupResidueSpool(spool_dir)

        locator = spool.preserve(_record())

        assert locator is not None
        assert Path(locator).is_absolute()
        assert list(spool_dir.iterdir()) == [Path(locator)]

    def test_a_relative_spool_dir_still_yields_an_absolute_path(self, tmp_path, monkeypatch):
        """The path reaches a caller that does not share the server's cwd."""
        monkeypatch.chdir(tmp_path)
        spool = MarkupResidueSpool(Path('data') / 'markup_residue')

        locator = spool.preserve(_record())

        assert locator is not None
        assert Path(locator).is_absolute()
        assert Path(locator).parent == tmp_path / 'data' / 'markup_residue'

    def test_the_file_name_leads_with_the_utc_time_it_was_written(self, tmp_path):
        """Fixed-width UTC timestamps sort lexically in time order."""
        spool = MarkupResidueSpool(tmp_path / 'spool')
        before = datetime.now(UTC).replace(tzinfo=None)

        locator = spool.preserve(_record())

        after = datetime.now(UTC).replace(tzinfo=None)
        assert locator is not None
        written = datetime.strptime(Path(locator).name.split('-', 1)[0], _LOCATOR_TIME_FORMAT)
        assert before <= written <= after

    def test_the_payload_round_trips_byte_for_byte(self, tmp_path):
        spool = MarkupResidueSpool(tmp_path / 'spool')

        locator = spool.preserve(_record())

        assert _stored(locator)['raw_value'] == _RAW_VALUE

    def test_identical_records_land_in_two_files_and_both_survive(self, tmp_path):
        spool = MarkupResidueSpool(tmp_path / 'spool')

        first = spool.preserve(_record())
        second = spool.preserve(_record())

        assert first != second
        assert _stored(first)['raw_value'] == _RAW_VALUE
        assert _stored(second)['raw_value'] == _RAW_VALUE

    def test_a_file_already_in_the_dir_is_never_overwritten(self, tmp_path):
        spool_dir = tmp_path / 'spool'
        spool_dir.mkdir()
        existing = spool_dir / 'operator-note.json'
        existing.write_text('{"keep": true}')
        spool = MarkupResidueSpool(spool_dir)

        locator = spool.preserve(_record())

        assert locator is not None
        assert Path(locator) != existing
        assert existing.read_text() == '{"keep": true}'
        assert len(list(spool_dir.iterdir())) == 2

    def test_a_non_json_value_still_preserves_the_payload(self, tmp_path):
        """The diagnostic fields beat nothing: an exotic value is repr'd."""
        spool = MarkupResidueSpool(tmp_path / 'spool')

        locator = spool.preserve(_record(exotic=object(), names={'a'}))

        assert _stored(locator)['raw_value'] == _RAW_VALUE

    def test_a_record_without_raw_value_still_writes(self, tmp_path):
        spool = MarkupResidueSpool(tmp_path / 'spool')
        record = _record()
        del record['raw_value']

        locator = spool.preserve(record)

        assert _stored(locator)['tool'] == 'add_memory'


class TestAPreservedPayloadSurvivesACrash:
    """The locator reaches the caller, so what it names must outlive power loss."""

    def test_no_json_name_exists_while_the_payload_is_being_synced(
        self, tmp_path, monkeypatch
    ):
        """A crash mid-write leaves no partial file that ``spooled_files`` counts."""
        spool_dir = tmp_path / 'spool'
        spool = MarkupResidueSpool(spool_dir)
        fsyncs = _record_fsyncs(monkeypatch, spool_dir)

        locator = spool.preserve(_record())

        payload_syncs = [call for call in fsyncs if not call.is_dir]
        assert payload_syncs, fsyncs
        for call in payload_syncs:
            assert [name for name in call.spool_names if name.endswith('.json')] == []
        assert [path.name for path in spool_dir.iterdir()] == [Path(locator).name]

    def test_the_spool_dir_is_synced_once_the_new_name_is_in_it(
        self, tmp_path, monkeypatch
    ):
        spool_dir = tmp_path / 'spool'
        spool_dir.mkdir()
        spool = MarkupResidueSpool(spool_dir)
        fsyncs = _record_fsyncs(monkeypatch, spool_dir)

        locator = spool.preserve(_record())

        assert locator is not None
        assert any(
            call.is_dir
            and call.inode == spool_dir.stat().st_ino
            and Path(locator).name in call.spool_names
            for call in fsyncs
        ), fsyncs

    def test_each_dir_it_creates_is_synced_into_its_parent(self, tmp_path, monkeypatch):
        spool_dir = tmp_path / 'data' / 'markup_residue'
        spool = MarkupResidueSpool(spool_dir)
        fsyncs = _record_fsyncs(monkeypatch, spool_dir)

        assert spool.preserve(_record()) is not None

        synced_dirs = {call.inode for call in fsyncs if call.is_dir}
        assert tmp_path.stat().st_ino in synced_dirs
        assert (tmp_path / 'data').stat().st_ino in synced_dirs

    def test_a_payload_that_cannot_be_synced_is_lost_and_leaves_nothing(
        self, tmp_path, monkeypatch
    ):
        spool_dir = tmp_path / 'spool'
        spool = MarkupResidueSpool(spool_dir)

        def _refuse_to_sync(fd: int) -> None:
            raise OSError('disk gone')

        monkeypatch.setattr(os, 'fsync', _refuse_to_sync)

        assert spool.preserve(_record()) is None
        assert list(spool_dir.iterdir()) == []
        assert spool.status()['preservation_failures'] == 1


class TestAFailedWriteIsALostPayload:
    def test_an_unwritable_dir_returns_none_without_raising(self, tmp_path):
        spool = _unwritable_spool(tmp_path)

        assert spool.preserve(_record()) is None

    def test_it_leaves_nothing_behind(self, tmp_path):
        spool = _unwritable_spool(tmp_path)

        spool.preserve(_record())

        assert [path.name for path in tmp_path.iterdir()] == ['not-a-directory']
        assert (tmp_path / 'not-a-directory').read_text() == 'occupied'

    def test_a_descriptor_that_cannot_be_wrapped_is_closed_and_its_file_removed(
        self, tmp_path, monkeypatch
    ):
        spool_dir = tmp_path / 'spool'
        spool = MarkupResidueSpool(spool_dir)
        claimed: list[int] = []

        def _refuse_to_wrap(fd: int, *args: object, **kwargs: object) -> object:
            claimed.append(fd)
            raise OSError('cannot wrap')

        monkeypatch.setattr(os, 'fdopen', _refuse_to_wrap)

        assert spool.preserve(_record()) is None
        [fd] = claimed
        with pytest.raises(OSError):
            os.fstat(fd)
        assert list(spool_dir.iterdir()) == []
        assert spool.status()['preservation_failures'] == 1

    def test_it_logs_one_greppable_error_line_with_the_running_total(self, tmp_path, caplog):
        spool = _unwritable_spool(tmp_path)
        raw_value = 'x' * 4321

        with caplog.at_level('ERROR', logger=_LOGGER):
            spool.preserve(_record(raw_value))
            spool.preserve(_record(raw_value, tool='add_episode'))

        lines = [
            record.getMessage()
            for record in caplog.records
            if record.name == _LOGGER and record.levelname == 'ERROR'
        ]
        assert len(lines) == 2, lines
        first, second = lines
        assert first.startswith('markup_residue_lost')
        assert 'add_memory' in first and 'content' in first
        assert '4321' in first
        assert 'preservation_failures=1' in first
        assert 'add_episode' in second
        assert 'preservation_failures=2' in second


class TestStatus:
    def test_a_fresh_spool_reports_zero_losses_and_its_dir(self, tmp_path):
        spool = MarkupResidueSpool(tmp_path / 'spool')

        status = spool.status()

        assert status['preservation_failures'] == 0
        assert status['last_failure'] is None
        assert status['spooled_files'] == 0
        assert status['last_preserved'] is None
        assert status['spool_dir'] == str(tmp_path / 'spool')

    def test_it_reports_the_files_waiting_and_the_newest_one(self, tmp_path):
        spool = MarkupResidueSpool(tmp_path / 'spool')

        spool.preserve(_record())
        newest = spool.preserve(_record())

        status = spool.status()
        assert status['spooled_files'] == 2
        assert status['last_preserved'] == newest

    def test_a_file_an_operator_drained_is_no_longer_counted(self, tmp_path):
        spool = MarkupResidueSpool(tmp_path / 'spool')
        drained = spool.preserve(_record())
        spool.preserve(_record())
        assert drained is not None

        Path(drained).unlink()

        assert spool.status()['spooled_files'] == 1

    def test_an_unlistable_dir_reports_an_unknown_count(self, tmp_path):
        assert _unwritable_spool(tmp_path).status()['spooled_files'] is None

    def test_a_failure_is_counted_with_its_tool_field_and_time(self, tmp_path):
        spool = _unwritable_spool(tmp_path)
        before = datetime.now(UTC)

        spool.preserve(_record())

        status = spool.status()
        assert status['preservation_failures'] == 1
        last = status['last_failure']
        assert last['tool'] == 'add_memory'
        assert last['field'] == 'content'
        assert before <= datetime.fromisoformat(last['at']) <= datetime.now(UTC)

    def test_the_status_is_a_copy(self, tmp_path):
        spool = _unwritable_spool(tmp_path)
        spool.preserve(_record())

        status = spool.status()
        status['preservation_failures'] = 99
        status['last_failure']['tool'] = 'tampered'

        assert spool.status()['preservation_failures'] == 1
        assert spool.status()['last_failure']['tool'] == 'add_memory'

    def test_the_count_is_per_instance(self, tmp_path):
        spool = _unwritable_spool(tmp_path)
        spool.preserve(_record())

        sibling = MarkupResidueSpool(tmp_path / 'not-a-directory' / 'spool')

        assert sibling.status()['preservation_failures'] == 0

    def test_a_burst_of_writes_lists_the_dir_at_most_once(self, tmp_path, monkeypatch):
        """Each write is O(1): the count is kept between status reads, not re-listed."""
        spool_dir = tmp_path / 'spool'
        spool = MarkupResidueSpool(spool_dir, soft_cap_files=2)
        listed = _record_listings(monkeypatch)

        for _ in range(5):
            assert spool.preserve(_record()) is not None

        assert listed.count(spool_dir) <= 1
        assert spool.status()['spooled_files'] == 5

    @pytest.mark.parametrize('calls', [1, 3])
    def test_a_successful_preserve_is_not_counted(self, tmp_path, calls):
        spool = MarkupResidueSpool(tmp_path / 'spool')

        for _ in range(calls):
            assert spool.preserve(_record()) is not None

        assert spool.status()['preservation_failures'] == 0
        assert spool.status()['last_failure'] is None


class TestSoftCap:
    """Past the soft cap every payload is still preserved, and the crossing is
    logged once rather than once per write."""

    @staticmethod
    def _over_cap_lines(caplog) -> list[str]:
        return [
            record.getMessage()
            for record in caplog.records
            if record.name == _LOGGER
            and record.getMessage().startswith('markup_residue_spool_over_soft_cap')
        ]

    def test_crossing_the_cap_logs_once_and_keeps_preserving(self, tmp_path, caplog):
        spool = MarkupResidueSpool(tmp_path / 'spool', soft_cap_files=2)

        with caplog.at_level('ERROR', logger=_LOGGER):
            locators = [spool.preserve(_record()) for _ in range(5)]

        assert all(locator is not None for locator in locators)
        assert spool.status()['spooled_files'] == 5
        assert len(self._over_cap_lines(caplog)) == 1

    def test_at_the_cap_nothing_is_logged(self, tmp_path, caplog):
        spool = MarkupResidueSpool(tmp_path / 'spool', soft_cap_files=2)

        with caplog.at_level('ERROR', logger=_LOGGER):
            spool.preserve(_record())
            spool.preserve(_record())

        assert self._over_cap_lines(caplog) == []

    def test_a_drained_spool_logs_again_when_it_next_crosses(self, tmp_path, caplog):
        """The drain is seen at the next status read, as get_status polls it."""
        spool_dir = tmp_path / 'spool'
        spool = MarkupResidueSpool(spool_dir, soft_cap_files=1)

        with caplog.at_level('ERROR', logger=_LOGGER):
            spool.preserve(_record())
            spool.preserve(_record())
            for path in spool_dir.iterdir():
                path.unlink()
            spool.status()
            spool.preserve(_record())
            spool.preserve(_record())

        assert len(self._over_cap_lines(caplog)) == 2
