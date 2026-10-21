"""MarkupResidueSpool — the durable floor under the markup-residue queue (task 4809).

Driven through the public surface only: ``preserve`` and ``status``. Every
envelope literal is BUILT from ``shared.toolcall_markup``'s constants, never
spelled raw (``tests/scripts/test_no_raw_envelope_literal.py``).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

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


def _stored(spool_dir, locator: str) -> dict:
    return json.loads((spool_dir / locator).read_text(encoding='utf-8'))


class TestPreserve:
    def test_it_creates_the_dir_and_writes_one_file_named_by_the_locator(self, tmp_path):
        spool_dir = tmp_path / 'data' / 'markup_residue'
        spool = MarkupResidueSpool(spool_dir)

        locator = spool.preserve(_record())

        assert locator is not None
        assert [path.name for path in spool_dir.iterdir()] == [locator]
        assert '/' not in locator

    def test_the_locator_leads_with_the_utc_time_it_was_written(self, tmp_path):
        """Fixed-width UTC timestamps sort lexically in time order."""
        spool = MarkupResidueSpool(tmp_path / 'spool')
        before = datetime.now(UTC).replace(tzinfo=None)

        locator = spool.preserve(_record())

        after = datetime.now(UTC).replace(tzinfo=None)
        assert locator is not None
        written = datetime.strptime(locator.split('-', 1)[0], _LOCATOR_TIME_FORMAT)
        assert before <= written <= after

    def test_the_payload_round_trips_byte_for_byte(self, tmp_path):
        spool = MarkupResidueSpool(tmp_path / 'spool')

        locator = spool.preserve(_record())

        assert locator is not None
        assert _stored(tmp_path / 'spool', locator)['raw_value'] == _RAW_VALUE

    def test_identical_records_land_in_two_files_and_both_survive(self, tmp_path):
        spool = MarkupResidueSpool(tmp_path / 'spool')

        first = spool.preserve(_record())
        second = spool.preserve(_record())

        assert first is not None and second is not None
        assert first != second
        assert _stored(tmp_path / 'spool', first)['raw_value'] == _RAW_VALUE
        assert _stored(tmp_path / 'spool', second)['raw_value'] == _RAW_VALUE

    def test_a_file_already_in_the_dir_is_never_overwritten(self, tmp_path):
        spool_dir = tmp_path / 'spool'
        spool_dir.mkdir()
        existing = spool_dir / 'operator-note.json'
        existing.write_text('{"keep": true}')
        spool = MarkupResidueSpool(spool_dir)

        locator = spool.preserve(_record())

        assert locator != existing.name
        assert existing.read_text() == '{"keep": true}'
        assert len(list(spool_dir.iterdir())) == 2

    def test_a_non_json_value_still_preserves_the_payload(self, tmp_path):
        """The diagnostic fields beat nothing: an exotic value is repr'd."""
        spool = MarkupResidueSpool(tmp_path / 'spool')

        locator = spool.preserve(_record(exotic=object(), names={'a'}))

        assert locator is not None
        assert _stored(tmp_path / 'spool', locator)['raw_value'] == _RAW_VALUE

    def test_a_record_without_raw_value_still_writes(self, tmp_path):
        spool = MarkupResidueSpool(tmp_path / 'spool')
        record = _record()
        del record['raw_value']

        locator = spool.preserve(record)

        assert locator is not None
        assert _stored(tmp_path / 'spool', locator)['tool'] == 'add_memory'


class TestAFailedWriteIsALostPayload:
    def test_an_unwritable_dir_returns_none_without_raising(self, tmp_path):
        spool = _unwritable_spool(tmp_path)

        assert spool.preserve(_record()) is None

    def test_it_leaves_nothing_behind(self, tmp_path):
        spool = _unwritable_spool(tmp_path)

        spool.preserve(_record())

        assert [path.name for path in tmp_path.iterdir()] == ['not-a-directory']
        assert (tmp_path / 'not-a-directory').read_text() == 'occupied'

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
        assert status['spool_dir'] == str(tmp_path / 'spool')

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

    @pytest.mark.parametrize('calls', [1, 3])
    def test_a_successful_preserve_is_not_counted(self, tmp_path, calls):
        spool = MarkupResidueSpool(tmp_path / 'spool')

        for _ in range(calls):
            assert spool.preserve(_record()) is not None

        assert spool.status()['preservation_failures'] == 0
        assert spool.status()['last_failure'] is None
