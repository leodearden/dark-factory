"""Contract guards on ``escalation.refile_absorption`` — which resolutions are absorbing refiles.

Task 4915. ``escalation/src/escalation/queue.py::EscalationQueue.note_suppressed_refile``
bumps ``refiles_suppressed`` on a resolved or dismissed record each time its
resolution suppresses an identical refile. No triage surface lists terminal
records, so the counter was readable only by someone who already knew the id.
These tests hold the listing that closes that discovery gap.

Every counted record is built through the REAL queue (submit -> resolve ->
note_suppressed_refile), so the listing is held against the archive layout and
JSON production writes, never against this file's beliefs about them.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from escalation.models import Escalation
from escalation.queue import EscalationQueue, iter_all_escalation_paths
from escalation.refile_absorption import AbsorptionReport, absorbing_resolutions, main

_CITATION = 'a' * 40


class _Fixture:
    """Builds records through the real queue's production write paths."""

    def __init__(self, tmp_path: Path):
        self.queue = EscalationQueue(tmp_path / 'escalations')
        self._seq = 0

    def _record(self, category: str, citation_sha: str | None) -> Escalation:
        self._seq += 1
        return Escalation(
            id=f'esc-4915-{self._seq}', task_id='4915',
            agent_role='orchestrator-landing-evidence', severity='blocking',
            category=category, summary='refile absorption subject', level=1,
            citation_sha=citation_sha,
        )

    def submit(
        self, *, category: str = 'provenance_unattributed',
        citation_sha: str | None = _CITATION,
    ) -> Escalation:
        record = self._record(category, citation_sha)
        self.queue.submit(record)
        return record

    def submit_task_less(self) -> Escalation:
        """Submit a record with ``task_id`` null, the shape
        ``fused_memory.reconciliation.backlog_policy.BacklogPolicy`` files."""
        shape = self._record('infra_issue', None).to_dict()
        record = Escalation.from_dict({**shape, 'task_id': None})
        self.queue.submit(record)
        return record

    def resolve(self, record: Escalation) -> None:
        resolved = self.queue.resolve(record.id, 'ruled', dismiss=True, resolved_by='watcher')
        assert resolved is not None

    def absorb(self, record: Escalation, times: int) -> None:
        for _ in range(times):
            assert self.queue.note_suppressed_refile(record.id) is not None

    def resolved_absorbing(
        self, times: int, *, category: str = 'provenance_unattributed',
        citation_sha: str | None = _CITATION,
    ) -> Escalation:
        record = self.submit(category=category, citation_sha=citation_sha)
        self.resolve(record)
        self.absorb(record, times)
        return record

    def write_by_hand(self, **fields: object) -> Path:
        """Write a dismissed record absorbing two refiles straight into the queue
        root, with *fields* overridden as a hand edit or a foreign writer could;
        the real queue never writes an ill-typed field."""
        record = self._record('provenance_unattributed', _CITATION)
        payload = {**record.to_dict(), 'status': 'dismissed', 'refiles_suppressed': 2, **fields}
        self.queue.queue_dir.mkdir(parents=True, exist_ok=True)
        path = self.queue.queue_dir / f'{record.id}.json'
        path.write_text(json.dumps(payload))
        return path

    def report(self) -> AbsorptionReport:
        return absorbing_resolutions(self.queue.queue_dir)


def _tier_of(fixture: _Fixture, escalation_id: str) -> str:
    paths = [
        p for p in iter_all_escalation_paths(fixture.queue.queue_dir)
        if p.stem == escalation_id
    ]
    assert len(paths) == 1, f'expected exactly one file for {escalation_id}, got {paths}'
    return 'archive' if 'archive' in paths[0].relative_to(fixture.queue.queue_dir).parts else 'root'


class TestListing:
    def test_a_resolution_bumped_three_times_is_one_row_carrying_the_record(
        self, tmp_path: Path,
    ):
        fixture = _Fixture(tmp_path)
        record = fixture.resolved_absorbing(3)

        report = fixture.report()

        assert len(report.rows) == 1
        row = report.rows[0]
        assert row.escalation_id == record.id
        assert row.task_id == '4915'
        assert row.category == 'provenance_unattributed'
        assert row.status == 'dismissed'
        assert row.refiles_suppressed == 3
        assert row.citation_sha == _CITATION
        assert row.resolved_by == 'watcher'
        assert row.resolved_at is not None
        assert report.total_absorbed == 3

    def test_unbumped_and_pending_records_are_scanned_but_not_listed(self, tmp_path: Path):
        fixture = _Fixture(tmp_path)
        bumped = fixture.resolved_absorbing(1)
        never_bumped = fixture.submit()
        fixture.resolve(never_bumped)
        fixture.submit()

        report = fixture.report()

        assert [row.escalation_id for row in report.rows] == [bumped.id]
        assert report.scanned == 3

    def test_rows_sort_by_count_descending_then_by_id(self, tmp_path: Path):
        fixture = _Fixture(tmp_path)
        two = fixture.resolved_absorbing(2)
        five = fixture.resolved_absorbing(5)
        tied_first = fixture.resolved_absorbing(1)
        tied_second = fixture.resolved_absorbing(1)

        report = fixture.report()

        assert [row.escalation_id for row in report.rows] == [
            five.id, two.id, tied_first.id, tied_second.id,
        ]
        assert report.total_absorbed == 9

    def test_the_listing_is_category_agnostic(self, tmp_path: Path):
        """Two producers bump this counter: landing evidence keyed on a citation
        SHA, and the reconciliation harness's finding-task filer with none."""
        fixture = _Fixture(tmp_path)
        provenance = fixture.resolved_absorbing(2)
        finding = fixture.resolved_absorbing(
            1, category='recon_task_finding', citation_sha=None,
        )

        rows = {row.escalation_id: row for row in fixture.report().rows}

        assert rows[provenance.id].category == 'provenance_unattributed'
        assert rows[provenance.id].citation_sha == _CITATION
        assert rows[finding.id].category == 'recon_task_finding'
        assert rows[finding.id].citation_sha is None

    def test_both_the_queue_root_and_the_archive_are_swept(self, tmp_path: Path):
        """A failed archive move leaves the resolved file in the queue root
        (``queue.py::EscalationQueue._archive_resolved``); a regular file where
        the archive directory belongs is a real way to make that move fail."""
        fixture = _Fixture(tmp_path)
        fixture.queue.queue_dir.mkdir(parents=True, exist_ok=True)
        blocker = fixture.queue.queue_dir / 'archive'
        blocker.write_text('not a directory')
        in_root = fixture.resolved_absorbing(4)
        blocker.unlink()
        in_archive = fixture.resolved_absorbing(2)
        assert _tier_of(fixture, in_root.id) == 'root'
        assert _tier_of(fixture, in_archive.id) == 'archive'

        report = fixture.report()

        assert [row.escalation_id for row in report.rows] == [in_root.id, in_archive.id]
        assert report.total_absorbed == 6


class TestScanCarriesItsShortfall:
    """One bad file costs one record, and the report says which kind it was."""

    def test_a_non_json_file_is_counted_unparsable(self, tmp_path: Path):
        fixture = _Fixture(tmp_path)
        record = fixture.resolved_absorbing(1)
        (fixture.queue.queue_dir / 'esc-junk-1.json').write_text('this is not json')

        report = fixture.report()

        assert [row.escalation_id for row in report.rows] == [record.id]
        assert report.skipped_unparsable == 1
        assert (report.skipped_vanished, report.skipped_unreadable) == (0, 0)
        assert report.scanned == 1

    def test_a_non_utf8_file_is_counted_unparsable_without_raising(self, tmp_path: Path):
        fixture = _Fixture(tmp_path)
        record = fixture.resolved_absorbing(1)
        (fixture.queue.queue_dir / 'esc-bin-1.json').write_bytes(b'\xff\xfe\x00')

        report = fixture.report()

        assert [row.escalation_id for row in report.rows] == [record.id]
        assert report.skipped_unparsable == 1

    @pytest.mark.parametrize('content', ['[]', 'null', '"text"', '42'])
    def test_valid_json_that_is_not_an_object_is_counted_unparsable(
        self, tmp_path: Path, content: str,
    ):
        fixture = _Fixture(tmp_path)
        record = fixture.resolved_absorbing(1)
        (fixture.queue.queue_dir / 'esc-list-1.json').write_text(content)

        report = fixture.report()

        assert [row.escalation_id for row in report.rows] == [record.id]
        assert report.skipped_unparsable == 1
        assert report.scanned == 1

    @pytest.mark.parametrize('counter', [None, '3', 1.5, True])
    def test_a_counter_that_is_not_a_count_is_counted_unparsable(
        self, tmp_path: Path, counter: object, caplog: pytest.LogCaptureFixture,
    ):
        fixture = _Fixture(tmp_path)
        record = fixture.resolved_absorbing(1)
        corrupt = fixture.write_by_hand(refiles_suppressed=counter)

        report = fixture.report()

        assert [row.escalation_id for row in report.rows] == [record.id]
        assert report.skipped_unparsable == 1
        assert report.scanned == 1
        assert str(corrupt) in caplog.text

    @pytest.mark.parametrize(
        ('field', 'value'),
        [('id', None), ('category', None), ('status', None), ('task_id', 4915),
         ('resolved_by', 7)],
    )
    def test_an_absorbing_record_with_an_ill_typed_row_field_is_counted_unparsable(
        self, tmp_path: Path, field: str, value: object,
    ):
        fixture = _Fixture(tmp_path)
        record = fixture.resolved_absorbing(1)
        fixture.write_by_hand(**{field: value})

        report = fixture.report()

        assert [row.escalation_id for row in report.rows] == [record.id]
        assert report.skipped_unparsable == 1

    def test_an_ill_typed_row_field_on_an_unlisted_record_is_not_a_loss(
        self, tmp_path: Path,
    ):
        """Only a record that becomes a row needs renderable text fields."""
        fixture = _Fixture(tmp_path)
        fixture.write_by_hand(refiles_suppressed=0, category=None)

        report = fixture.report()

        assert report.rows == ()
        assert (report.scanned, report.skipped_unparsable) == (1, 0)

    def test_a_file_gone_between_listing_and_read_is_counted_vanished(self, tmp_path: Path):
        """A dangling symlink is globbed like any other esc-*.json and then
        fails the read with the same ENOENT a concurrent relocation produces."""
        fixture = _Fixture(tmp_path)
        record = fixture.resolved_absorbing(1)
        queue_dir = fixture.queue.queue_dir
        (queue_dir / 'esc-vanished-1.json').symlink_to(queue_dir / 'gone.json')

        report = fixture.report()

        assert [row.escalation_id for row in report.rows] == [record.id]
        assert report.skipped_vanished == 1
        assert (report.skipped_unreadable, report.skipped_unparsable) == (0, 0)

    @pytest.mark.skipif(os.geteuid() == 0, reason='root reads a mode-000 file')
    def test_a_file_present_but_unreadable_is_counted_unreadable(self, tmp_path: Path):
        fixture = _Fixture(tmp_path)
        record = fixture.resolved_absorbing(1)
        unreadable = fixture.queue.queue_dir / 'esc-unreadable-1.json'
        unreadable.write_text('{}')
        unreadable.chmod(0o000)
        try:
            report = fixture.report()
        finally:
            unreadable.chmod(0o600)

        assert [row.escalation_id for row in report.rows] == [record.id]
        assert report.skipped_unreadable == 1
        assert (report.skipped_vanished, report.skipped_unparsable) == (0, 0)

    def test_a_missing_queue_dir_is_an_empty_report(self, tmp_path: Path):
        report = absorbing_resolutions(tmp_path / 'does-not-exist')

        assert report.rows == ()
        assert report.total_absorbed == 0
        assert (
            report.scanned, report.skipped_vanished,
            report.skipped_unreadable, report.skipped_unparsable,
        ) == (0, 0, 0, 0)


class TestReportIsImmutable:
    def test_the_report_is_frozen(self, tmp_path: Path):
        fixture = _Fixture(tmp_path)
        fixture.resolved_absorbing(1)
        report = fixture.report()

        field = 'scanned'
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(report, field, 0)

    def test_a_row_is_frozen(self, tmp_path: Path):
        fixture = _Fixture(tmp_path)
        fixture.resolved_absorbing(1)
        row = fixture.report().rows[0]

        field = 'refiles_suppressed'
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(row, field, 0)


def _summary_tokens(out: str) -> set[str]:
    return {token for token in out.split() if '=' in token}


class TestCliTable:
    def test_a_bumped_record_is_one_row_and_the_summary_line_follows(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ):
        fixture = _Fixture(tmp_path)
        record = fixture.resolved_absorbing(4)

        rc = main(['--queue-dir', str(fixture.queue.queue_dir)])

        out = capsys.readouterr().out
        assert rc == 0
        row_lines = [line for line in out.splitlines() if record.id in line]
        assert len(row_lines) == 1
        cells = row_lines[0].split()
        for value in ('4915', 'provenance_unattributed', 'dismissed', '4'):
            assert value in cells
        assert {
            'total_absorbed=4', 'records=1', 'scanned=1', 'skipped_vanished=0',
            'skipped_unreadable=0', 'skipped_unparsable=0',
        } <= _summary_tokens(out)

    def test_an_empty_listing_says_so_and_still_prints_the_summary(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ):
        fixture = _Fixture(tmp_path)
        unbumped = fixture.submit()
        fixture.resolve(unbumped)

        rc = main(['--queue-dir', str(fixture.queue.queue_dir)])

        out = capsys.readouterr().out
        assert rc == 0
        assert 'no resolution has absorbed a refile' in out
        assert {
            'total_absorbed=0', 'records=0', 'scanned=1', 'skipped_vanished=0',
            'skipped_unreadable=0', 'skipped_unparsable=0',
        } <= _summary_tokens(out)

    def test_a_record_filed_with_no_task_id_is_a_row_with_a_dash(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ):
        """Production files task-less records; the row must not lose them."""
        fixture = _Fixture(tmp_path)
        record = fixture.submit_task_less()
        fixture.resolve(record)
        fixture.absorb(record, 2)

        rc = main(['--queue-dir', str(fixture.queue.queue_dir)])

        out = capsys.readouterr().out
        assert rc == 0
        row_lines = [line for line in out.splitlines() if record.id in line]
        assert len(row_lines) == 1
        assert row_lines[0].split()[1] == '-'
        assert 'records=1' in _summary_tokens(out)

    def test_ill_typed_records_cost_one_row_each_not_the_table(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ):
        fixture = _Fixture(tmp_path)
        record = fixture.resolved_absorbing(1)
        fixture.write_by_hand(category=None)
        fixture.write_by_hand(refiles_suppressed=None)
        (fixture.queue.queue_dir / 'esc-list-1.json').write_text('[]')

        rc = main(['--queue-dir', str(fixture.queue.queue_dir)])

        out = capsys.readouterr().out
        assert rc == 0
        assert record.id in out
        assert {'records=1', 'scanned=1', 'skipped_unparsable=3'} <= _summary_tokens(out)

    def test_a_lost_record_is_printed_in_the_summary(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ):
        fixture = _Fixture(tmp_path)
        fixture.resolved_absorbing(1)
        (fixture.queue.queue_dir / 'esc-junk-1.json').write_text('this is not json')

        rc = main(['--queue-dir', str(fixture.queue.queue_dir)])

        assert rc == 0
        assert 'skipped_unparsable=1' in _summary_tokens(capsys.readouterr().out)


class TestCliJson:
    def test_json_carries_the_rows_and_the_shortfall(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ):
        fixture = _Fixture(tmp_path)
        fixture.resolved_absorbing(3)
        fixture.resolved_absorbing(1, category='recon_task_finding', citation_sha=None)
        (fixture.queue.queue_dir / 'esc-junk-1.json').write_text('this is not json')
        report = fixture.report()

        rc = main(['--queue-dir', str(fixture.queue.queue_dir), '--json'])

        payload = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert [
            (r['escalation_id'], r['refiles_suppressed'], r['category'], r['status'],
             r['citation_sha'])
            for r in payload['rows']
        ] == [
            (row.escalation_id, row.refiles_suppressed, row.category, row.status,
             row.citation_sha)
            for row in report.rows
        ]
        assert payload['total_absorbed'] == report.total_absorbed == 4
        assert payload['scanned'] == report.scanned
        assert payload['skipped_vanished'] == report.skipped_vanished
        assert payload['skipped_unreadable'] == report.skipped_unreadable
        assert payload['skipped_unparsable'] == report.skipped_unparsable == 1


class TestCliDegradesLoudly:
    def test_a_missing_queue_dir_exits_2_naming_it(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ):
        """A typo'd path must never read back as a clean empty listing."""
        missing = tmp_path / 'not-a-queue'

        rc = main(['--queue-dir', str(missing)])

        captured = capsys.readouterr()
        assert rc == 2
        assert str(missing) in captured.err
        assert 'total_absorbed' not in captured.out

    def test_the_documented_module_invocation_runs(self, tmp_path: Path):
        fixture = _Fixture(tmp_path)
        fixture.resolved_absorbing(2)

        completed = subprocess.run(
            [sys.executable, '-m', 'escalation.refile_absorption',
             '--queue-dir', str(fixture.queue.queue_dir)],
            capture_output=True, text=True, check=False,
        )

        assert completed.returncode == 0, completed.stderr
        assert 'total_absorbed=2' in _summary_tokens(completed.stdout)
