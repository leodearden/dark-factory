"""B32-B34: a publisher pass acknowledges each client defect report it reads on the inbox (plans/client-feed-prd.md §14.2 decisions 8 and 14)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import pytest
import yaml
from _in_dir import append_inbox, write_engagement
from _publisher_pass import REVEALABLE_PATH, inbox_offset, reveal_answers, reveal_policy
from _stores import TasksDb
from shared.project_layout import tasks_db

from client_feed.cli import main
from client_feed.contract import (
    ENGAGEMENT_FILENAME,
    INBOX_FILENAME,
    STATE_FILENAME,
    DefectFiled,
    DefectReported,
    DefectRuled,
    Degradation,
    FiledDefect,
    Heartbeat,
    Restated,
    Restater,
    RevealRequested,
    RulingKind,
    Source,
    parse_state,
)
from client_feed.feed_io import CURSOR_FILENAME, Cursor, OutDir, read_feed_lines
from client_feed.policy import Policy
from client_feed.publisher import PassReport, run_pass
from client_feed.validate import validate_out_dir

PROJECT: Final = 'acme'
T: Final = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
UPDATED: Final = '2026-10-21T11:00:00.000Z'
LATER: Final = '2026-10-21T11:30:00.000Z'
GRACE: Final = 600
FIX: Final = 'portal:acme:2:1'
UNFILED: Final = Degradation(
    source=Source.FILER, reason=f'1 defect report(s) unfiled past 600s: {FIX}'
)


def seconds(n: float) -> datetime:
    return T + timedelta(seconds=n)


@dataclass(frozen=True)
class Root:
    """A project with a usable engagement, its task store, and the in/ and out/ the publisher reads and writes."""

    policy: Policy
    tasks: TasksDb

    @property
    def in_dir(self) -> Path:
        return self.policy.in_dir

    @property
    def out_dir(self) -> Path:
        return self.policy.out_dir

    def run(self, now: datetime, *, rescan_tasks: bool = True) -> PassReport:
        out = OutDir.open(self.policy.out_dir, project_id=PROJECT)
        return run_pass(self.policy, out, now=now, rescan_tasks=rescan_tasks)

    def cursor(self) -> Cursor:
        return Cursor.model_validate_json((self.out_dir / CURSOR_FILENAME).read_text())

    def inbox_size(self) -> int:
        return (self.in_dir / INBOX_FILENAME).stat().st_size

    def filed(self) -> list[DefectFiled]:
        return [line for line in read_feed_lines(self.out_dir) if isinstance(line, DefectFiled)]

    def last_heartbeat(self) -> Heartbeat:
        return [line for line in read_feed_lines(self.out_dir) if isinstance(line, Heartbeat)][-1]


def make_root(tmp_path: Path) -> Root:
    in_dir = tmp_path / 'in'
    policy = reveal_policy(tmp_path, in_dir, project_id=PROJECT, committed_at=T - timedelta(days=1))
    write_engagement(in_dir, project_id=PROJECT)
    return Root(policy=policy, tasks=TasksDb(tasks_db(policy.project_root)))


@pytest.fixture
def root(tmp_path: Path) -> Root:
    return make_root(tmp_path)


def reported(block: int, report: int, *, at: datetime, seq: int = 1) -> DefectReported:
    return DefectReported(
        seq=seq,
        at=at,
        block=block,
        report=report,
        criterion_id='C-1',
        restater=Restater(prompt_version='restate-v1'),
        outcome=Restated(observed='o', expected='e', steps=('s',)),
    )


def ruled(block: int, report: int, *, at: datetime, seq: int) -> DefectRuled:
    return DefectRuled(seq=seq, at=at, block=block, report=report, kind=RulingKind.NOT_COVERED)


def declared(ref: str, *, origin: str = 'client') -> dict[str, object]:
    """One metadata.defects entry as declare_defect writes it."""
    return {
        'ref': ref,
        'origin': origin,
        'summary': 'SECRET-SUMMARY',
        'declared_at': '2026-10-21T11:00:00+00:00',
        'declared_by': 'client-defect-filer',
        'status': 'open',
        'sightings': 1,
    }


def carrying(ref: str) -> dict[str, object]:
    return {'defects': [declared(ref)]}


def gate_for(ref: str) -> dict[str, object]:
    return {'source': 'client-defect-filer', 'x_client_defect_gate_for': ref}


def add_edge(tasks: TasksDb, task_id: int, depends_on: int) -> None:
    """An edge inserted as fused-memory's same-project add_dependency inserts it: the task's updated_at untouched."""
    conn = sqlite3.connect(tasks.path)
    try:
        conn.execute(
            'INSERT INTO dependencies (tag, task_id, depends_on) VALUES (?, ?, ?)',
            ('master', task_id, depends_on),
        )
        conn.commit()
    finally:
        conn.close()


def sources(report: PassReport) -> list[Source]:
    return [degradation.source for degradation in report.degraded]


class TestACarrierIsFiled:
    def test_a_pending_carrier_is_filed_once_into_the_state(self, root: Root) -> None:
        root.tasks.add(6, updated_at=UPDATED, metadata=carrying(FIX))

        root.run(T)

        (filed,) = root.filed()
        assert (filed.ref, filed.block, filed.report, filed.task_id, filed.gated) == (
            FIX, 2, 1, 6, False,
        )
        state = parse_state((root.out_dir / STATE_FILENAME).read_text())
        assert state.defects_filed == {FIX: FiledDefect(block=2, report=1, task_id=6, gated=False)}
        assert validate_out_dir(root.out_dir).ok

    def test_a_gate_edge_files_it_again_gated_without_a_task_rescan(self, root: Root) -> None:
        root.tasks.add(6, updated_at=UPDATED, metadata=carrying(FIX))
        root.run(T)
        root.tasks.add(5, updated_at=LATER, metadata=gate_for(FIX))
        add_edge(root.tasks, 6, 5)

        root.run(seconds(60), rescan_tasks=False)

        assert [(f.ref, f.task_id, f.gated) for f in root.filed()] == [
            (FIX, 6, False),
            (FIX, 6, True),
        ]
        assert validate_out_dir(root.out_dir).ok

    def test_a_deferred_carrier_is_not_filed(self, root: Root) -> None:
        root.tasks.add(7, status='deferred', updated_at=UPDATED, metadata=carrying('portal:acme:2:2'))

        root.run(T)

        assert root.filed() == []

    def test_a_reopened_out_dir_files_nothing_again(self, root: Root) -> None:
        root.tasks.add(6, updated_at=UPDATED, metadata=carrying(FIX))
        root.run(T)

        root.run(seconds(60))
        root.run(seconds(120))

        assert len(root.filed()) == 1


class TestAnUnfiledReportIsLoud:
    def test_it_waits_in_the_cursor_and_is_named_once_past_the_grace(self, root: Root) -> None:
        line = reported(2, 1, at=T)
        append_inbox(root.in_dir, line)

        young = root.run(seconds(60))

        assert Source.FILER not in sources(young)
        cursor = root.cursor()
        assert cursor.pending_reports == {FIX: line.at}
        assert cursor.defect_scan_offset == root.inbox_size()

        overdue = root.run(seconds(GRACE + 1))

        assert UNFILED in overdue.degraded
        assert UNFILED in root.last_heartbeat().degraded

    def test_it_is_still_named_while_the_engagement_is_unusable(self, root: Root) -> None:
        append_inbox(root.in_dir, reported(2, 1, at=T))
        root.run(seconds(60))
        (root.in_dir / ENGAGEMENT_FILENAME).write_text('{not json')

        report = root.run(seconds(GRACE + 1))

        assert Source.ENGAGEMENT in sources(report)
        assert UNFILED in report.degraded

    def test_it_is_still_named_when_only_a_deferred_task_holds_it(self, root: Root) -> None:
        root.tasks.add(6, status='deferred', updated_at=UPDATED, metadata=carrying(FIX))
        append_inbox(root.in_dir, reported(2, 1, at=T))

        report = root.run(seconds(GRACE + 1))

        assert UNFILED in report.degraded

    def test_it_clears_once_its_carrier_is_pending(self, root: Root) -> None:
        append_inbox(root.in_dir, reported(2, 1, at=T))
        assert UNFILED in root.run(seconds(GRACE + 1)).degraded
        root.tasks.add(6, updated_at=UPDATED, metadata=carrying(FIX))

        report = root.run(seconds(GRACE + 2))

        assert Source.FILER not in sources(report)
        assert root.cursor().pending_reports == {}
        assert [(f.ref, f.task_id) for f in root.filed()] == [(FIX, 6)]
        assert root.filed()[0].seq > root.last_heartbeat().watermark_seq

    def test_it_is_still_named_while_the_carriers_cannot_be_read(self, root: Root) -> None:
        append_inbox(root.in_dir, reported(2, 1, at=T))
        tasks_db(root.policy.project_root).unlink()

        report = root.run(seconds(GRACE + 1))

        assert UNFILED in report.degraded
        assert root.cursor().pending_reports == {FIX: T}

    def test_a_ruled_report_is_never_named(self, root: Root) -> None:
        append_inbox(
            root.in_dir, reported(2, 1, at=T, seq=1), ruled(2, 1, at=seconds(5), seq=2)
        )

        root.run(seconds(60))

        assert root.cursor().pending_reports == {}
        assert Source.FILER not in sources(root.run(seconds(GRACE + 1)))


def test_with_reveals_disabled_from_the_start_the_defect_scan_still_reaches_the_inbox_end(
    root: Root,
) -> None:
    (root.in_dir / ENGAGEMENT_FILENAME).unlink()
    append_inbox(root.in_dir, reported(2, 1, at=T))

    report = root.run(seconds(60))

    assert Source.ENGAGEMENT in sources(report)
    cursor = root.cursor()
    assert cursor.defect_scan_offset == root.inbox_size()
    assert cursor.inbox_offset == 0


def test_reveal_requests_and_a_defect_report_interleave(root: Root) -> None:
    append_inbox(
        root.in_dir,
        RevealRequested(seq=1, at=T, path=REVEALABLE_PATH),
        reported(2, 1, at=T, seq=2),
        RevealRequested(seq=3, at=T, path='src/missing.py'),
    )

    report = root.run(seconds(60))

    assert [answer.path for answer in reveal_answers(root.out_dir)] == [
        REVEALABLE_PATH,
        'src/missing.py',
    ]
    assert Source.INBOX not in sources(report)
    assert inbox_offset(root.out_dir) == root.cursor().defect_scan_offset == root.inbox_size()
    assert set(root.cursor().pending_reports) == {FIX}


def test_refs_of_other_origins_and_projects_never_reach_the_out_dir(root: Root) -> None:
    review = declared('review:7:0123456789ab', origin='review')
    root.tasks.add(5, updated_at=UPDATED, metadata={'defects': [review]})
    root.tasks.add(6, updated_at=UPDATED, metadata=carrying('portal:other-client:2:1'))
    root.tasks.add(7, updated_at=UPDATED, metadata=gate_for('portal:other-client:2:1'))

    root.run(T)

    for written in root.out_dir.iterdir():
        data = written.read_bytes()
        assert b'review:7:0123456789ab' not in data, written.name
        assert b'other-client' not in data, written.name


class TestOneSourceIsNamedOnce:
    def test_a_missing_task_store_is_named_tasks_once(self, root: Root) -> None:
        tasks_db(root.policy.project_root).unlink()

        report = root.run(T)

        assert sources(report).count(Source.TASKS) == 1

    def test_an_unusable_inbox_is_named_inbox_once_and_the_defect_scan_holds(
        self, root: Root
    ) -> None:
        append_inbox(root.in_dir, reported(2, 1, at=T))
        root.run(seconds(60))
        held = root.cursor()
        inbox = root.in_dir / INBOX_FILENAME
        inbox.unlink()
        inbox.mkdir()

        report = root.run(seconds(120))

        assert sources(report).count(Source.INBOX) == 1
        cursor = root.cursor()
        assert (cursor.defect_scan_offset, cursor.pending_reports) == (
            held.defect_scan_offset,
            held.pending_reports,
        )


def write_config(tmp_path: Path, policy: Policy) -> Path:
    config = tmp_path / 'policy.yaml'
    config.write_text(yaml.safe_dump(policy.model_dump(mode='json', exclude_unset=True)))
    return config


class TestTheSnapshotCommand:
    def test_it_files_a_reported_defect_its_carrier_holds(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = make_root(tmp_path)
        append_inbox(root.in_dir, reported(2, 1, at=T))
        root.tasks.add(6, updated_at=UPDATED, metadata=carrying(FIX))

        assert main(['snapshot', '--config', str(write_config(tmp_path, root.policy))]) == 0

        (filed,) = root.filed()
        state = parse_state((root.out_dir / STATE_FILENAME).read_text())
        assert state.defects_filed == {
            FIX: FiledDefect(
                block=filed.block, report=filed.report, task_id=filed.task_id, gated=filed.gated
            )
        }
        assert root.cursor().pending_reports == {}
        assert main(['validate-feed', '--out-dir', str(root.out_dir)]) == 0

    def test_it_names_the_filer_for_a_report_unfiled_past_the_grace(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = make_root(tmp_path)
        append_inbox(root.in_dir, reported(2, 1, at=datetime(2026, 10, 21, 9, 0, tzinfo=UTC)))

        assert main(['snapshot', '--config', str(write_config(tmp_path, root.policy))]) == 0

        printed = PassReport.model_validate(json.loads(capsys.readouterr().out))
        assert UNFILED in printed.degraded
        assert main(['validate-feed', '--out-dir', str(root.out_dir)]) == 0
