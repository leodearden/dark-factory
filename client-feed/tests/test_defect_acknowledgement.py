"""B32-B34 and B41: a publisher pass acknowledges each client defect report it reads on the inbox, and names a filed one held behind a deferred gate (plans/client-feed-prd.md §14.2 decisions 8 and 14, §14.6 B41)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import pytest
import yaml
from _defects import add_dependency, carrying, declared, gate_for, reported, ruled
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
    Degradation,
    FiledDefect,
    Heartbeat,
    RevealRequested,
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
HELD: Final = Degradation(
    source=Source.FILER, reason=f'1 defect report(s) waiting on a deferred gate past 600s: {FIX}'
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
        add_dependency(root.tasks, 6, 5)

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

    def test_it_clears_once_its_carrier_is_pending_while_the_inbox_is_unusable(
        self, root: Root
    ) -> None:
        append_inbox(root.in_dir, reported(2, 1, at=T))
        root.run(seconds(60))
        held = root.cursor()
        inbox = root.in_dir / INBOX_FILENAME
        inbox.unlink()
        inbox.mkdir()
        root.tasks.add(6, updated_at=UPDATED, metadata=carrying(FIX))

        report = root.run(seconds(GRACE + 1))

        assert Source.INBOX in sources(report)
        assert Source.FILER not in sources(report)
        cursor = root.cursor()
        assert (cursor.defect_scan_offset, cursor.pending_reports) == (held.defect_scan_offset, {})

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


def hold_behind_a_deferred_gate(root: Root, *, carrier_status: str = 'pending') -> None:
    """The fix released and its gate left deferred, as a commit_planning refused for the gate leaves them."""
    root.tasks.add(5, status='deferred', updated_at=UPDATED, metadata=gate_for(FIX))
    root.tasks.add(
        6, status=carrier_status, updated_at=UPDATED, metadata=carrying(FIX), depends_on=(5,)
    )


class TestAReportBehindADeferredGateIsLoud:
    def test_it_is_filed_gated_held_from_the_first_pass_and_named_past_the_grace(
        self, root: Root
    ) -> None:
        hold_behind_a_deferred_gate(root)

        first = root.run(T)

        assert [(f.ref, f.task_id, f.gated) for f in root.filed()] == [(FIX, 6, True)]
        assert Source.FILER not in sources(first)
        assert root.cursor().deferred_gate_reports == {FIX: T}

        overdue = root.run(seconds(GRACE + 1))

        assert HELD in overdue.degraded
        assert HELD in root.last_heartbeat().degraded
        assert UNFILED not in overdue.degraded
        assert len(root.filed()) == 1

    def test_it_clears_once_the_gate_is_released(self, root: Root) -> None:
        hold_behind_a_deferred_gate(root)
        root.run(T)
        assert HELD in root.run(seconds(GRACE + 1)).degraded
        root.tasks.set_status(5, 'pending', updated_at=LATER)

        report = root.run(seconds(GRACE + 2))

        assert Source.FILER not in sources(report)
        assert root.cursor().deferred_gate_reports == {}

    def test_it_is_still_named_while_the_carriers_cannot_be_read(self, root: Root) -> None:
        hold_behind_a_deferred_gate(root)
        root.run(T)
        tasks_db(root.policy.project_root).unlink()

        report = root.run(seconds(GRACE + 1))

        assert Source.TASKS in sources(report)
        assert HELD in report.degraded
        assert root.cursor().deferred_gate_reports == {FIX: T}

    @pytest.mark.parametrize('carrier_status', ['done', 'cancelled'])
    def test_a_finished_fix_behind_a_deferred_gate_is_never_named(
        self, root: Root, carrier_status: str
    ) -> None:
        hold_behind_a_deferred_gate(root, carrier_status=carrier_status)

        root.run(T)
        report = root.run(seconds(GRACE + 1))

        assert Source.FILER not in sources(report)
        assert root.cursor().deferred_gate_reports == {}

    def test_an_unfiled_report_filed_behind_a_deferred_gate_is_held_from_that_pass(
        self, root: Root
    ) -> None:
        append_inbox(root.in_dir, reported(2, 1, at=T))
        assert UNFILED in root.run(seconds(GRACE + 1)).degraded
        hold_behind_a_deferred_gate(root)

        filed = root.run(seconds(GRACE + 2))

        assert Source.FILER not in sources(filed)
        cursor = root.cursor()
        assert cursor.pending_reports == {}
        assert cursor.deferred_gate_reports == {FIX: seconds(GRACE + 2)}
        assert HELD in root.run(seconds(2 * GRACE + 3)).degraded

    def test_one_heartbeat_names_both_filer_reasons_and_the_feed_stays_valid(
        self, root: Root
    ) -> None:
        unfiled_elsewhere = Degradation(
            source=Source.FILER, reason='1 defect report(s) unfiled past 600s: portal:acme:3:4'
        )
        append_inbox(root.in_dir, reported(3, 4, at=T))
        hold_behind_a_deferred_gate(root)
        root.run(T)

        report = root.run(seconds(GRACE + 1))

        assert unfiled_elsewhere in report.degraded
        assert HELD in report.degraded
        heartbeat = root.last_heartbeat()
        assert unfiled_elsewhere in heartbeat.degraded
        assert HELD in heartbeat.degraded
        assert validate_out_dir(root.out_dir).ok

    def test_a_gate_held_again_after_its_release_is_named(self, root: Root) -> None:
        root.tasks.add(5, updated_at=UPDATED, metadata=gate_for(FIX))
        root.tasks.add(6, updated_at=UPDATED, metadata=carrying(FIX), depends_on=(5,))
        assert Source.FILER not in sources(root.run(T))
        root.tasks.set_status(5, 'deferred', updated_at=LATER)

        root.run(seconds(60))

        assert root.cursor().deferred_gate_reports == {FIX: seconds(60)}
        assert HELD in root.run(seconds(GRACE + 61)).degraded


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
