"""Tests for escalation.watcher — ntfy push and loop behavior."""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from inotify_simple import Event, flags

from escalation.models import Escalation
from escalation.watcher import (
    _initial_scan,
    armed_source,
    emit,
    exclusions,
    main,
    ntfy_request,
    watch,
)


@pytest.fixture
def blocking_escalation() -> Escalation:
    return Escalation(
        id='esc-42-1',
        task_id='42',
        agent_role='orchestrator',
        severity='blocking',
        category='task_failure',
        summary='Verification attempts exhausted',
        detail='Verification attempts exhausted — tests failed 3 times',
        suggested_action='investigate_and_retry',
        worktree='/tmp/worktrees/42',
        workflow_state='blocked',
    )


@pytest.fixture
def info_escalation() -> Escalation:
    return Escalation(
        id='esc-42-2',
        task_id='42',
        agent_role='implementer',
        severity='info',
        category='design_concern',
        summary='API shape differs from convention',
    )


@pytest.fixture
def critical_escalation() -> Escalation:
    return Escalation(
        id='esc-99-1',
        task_id='99',
        agent_role='orchestrator',
        severity='critical',
        category='infra_issue',
        summary='Production database unreachable',
        detail='Connection refused on all replicas',
    )


@pytest.fixture
def urgent_escalation() -> Escalation:
    return Escalation(
        id='esc-99-2',
        task_id='99',
        agent_role='steward',
        severity='urgent',
        category='risk_identified',
        summary='Data loss imminent',
    )


class TestNtfyRequest:
    """ntfy_request builds the POST with the correct headers and body."""

    def test_blocking_escalation_sends_urgent(self, blocking_escalation: Escalation):
        req = ntfy_request('https://ntfy.sh/test-topic', blocking_escalation)

        assert req.full_url == 'https://ntfy.sh/test-topic'
        assert req.get_method() == 'POST'
        assert req.get_header('Title') == '[BLOCKING] Task 42: task_failure'
        assert req.get_header('Priority') == 'urgent'
        assert req.get_header('Tags') == 'rotating_light'
        assert isinstance(req.data, bytes)
        assert b'Verification attempts exhausted' in req.data

    def test_info_escalation_sends_default_priority(self, info_escalation: Escalation):
        req = ntfy_request('https://ntfy.sh/test-topic', info_escalation)

        assert req.get_header('Title') == '[INFO] Task 42: design_concern'
        assert req.get_header('Priority') == 'default'
        assert req.get_header('Tags') == 'information_source'

    def test_body_includes_detail(self, blocking_escalation: Escalation):
        req = ntfy_request('https://ntfy.sh/test-topic', blocking_escalation)

        assert isinstance(req.data, bytes)
        assert 'tests failed 3 times' in req.data.decode('utf-8')

    def test_body_omits_duplicate_detail(self):
        """When detail == summary, don't repeat it."""
        esc = Escalation(
            id='esc-1-1',
            task_id='1',
            agent_role='orchestrator',
            severity='info',
            category='task_failure',
            summary='Planning failed',
            detail='Planning failed',
        )
        req = ntfy_request('https://ntfy.sh/t', esc)

        assert isinstance(req.data, bytes)
        assert req.data.decode('utf-8') == 'Planning failed'

    def test_critical_escalation_sends_urgent(self, critical_escalation: Escalation):
        """critical severity -> [CRITICAL] title, Priority=urgent, Tags=rotating_light."""
        req = ntfy_request('https://ntfy.sh/test-topic', critical_escalation)

        assert req.get_header('Title') == '[CRITICAL] Task 99: infra_issue'
        assert req.get_header('Priority') == 'urgent'
        assert req.get_header('Tags') == 'rotating_light'

    def test_urgent_escalation_sends_urgent(self, urgent_escalation: Escalation):
        """urgent severity -> [URGENT] title, Priority=urgent, Tags=rotating_light."""
        req = ntfy_request('https://ntfy.sh/test-topic', urgent_escalation)

        assert req.get_header('Title') == '[URGENT] Task 99: risk_identified'
        assert req.get_header('Priority') == 'urgent'
        assert req.get_header('Tags') == 'rotating_light'


def _outage(url: str, esc: Escalation) -> None:
    raise RuntimeError('simulated ntfy outage')


class _RecordingSender:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, url: str, esc: Escalation) -> None:
        self.calls.append((url, esc.id))


class TestEmitNtfyFailure:
    """A dropped push must be countable, and must cost the operator nothing else.

    `emit` once reported an ntfy failure with a bare `print(..., file=sys.stderr)`
    — watcher.py was the one module in the package with no logger, so the
    failure never reached the stream the watcher's other failures travel, and
    carried no marker an operator or skill agent could grep for.  Because
    `emit` runs at most once per process (its one call site in main exits on
    the next line), the only countable unit is the marker line itself,
    aggregated across the re-arm loop's stderr.
    """

    def test_failure_logs_the_marker_at_error(self, blocking_escalation: Escalation, caplog):
        """(a) ERROR on escalation.watcher, carrying marker + id + cause."""
        with caplog.at_level(logging.ERROR, logger='escalation.watcher'):
            emit(blocking_escalation, 'https://ntfy.sh/t', send=_outage)

        matching = [
            r for r in caplog.records
            if r.levelno >= logging.ERROR
            and 'WATCHER_NTFY_OUTCOME: FAILED' in r.getMessage()
        ]
        assert matching, (
            'Expected an ERROR carrying the WATCHER_NTFY_OUTCOME marker; got: '
            f'{[(r.name, r.levelname, r.getMessage()) for r in caplog.records]}'
        )
        message = matching[0].getMessage()
        assert matching[0].name == 'escalation.watcher', (
            f'Marker logged on the wrong logger: {matching[0].name!r}'
        )
        assert blocking_escalation.id in message, (
            f'Marker must name the escalation whose push was dropped: {message!r}'
        )
        assert 'simulated ntfy outage' in message, (
            f'Marker must carry the underlying cause: {message!r}'
        )

    def test_failure_is_logged_not_bare_printed(
        self, blocking_escalation: Escalation, capsys,
    ):
        """(b) The failure travels the logging stream, not a bare print.

        The empty direct-stderr read is the ABSENCE OF A BARE PRINT, not a
        claim that the marker never reaches stderr — under `caplog` the record
        is captured by a handler, and in production `main` configures one that
        writes to stderr.  `TestMarkerReachesRealStderr` below pins that
        delivery, in a real child process, where it can actually be observed.
        """
        emit(blocking_escalation, 'https://ntfy.sh/t', send=_outage)

        assert capsys.readouterr().err == '', (
            'The failure must go through logging, not a bare print to stderr'
        )

    def test_stdout_still_carries_the_escalation_json(
        self, blocking_escalation: Escalation, capsys,
    ):
        """(c) A dropped push never costs the operator the queue item.

        stdout must also stay PURE JSON: the skill tells agents to parse it
        without `2>&1`, so a diagnostic leaking there would break every
        consumer.
        """
        emit(blocking_escalation, 'https://ntfy.sh/t', send=_outage)

        parsed = json.loads(capsys.readouterr().out)
        assert parsed['id'] == blocking_escalation.id, f'Unexpected stdout payload: {parsed}'
        assert parsed['summary'] == blocking_escalation.summary

    def test_failure_does_not_raise(self, blocking_escalation: Escalation):
        """(d) The push is best-effort — its failure never propagates."""
        emit(blocking_escalation, 'https://ntfy.sh/t', send=_outage)  # must not raise

    def test_successful_push_logs_no_error(self, blocking_escalation: Escalation, caplog):
        """(d) A working push raises no false outage signal.

        Only a FAILED line is emitted, never a SENT companion: the marker is
        the countable unit of an OUTAGE, so a success line carrying the same
        prefix would make every count of it wrong.
        """
        sender = _RecordingSender()
        with caplog.at_level(logging.DEBUG):
            emit(blocking_escalation, 'https://ntfy.sh/t', send=sender)

        assert sender.calls == [('https://ntfy.sh/t', blocking_escalation.id)], (
            f'The push must be sent exactly once to the given URL: {sender.calls}'
        )
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR], (
            f'A successful push must log no ERROR; got: '
            f'{[(r.levelname, r.getMessage()) for r in caplog.records]}'
        )
        assert not [
            r for r in caplog.records if 'WATCHER_NTFY_OUTCOME' in r.getMessage()
        ], 'The marker must mean a DROPPED push and nothing else'

    def test_no_url_sends_nothing(self, blocking_escalation: Escalation, capsys):
        """Without --ntfy-url the escalation is still printed, and no push is attempted."""
        sender = _RecordingSender()
        emit(blocking_escalation, None, send=sender)

        assert sender.calls == []
        assert json.loads(capsys.readouterr().out)['id'] == blocking_escalation.id


class TestMarkerReachesRealStderr:
    """The stderr promise holds because `main` configures logging, not by luck.

    Every test above proves only that the marker is LOGGED: `caplog` attaches
    its own handler, which is also what detaches `logging.lastResort`.  What
    invariant (c) and `skills/escalation-watcher/SKILL.md` promise an operator
    is a greppable line on REAL stderr from a plain `python -m
    escalation.watcher` run — a promise that, while it rested on
    `logging.lastResort`, would have evaporated unnoticed the moment anything
    configured a root handler in-process.  Only a child process can observe
    that, so exactly one test pays for one.

    The ntfy URL carries a scheme no opener handles, so the push fails inside
    `urlopen` before any socket is opened: the drop is deterministic and the
    test needs no network.
    """

    def test_dropped_push_marks_stderr_and_leaves_stdout_pure(self, tmp_path):
        repo_root = Path(__file__).resolve().parents[2]
        queue_dir = tmp_path / 'queue'
        queue_dir.mkdir()
        esc = Escalation(
            id='esc-5368-1', task_id='5368', agent_role='implementer',
            severity='blocking', category='infra_issue',
            summary='a filing whose push we then drop',
        )
        _write_esc(queue_dir, esc)

        proc = subprocess.run(
            [
                sys.executable, '-m', 'escalation.watcher',
                '--queue-dir', str(queue_dir),
                '--ntfy-url', 'ntfy-outage://unreachable',
                '--timeout', '30',
            ],
            capture_output=True, text=True, timeout=120,
            env={
                **os.environ,
                'PYTHONPATH': os.pathsep.join([
                    str(repo_root / 'escalation' / 'src'),
                    str(repo_root / 'shared' / 'src'),
                ]),
            },
        )

        assert proc.returncode == 0, (
            f'A dropped push must not change the exit code: rc={proc.returncode} '
            f'stderr={proc.stderr!r}'
        )
        marker_line = next(
            (ln for ln in proc.stderr.splitlines() if 'WATCHER_NTFY_OUTCOME: FAILED' in ln),
            None,
        )
        assert marker_line is not None, (
            f'The marker never reached real stderr: {proc.stderr!r}'
        )
        assert 'ERROR' in marker_line, (
            'The line must name its severity, which is also the evidence that this '
            'process configured its own handler: `logging.lastResort` emits the bare '
            f'message with no level at all.  Got: {marker_line!r}'
        )
        assert esc.id in proc.stderr, (
            f'The marker must name the escalation whose push was dropped: {proc.stderr!r}'
        )
        assert json.loads(proc.stdout)['id'] == esc.id, (
            f'stdout must stay pure escalation JSON: {proc.stdout!r}'
        )


def _write_esc(queue_dir, esc: Escalation) -> None:
    """Helper: write an escalation JSON file named by its id."""
    path = queue_dir / f'{esc.id}.json'
    path.write_text(esc.to_json())


class TestInitialScan:
    """_initial_scan(queue_dir, task_id, level) -> Escalation|None."""

    def test_returns_oldest_matching_pending(self, tmp_path):
        """Among multiple matching pending files, return the OLDEST by timestamp."""
        old_ts = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        new_ts = datetime.now(UTC).isoformat()
        old_esc = Escalation(
            id='esc-1-1', task_id='1', agent_role='orchestrator',
            severity='blocking', category='task_failure', summary='old',
            timestamp=old_ts,
        )
        new_esc = Escalation(
            id='esc-1-2', task_id='1', agent_role='orchestrator',
            severity='blocking', category='task_failure', summary='new',
            timestamp=new_ts,
        )
        queue_dir = tmp_path / 'queue'
        queue_dir.mkdir()
        _write_esc(queue_dir, old_esc)
        _write_esc(queue_dir, new_esc)

        result = _initial_scan(queue_dir, task_id=None, level=None)
        assert result is not None
        assert result.id == 'esc-1-1'

    def test_returns_none_on_empty_dir(self, tmp_path):
        """Returns None when queue dir is empty."""
        queue_dir = tmp_path / 'queue'
        queue_dir.mkdir()
        assert _initial_scan(queue_dir, task_id=None, level=None) is None

    def test_skips_non_pending(self, tmp_path):
        """Escalations with status != pending are skipped."""
        queue_dir = tmp_path / 'queue'
        queue_dir.mkdir()
        resolved = Escalation(
            id='esc-2-1', task_id='2', agent_role='steward',
            severity='info', category='design_concern', summary='resolved',
            status='resolved',
        )
        dismissed = Escalation(
            id='esc-2-2', task_id='2', agent_role='steward',
            severity='info', category='design_concern', summary='dismissed',
            status='dismissed',
        )
        _write_esc(queue_dir, resolved)
        _write_esc(queue_dir, dismissed)
        assert _initial_scan(queue_dir, task_id=None, level=None) is None

    def test_respects_task_id_filter(self, tmp_path):
        """Non-matching task_id -> None."""
        queue_dir = tmp_path / 'queue'
        queue_dir.mkdir()
        esc = Escalation(
            id='esc-3-1', task_id='3', agent_role='orchestrator',
            severity='blocking', category='task_failure', summary='fail',
        )
        _write_esc(queue_dir, esc)
        assert _initial_scan(queue_dir, task_id='99', level=None) is None

    def test_task_id_match_returns_escalation(self, tmp_path):
        """Matching task_id -> returns the escalation."""
        queue_dir = tmp_path / 'queue'
        queue_dir.mkdir()
        esc = Escalation(
            id='esc-4-1', task_id='4', agent_role='orchestrator',
            severity='blocking', category='task_failure', summary='fail',
        )
        _write_esc(queue_dir, esc)
        result = _initial_scan(queue_dir, task_id='4', level=None)
        assert result is not None
        assert result.id == 'esc-4-1'

    def test_respects_level_filter(self, tmp_path):
        """Non-matching level -> None; matching level -> returns escalation."""
        queue_dir = tmp_path / 'queue'
        queue_dir.mkdir()
        esc_l0 = Escalation(
            id='esc-5-1', task_id='5', agent_role='orchestrator',
            severity='blocking', category='task_failure', summary='l0', level=0,
        )
        esc_l1 = Escalation(
            id='esc-5-2', task_id='5', agent_role='steward',
            severity='blocking', category='task_failure', summary='l1', level=1,
        )
        _write_esc(queue_dir, esc_l0)
        _write_esc(queue_dir, esc_l1)
        result = _initial_scan(queue_dir, task_id=None, level=1)
        assert result is not None
        assert result.id == 'esc-5-2'

    def test_tolerates_malformed_json(self, tmp_path):
        """Malformed .json file is skipped; valid match is still returned."""
        queue_dir = tmp_path / 'queue'
        queue_dir.mkdir()
        # Write a garbage file
        (queue_dir / 'esc-garbage.json').write_text('{not valid json}}}')
        # Write a valid escalation
        esc = Escalation(
            id='esc-6-1', task_id='6', agent_role='orchestrator',
            severity='blocking', category='task_failure', summary='valid',
        )
        _write_esc(queue_dir, esc)
        result = _initial_scan(queue_dir, task_id=None, level=None)
        assert result is not None
        assert result.id == 'esc-6-1'

    def test_malformed_json_is_warned_not_silent(self, tmp_path, caplog):
        """A corrupt record is skipped AND logged at WARNING naming the file."""
        queue_dir = tmp_path / 'queue'
        queue_dir.mkdir()
        (queue_dir / 'esc-garbage.json').write_text('{not valid json}}}')
        with caplog.at_level(logging.WARNING):
            assert _initial_scan(queue_dir, task_id=None, level=None) is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            'watcher._initial_scan' in r.getMessage() and 'esc-garbage.json' in r.getMessage()
            for r in warnings
        ), [(r.levelname, r.getMessage()) for r in caplog.records]

    def test_undecodable_file_is_warned_and_scan_continues(self, tmp_path, caplog):
        """A non-UTF-8 record must not crash the scan; a valid sibling is still found."""
        queue_dir = tmp_path / 'queue'
        queue_dir.mkdir()
        (queue_dir / 'esc-binary.json').write_bytes(b'\xff\xfe\x00\x80')
        esc = Escalation(
            id='esc-6-2', task_id='6', agent_role='orchestrator',
            severity='blocking', category='task_failure', summary='valid',
        )
        _write_esc(queue_dir, esc)
        with caplog.at_level(logging.WARNING):
            result = _initial_scan(queue_dir, task_id=None, level=None)
        assert result is not None
        assert result.id == 'esc-6-2'
        assert any(
            r.levelno == logging.WARNING and 'esc-binary.json' in r.getMessage()
            for r in caplog.records
        ), [(r.levelname, r.getMessage()) for r in caplog.records]

    def test_unreadable_file_is_warned(self, tmp_path, caplog):
        """A permission-denied record is skipped AND logged at WARNING."""
        queue_dir = tmp_path / 'queue'
        queue_dir.mkdir()
        (queue_dir / 'esc-denied.json').write_text('{}')
        real_read_text = Path.read_text

        def deny(self, *a, **kw):
            if self.name == 'esc-denied.json':
                raise PermissionError(13, 'Permission denied')
            return real_read_text(self, *a, **kw)

        with patch.object(Path, 'read_text', deny), caplog.at_level(logging.WARNING):
            assert _initial_scan(queue_dir, task_id=None, level=None) is None
        assert any(
            r.levelno == logging.WARNING and 'esc-denied.json' in r.getMessage()
            for r in caplog.records
        ), [(r.levelname, r.getMessage()) for r in caplog.records]

    def test_exclude_single_pending(self, tmp_path):
        """Sole pending escalation in exclude_ids -> returns None."""
        queue_dir = tmp_path / 'queue'
        queue_dir.mkdir()
        esc = Escalation(
            id='esc-7-1', task_id='7', agent_role='orchestrator',
            severity='blocking', category='task_failure', summary='pending',
        )
        _write_esc(queue_dir, esc)
        result = _initial_scan(queue_dir, task_id=None, level=None, exclude_ids={'esc-7-1'})
        assert result is None

    def test_exclude_older_returns_newer(self, tmp_path):
        """When the older of two pending escalations is excluded, returns the newer one."""
        old_ts = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        new_ts = datetime.now(UTC).isoformat()
        old_esc = Escalation(
            id='esc-8-1', task_id='8', agent_role='orchestrator',
            severity='blocking', category='task_failure', summary='old',
            timestamp=old_ts,
        )
        new_esc = Escalation(
            id='esc-8-2', task_id='8', agent_role='orchestrator',
            severity='blocking', category='task_failure', summary='new',
            timestamp=new_ts,
        )
        queue_dir = tmp_path / 'queue'
        queue_dir.mkdir()
        _write_esc(queue_dir, old_esc)
        _write_esc(queue_dir, new_esc)
        result = _initial_scan(queue_dir, task_id=None, level=None, exclude_ids={'esc-8-1'})
        assert result is not None
        assert result.id == 'esc-8-2'

    def test_exclude_all_pending_returns_none(self, tmp_path):
        """All pending escalations in exclude_ids -> returns None."""
        queue_dir = tmp_path / 'queue'
        queue_dir.mkdir()
        esc1 = Escalation(
            id='esc-9-1', task_id='9', agent_role='orchestrator',
            severity='blocking', category='task_failure', summary='a',
        )
        esc2 = Escalation(
            id='esc-9-2', task_id='9', agent_role='orchestrator',
            severity='blocking', category='task_failure', summary='b',
        )
        _write_esc(queue_dir, esc1)
        _write_esc(queue_dir, esc2)
        result = _initial_scan(queue_dir, task_id=None, level=None, exclude_ids={'esc-9-1', 'esc-9-2'})
        assert result is None

    def test_find_pending_corrupt_timestamp_warns(self, tmp_path, caplog):
        """Corrupt timestamp on a matching pending escalation → still returned AND a WARNING emitted.

        Bug in current code: silent datetime.min substitution produces no log output.
        The entry is still considered (datetime.min sorts oldest, included), but
        operators cannot see the data-quality issue.

        Fix: parse_timestamp_or_warn emits a WARNING mentioning the context string.
        RED assertion: zero warnings → assertion (b) fails.
        """
        queue_dir = tmp_path / 'queue'
        queue_dir.mkdir()

        esc = Escalation(
            id='esc-corrupt-1',
            task_id='99',
            agent_role='orchestrator',
            severity='blocking',
            category='task_failure',
            summary='corrupt timestamp test',
        )
        esc.timestamp = 'not-a-timestamp'
        _write_esc(queue_dir, esc)

        with caplog.at_level(logging.WARNING, logger='shared.timestamps'):
            result = _initial_scan(queue_dir, task_id=None, level=None)

        # (a) Entry still CONSIDERED — datetime.min sorts oldest, included in results
        assert result is not None, 'Expected corrupt-ts escalation to still be returned'
        assert result.id == 'esc-corrupt-1', (
            f'Expected esc-corrupt-1; got result.id={result.id!r}'
        )

        # (b) A WARNING must be emitted mentioning the context tag
        warning_records = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and 'watcher._initial_scan' in r.message
        ]
        assert warning_records, (
            f"Expected >=1 WARNING mentioning 'watcher._initial_scan'; "
            f"got caplog.records: {[(r.levelname, r.message) for r in caplog.records]}"
        )


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, secs: float) -> None:
        self.now += secs


class ScriptedSource:
    """An EventSource whose reads replay scripted batches.

    Each batch runs at read time, so it can write arrivals into the queue dir
    or move the clock, and returns the filenames to report as MOVED_TO events.
    """

    def __init__(self, batches: list[Callable[[], list[str]]]) -> None:
        self._batches = list(batches)
        self.reads: list[int | None] = []

    def read(self, timeout: int | None = None) -> list[Event]:
        self.reads.append(timeout)
        if not self._batches:
            if timeout is None:
                raise AssertionError('would block forever')
            return []
        names = self._batches.pop(0)()
        return [Event(wd=1, mask=flags.MOVED_TO, cookie=0, name=n) for n in names]


def _arrive(queue_dir: Path, esc: Escalation) -> str:
    """File esc the way every queue writer does: tmp-file + rename (invariant (a))."""
    name = f'{esc.id}.json'
    tmp = queue_dir / f'.{name}.tmp'
    tmp.write_text(esc.to_json())
    os.rename(tmp, queue_dir / name)
    return name


def _watch(queue_dir: Path, source: ScriptedSource, **overrides) -> Escalation | None:
    kwargs: dict = {
        'task_id': None,
        'level': None,
        'timeout': None,
        'current_excludes': lambda: frozenset(),
        'clock': FakeClock(),
    }
    kwargs.update(overrides)
    return watch(queue_dir, source, **kwargs)


def _esc(esc_id: str, **fields) -> Escalation:
    defaults: dict = {
        'task_id': esc_id.split('-')[1],
        'agent_role': 'orchestrator',
        'severity': 'blocking',
        'category': 'task_failure',
        'summary': esc_id,
    }
    defaults.update(fields)
    return Escalation(id=esc_id, **defaults)


@pytest.fixture
def queue_dir(tmp_path: Path) -> Path:
    path = tmp_path / 'queue'
    path.mkdir()
    return path


class TestWatch:
    """watch() returns the first pending match, scanning first and then polling."""

    def test_pending_on_disk_is_returned_without_reading(self, queue_dir: Path):
        esc = _esc('esc-50-1')
        _write_esc(queue_dir, esc)
        source = ScriptedSource([])

        result = _watch(queue_dir, source)

        assert result is not None and result.id == esc.id
        assert source.reads == []

    def test_arrival_on_first_read_is_returned(self, queue_dir: Path):
        esc = _esc('esc-51-1')
        source = ScriptedSource([lambda: [_arrive(queue_dir, esc)]])

        result = _watch(queue_dir, source)

        assert result is not None and result.id == esc.id
        assert source.reads == [None]

    def test_non_escalation_names_are_skipped(self, queue_dir: Path):
        """Names failing the esc-*.json gate are never read, even if they hold a valid record."""
        decoy = _esc('esc-52-9', summary='decoy behind a non-escalation name')
        for name in ('unrelated.json', 'x.tmp'):
            (queue_dir / name).write_text(decoy.to_json())
        esc = _esc('esc-52-1')
        source = ScriptedSource([
            lambda: ['unrelated.json', 'x.tmp', ''],
            lambda: [_arrive(queue_dir, esc)],
        ])

        result = _watch(queue_dir, source)

        assert result is not None and result.id == esc.id
        assert len(source.reads) == 2

    def test_excluded_arrival_is_delivered_then_dropped(self, queue_dir: Path):
        esc = _esc('esc-53-1')
        source = ScriptedSource([lambda: [_arrive(queue_dir, esc)]])

        result = _watch(
            queue_dir, source, timeout=5, current_excludes=lambda: frozenset({esc.id}),
        )

        assert result is None
        assert source.reads == [5000, 5000]

    def test_non_excluded_sibling_of_excluded_arrival_is_returned(self, queue_dir: Path):
        excluded = _esc('esc-54-1')
        live = _esc('esc-54-2')
        source = ScriptedSource([
            lambda: [_arrive(queue_dir, excluded), _arrive(queue_dir, live)],
        ])

        result = _watch(queue_dir, source, current_excludes=lambda: frozenset({excluded.id}))

        assert result is not None and result.id == live.id

    def test_excludes_are_reevaluated_each_poll(self, queue_dir: Path):
        """An id excluded after the scan but before its event is processed is suppressed."""
        esc = _esc('esc-55-1')
        excluded: set[str] = set()

        def exclude_then_arrive() -> list[str]:
            excluded.add(esc.id)
            return [_arrive(queue_dir, esc)]

        source = ScriptedSource([exclude_then_arrive])

        result = _watch(
            queue_dir, source, timeout=5, current_excludes=lambda: frozenset(excluded),
        )

        assert result is None

    def test_id_appended_to_exclude_file_mid_wait_is_suppressed(
        self, queue_dir: Path, tmp_path: Path,
    ):
        esc = _esc('esc-80-1')
        exclude_file = tmp_path / 'excludes.txt'
        exclude_file.write_text('')

        def append_then_arrive() -> list[str]:
            with exclude_file.open('a') as f:
                f.write(f'{esc.id}\n')
            return [_arrive(queue_dir, esc)]

        source = ScriptedSource([append_then_arrive])

        result = _watch(
            queue_dir, source, timeout=5,
            current_excludes=exclusions(
                static=frozenset(), baseline=frozenset(), exclude_file=exclude_file,
            ),
        )

        assert result is None
        assert source.reads == [5000, 5000]

    def test_arrival_absent_from_baseline_is_returned(self, queue_dir: Path):
        baselined = _esc('esc-81-1')
        new = _esc('esc-81-2')
        _write_esc(queue_dir, baselined)
        source = ScriptedSource([lambda: [_arrive(queue_dir, new)]])

        result = _watch(
            queue_dir, source,
            current_excludes=exclusions(
                static=frozenset(), baseline=frozenset({baselined.id}), exclude_file=None,
            ),
        )

        assert result is not None and result.id == new.id
        assert source.reads == [None]

    @pytest.mark.parametrize(
        ('fields', 'filters', 'returned'),
        [
            pytest.param({'status': 'resolved'}, {}, False, id='not-pending'),
            pytest.param({'level': 1}, {'level': 0}, False, id='level-mismatch'),
            pytest.param({'level': 1}, {'level': 1}, True, id='level-match'),
            pytest.param({'task_id': '99'}, {'task_id': '42'}, False, id='task-mismatch'),
            pytest.param({'task_id': '42'}, {'task_id': '42'}, True, id='task-match'),
        ],
    )
    def test_arrival_must_be_pending_and_pass_the_filters(
        self, queue_dir: Path, fields: dict, filters: dict, returned: bool,
    ):
        esc = _esc('esc-56-1', **fields)
        source = ScriptedSource([lambda: [_arrive(queue_dir, esc)]])

        result = _watch(queue_dir, source, timeout=5, **filters)

        assert (result is not None and result.id == esc.id) is returned

    def test_undecodable_arrival_is_warned_and_sibling_returned(
        self, queue_dir: Path, caplog: pytest.LogCaptureFixture,
    ):
        esc = _esc('esc-57-1')

        def binary_then_valid() -> list[str]:
            (queue_dir / 'esc-binary.json').write_bytes(b'\xff\xfe\x00\x80')
            return ['esc-binary.json', _arrive(queue_dir, esc)]

        source = ScriptedSource([binary_then_valid])

        with caplog.at_level(logging.WARNING):
            result = _watch(queue_dir, source)

        assert result is not None and result.id == esc.id
        assert any(
            r.levelno == logging.WARNING
            and 'watcher.watch' in r.getMessage()
            and 'esc-binary.json' in r.getMessage()
            for r in caplog.records
        ), [(r.levelname, r.getMessage()) for r in caplog.records]

    def test_returns_none_when_nothing_arrives_before_timeout(self, queue_dir: Path):
        source = ScriptedSource([])

        assert _watch(queue_dir, source, timeout=2) is None
        assert source.reads == [2000]

    def test_read_timeout_is_recomputed_from_the_clock(self, queue_dir: Path):
        clock = FakeClock()
        esc = _esc('esc-58-1')

        def time_passes() -> list[str]:
            clock.advance(1.5)
            return ['unrelated.json']

        source = ScriptedSource([time_passes, lambda: [_arrive(queue_dir, esc)]])

        result = _watch(queue_dir, source, timeout=5, clock=clock)

        assert result is not None and result.id == esc.id
        assert source.reads == [5000, 3500]

    def test_deadline_passed_during_a_poll_returns_without_another_read(
        self, queue_dir: Path,
    ):
        clock = FakeClock()

        def overrun() -> list[str]:
            clock.advance(6)
            return ['unrelated.json']

        source = ScriptedSource([overrun])

        assert _watch(queue_dir, source, timeout=5, clock=clock) is None
        assert source.reads == [5000]

    def test_deadline_is_taken_after_the_pre_scan_excludes_call(self, queue_dir: Path):
        clock = FakeClock()
        scans: list[None] = []

        def slow_first_excludes() -> frozenset[str]:
            if not scans:
                clock.advance(3)
            scans.append(None)
            return frozenset()

        esc = _esc('esc-59-1')
        source = ScriptedSource([lambda: [_arrive(queue_dir, esc)]])

        _watch(
            queue_dir, source, timeout=5, clock=clock, current_excludes=slow_first_excludes,
        )

        assert source.reads[0] == 5000

    def test_real_inotify_delivers_an_arrival_after_the_scan(self, queue_dir: Path):
        esc = _esc('esc-60-1')

        with armed_source(queue_dir) as inotify:
            class ArriveThenRead:
                def read(self, timeout: int | None = None) -> list[Event]:
                    if not (queue_dir / f'{esc.id}.json').exists():
                        _arrive(queue_dir, esc)
                    return inotify.read(timeout=timeout)

            result = watch(
                queue_dir, ArriveThenRead(), task_id=None, level=None, timeout=5,
                current_excludes=lambda: frozenset(), clock=time.monotonic,
            )

        assert result is not None and result.id == esc.id


@pytest.fixture
def restore_sigterm():
    """main installs a SIGTERM handler; keep it from leaking into the pytest process."""
    saved = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGTERM, saved)


def _deliver_while_scan_reads(gate: Path, deliver: Callable[[], None]) -> threading.Thread:
    """Run `deliver` on a thread while the initial scan is blocked reading `gate`.

    `gate` is created as a FIFO in the queue dir. Opening it for writing
    returns only once the scan has opened it for reading, and the scan lists
    the directory before reading any entry, so the delivery is invisible to
    the scan. The gate then yields a resolved record, which the scan skips.
    """
    os.mkfifo(gate)
    record = _esc(gate.stem, status='resolved').to_json().encode()

    def run() -> None:
        fd = os.open(gate, os.O_WRONLY)
        try:
            deliver()
            os.write(fd, record)
        finally:
            os.close(fd)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _exit_code(argv: list[str]) -> int | str | None:
    with pytest.raises(SystemExit) as exc_info:
        main(argv)
    return exc_info.value.code


@pytest.mark.usefixtures('restore_sigterm')
class TestMain:
    """main(argv) wires the CLI flags into watch() and maps its result to an exit code."""

    def test_pending_escalation_exits_0_with_its_json_on_stdout(
        self, queue_dir: Path, capsys: pytest.CaptureFixture[str],
    ):
        esc = _esc('esc-70-1')
        _write_esc(queue_dir, esc)

        assert _exit_code(['--queue-dir', str(queue_dir), '--timeout', '5']) == 0
        assert json.loads(capsys.readouterr().out) == esc.to_dict()

    def test_empty_queue_exits_124_with_nothing_on_stdout(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ):
        queue_dir = tmp_path / 'not-yet-created'

        assert _exit_code(['--queue-dir', str(queue_dir), '--timeout', '0.05']) == 124
        assert capsys.readouterr().out == ''
        assert queue_dir.is_dir()

    @pytest.mark.parametrize('suffix', ['', '.json'], ids=['bare', 'json-suffixed'])
    def test_exclude_id_excludes_the_pending_item(self, queue_dir: Path, suffix: str):
        esc = _esc('esc-71-1')
        _write_esc(queue_dir, esc)

        assert _exit_code([
            '--queue-dir', str(queue_dir), '--timeout', '0.05',
            '--exclude-id', f'{esc.id}{suffix}',
        ]) == 124

    def test_exclude_file_excludes_the_pending_item(self, queue_dir: Path, tmp_path: Path):
        esc = _esc('esc-72-1')
        _write_esc(queue_dir, esc)
        exclude_file = tmp_path / 'excludes.txt'
        exclude_file.write_text(f'{esc.id}\n')

        assert _exit_code([
            '--queue-dir', str(queue_dir), '--timeout', '0.05',
            '--exclude-file', str(exclude_file),
        ]) == 124

    def test_baseline_excludes_the_item_pending_at_launch(self, queue_dir: Path):
        _write_esc(queue_dir, _esc('esc-73-1'))

        assert _exit_code([
            '--queue-dir', str(queue_dir), '--timeout', '0.05', '--baseline',
        ]) == 124

    @pytest.mark.parametrize(
        ('fields', 'extra_argv', 'code'),
        [
            pytest.param({'level': 1}, ['--level', '0'], 124, id='level-mismatch'),
            pytest.param({'level': 1}, ['--level', '1'], 0, id='level-match'),
            pytest.param({'task_id': '42'}, ['--task-id', '99'], 124, id='task-mismatch'),
        ],
    )
    def test_level_and_task_id_reach_the_filter(
        self, queue_dir: Path, fields: dict, extra_argv: list[str], code: int,
    ):
        _write_esc(queue_dir, _esc('esc-74-1', **fields))

        assert _exit_code(['--queue-dir', str(queue_dir), '--timeout', '0.05', *extra_argv]) == code

    @pytest.mark.parametrize('deliver', [os.rename, os.link], ids=['moved-to', 'create'])
    def test_arrival_the_scan_cannot_see_is_delivered_by_mains_armed_watch(
        self,
        queue_dir: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        deliver: Callable[[Path, Path], None],
    ):
        """An arrival after the scan has listed the queue dir must still exit 0.

        The staged file sits outside the queue dir, so its only queue event is
        the delivery itself: a rename (MOVED_TO) or a hard link (CREATE, with
        the content already complete). A main that arms no watch, or arms it
        without that flag, exits 124.
        """
        esc = _esc('esc-75-1')
        staged = tmp_path / f'{esc.id}.json'
        staged.write_text(esc.to_json())
        delivery = _deliver_while_scan_reads(
            queue_dir / 'esc-75-0.json', lambda: deliver(staged, queue_dir / staged.name),
        )

        assert _exit_code(['--queue-dir', str(queue_dir), '--timeout', '5']) == 0
        delivery.join(timeout=5)
        assert json.loads(capsys.readouterr().out) == esc.to_dict()


class TestExclusions:
    """exclusions() keeps static and baseline fixed and re-reads the exclude file per call."""

    def test_exclude_file_is_reread_on_every_call(self, tmp_path: Path):
        exclude_file = tmp_path / 'excludes.txt'
        current = exclusions(
            static=frozenset({'esc-1-1'}), baseline=frozenset({'esc-2-1'}),
            exclude_file=exclude_file,
        )

        assert current() == {'esc-1-1', 'esc-2-1'}

        exclude_file.write_text('esc-3-1\n')
        assert current() == {'esc-1-1', 'esc-2-1', 'esc-3-1'}

        with exclude_file.open('a') as f:
            f.write('esc-4-1\n')
        assert current() == {'esc-1-1', 'esc-2-1', 'esc-3-1', 'esc-4-1'}

    def test_without_exclude_file_is_static_union_baseline(self):
        current = exclusions(
            static=frozenset({'esc-1-1'}), baseline=frozenset({'esc-2-1'}), exclude_file=None,
        )

        assert current() == {'esc-1-1', 'esc-2-1'}


class TestExcludeFile:
    """_read_exclude_file(path) -> frozenset[str]; fail-open on missing/None."""

    def test_read_exclude_file_helper(self, tmp_path):
        from escalation.watcher import _read_exclude_file

        # (a) None path and a nonexistent path both -> empty frozenset (fail-open)
        assert _read_exclude_file(None) == frozenset()
        assert _read_exclude_file(tmp_path / 'does-not-exist') == frozenset()

        # (b) blank lines and #-comments skipped, whitespace stripped, .json
        # suffix normalized the same way --exclude-id is (watcher.py:154-156)
        exclude_file = tmp_path / 'excludes.txt'
        exclude_file.write_text(
            'esc-1-1\n'
            'esc-2-1.json\n'
            '\n'
            '# a comment\n'
            '  esc-3-1  \n'
        )
        assert _read_exclude_file(exclude_file) == frozenset({'esc-1-1', 'esc-2-1', 'esc-3-1'})
