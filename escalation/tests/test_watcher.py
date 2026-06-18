"""Tests for escalation.watcher — ntfy push and loop behavior."""

from __future__ import annotations

import contextlib
import json
import logging
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from escalation.models import Escalation
from escalation.watcher import _initial_scan, _send_ntfy


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


class TestSendNtfy:
    """_send_ntfy sends POST with correct headers/body."""

    def test_blocking_escalation_sends_urgent(self, blocking_escalation: Escalation):
        with patch('escalation.watcher.urllib.request.urlopen') as mock_urlopen:
            _send_ntfy('https://ntfy.sh/test-topic', blocking_escalation)

            mock_urlopen.assert_called_once()
            req = mock_urlopen.call_args[0][0]

            assert req.full_url == 'https://ntfy.sh/test-topic'
            assert req.get_method() == 'POST'
            assert req.get_header('Title') == '[BLOCKING] Task 42: task_failure'
            assert req.get_header('Priority') == 'urgent'
            assert req.get_header('Tags') == 'rotating_light'
            assert b'Verification attempts exhausted' in req.data

    def test_info_escalation_sends_default_priority(self, info_escalation: Escalation):
        with patch('escalation.watcher.urllib.request.urlopen') as mock_urlopen:
            _send_ntfy('https://ntfy.sh/test-topic', info_escalation)

            req = mock_urlopen.call_args[0][0]

            assert req.get_header('Title') == '[INFO] Task 42: design_concern'
            assert req.get_header('Priority') == 'default'
            assert req.get_header('Tags') == 'information_source'

    def test_body_includes_detail(self, blocking_escalation: Escalation):
        with patch('escalation.watcher.urllib.request.urlopen') as mock_urlopen:
            _send_ntfy('https://ntfy.sh/test-topic', blocking_escalation)

            req = mock_urlopen.call_args[0][0]
            body = req.data.decode('utf-8')
            assert 'tests failed 3 times' in body

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
        with patch('escalation.watcher.urllib.request.urlopen') as mock_urlopen:
            _send_ntfy('https://ntfy.sh/t', esc)

            body = mock_urlopen.call_args[0][0].data.decode('utf-8')
            # Should appear once (the summary), not twice
            assert body == 'Planning failed'

    def test_critical_escalation_sends_urgent(self, critical_escalation: Escalation):
        """critical severity -> [CRITICAL] title, Priority=urgent, Tags=rotating_light."""
        with patch('escalation.watcher.urllib.request.urlopen') as mock_urlopen:
            _send_ntfy('https://ntfy.sh/test-topic', critical_escalation)

            req = mock_urlopen.call_args[0][0]
            assert req.get_header('Title') == '[CRITICAL] Task 99: infra_issue'
            assert req.get_header('Priority') == 'urgent'
            assert req.get_header('Tags') == 'rotating_light'

    def test_urgent_escalation_sends_urgent(self, urgent_escalation: Escalation):
        """urgent severity -> [URGENT] title, Priority=urgent, Tags=rotating_light."""
        with patch('escalation.watcher.urllib.request.urlopen') as mock_urlopen:
            _send_ntfy('https://ntfy.sh/test-topic', urgent_escalation)

            req = mock_urlopen.call_args[0][0]
            assert req.get_header('Title') == '[URGENT] Task 99: risk_identified'
            assert req.get_header('Priority') == 'urgent'
            assert req.get_header('Tags') == 'rotating_light'


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

        with caplog.at_level(logging.WARNING, logger='escalation.watcher'):
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


class TestInitialScanMain:
    """main() emits a pre-existing pending escalation without any inotify event."""

    def test_emits_pre_existing_pending_without_inotify_event(
        self, tmp_path, blocking_escalation: Escalation, capsys
    ):
        """Pre-existing pending file is emitted at startup; inotify.read never called."""
        queue_dir = tmp_path / 'queue'
        queue_dir.mkdir()
        esc_path = queue_dir / f'{blocking_escalation.id}.json'
        esc_path.write_text(blocking_escalation.to_json())

        with (
            patch('escalation.watcher.INotify') as MockINotify,
            patch('escalation.watcher.sys.argv', [
                'watcher', '--queue-dir', str(queue_dir),
            ]),
        ):
            mock_inotify = MockINotify.return_value
            # read must NOT be called — if it is, something is wrong
            mock_inotify.read.side_effect = AssertionError('inotify.read should not be called')

            from escalation.watcher import main

            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0

        # add_watch was armed before the scan
        mock_inotify.add_watch.assert_called_once()

        # stdout contains the escalation JSON
        captured = capsys.readouterr()
        assert blocking_escalation.id in captured.out
        data = json.loads(captured.out)
        assert data['id'] == blocking_escalation.id


class TestExcludeId:
    """--exclude-id flag: repeatable, honoured in initial scan AND event loop."""

    def test_startup_excludes_both_pending_enters_event_loop(self, tmp_path, capsys):
        """With two pending files both excluded, initial scan returns None -> event loop entered."""
        queue_dir = tmp_path / 'queue'
        queue_dir.mkdir()
        esc1 = Escalation(
            id='esc-20-1', task_id='20', agent_role='orchestrator',
            severity='blocking', category='task_failure', summary='first',
        )
        esc2 = Escalation(
            id='esc-20-2', task_id='20', agent_role='orchestrator',
            severity='blocking', category='task_failure', summary='second',
        )
        (queue_dir / f'{esc1.id}.json').write_text(esc1.to_json())
        (queue_dir / f'{esc2.id}.json').write_text(esc2.to_json())

        with (
            patch('escalation.watcher.INotify') as MockINotify,
            patch('escalation.watcher.sys.argv', [
                'watcher', '--queue-dir', str(queue_dir),
                '--exclude-id', esc1.id,
                '--exclude-id', esc2.id,
            ]),
        ):
            mock_inotify = MockINotify.return_value
            mock_inotify.read.side_effect = KeyboardInterrupt

            from escalation.watcher import main

            with contextlib.suppress(KeyboardInterrupt):
                main()

        captured = capsys.readouterr()
        assert captured.out == ''
        mock_inotify.read.assert_called_once()

    def test_exclude_id_json_suffix_normalized(self, tmp_path, capsys):
        """--exclude-id with .json suffix is normalized; esc-7-1.json excludes esc-7-1."""
        queue_dir = tmp_path / 'queue'
        queue_dir.mkdir()
        esc = Escalation(
            id='esc-7-1', task_id='7', agent_role='orchestrator',
            severity='blocking', category='task_failure', summary='pending',
        )
        (queue_dir / f'{esc.id}.json').write_text(esc.to_json())

        with (
            patch('escalation.watcher.INotify') as MockINotify,
            patch('escalation.watcher.sys.argv', [
                'watcher', '--queue-dir', str(queue_dir),
                '--exclude-id', f'{esc.id}.json',
            ]),
        ):
            mock_inotify = MockINotify.return_value
            mock_inotify.read.side_effect = KeyboardInterrupt

            from escalation.watcher import main

            with contextlib.suppress(KeyboardInterrupt):
                main()

        captured = capsys.readouterr()
        assert captured.out == ''
        mock_inotify.read.assert_called_once()

    def test_event_loop_suppresses_excluded_event(self, tmp_path, capsys):
        """Event for an excluded file (e.g. dedupe MOVED_TO rewrite) is silently ignored."""
        queue_dir = tmp_path / 'queue'
        queue_dir.mkdir()
        esc = Escalation(
            id='esc-21-1', task_id='21', agent_role='orchestrator',
            severity='blocking', category='task_failure', summary='deliberately-pending',
        )
        (queue_dir / f'{esc.id}.json').write_text(esc.to_json())

        mock_event = MagicMock()
        mock_event.name = f'{esc.id}.json'

        with (
            patch('escalation.watcher.INotify') as MockINotify,
            patch('escalation.watcher.sys.exit') as mock_exit,
            patch('escalation.watcher.sys.argv', [
                'watcher', '--queue-dir', str(queue_dir),
                '--exclude-id', esc.id,
            ]),
            patch('escalation.watcher._initial_scan', return_value=None),
        ):
            mock_inotify = MockINotify.return_value
            mock_inotify.read.side_effect = [[mock_event], KeyboardInterrupt]

            from escalation.watcher import main

            with contextlib.suppress(KeyboardInterrupt):
                main()

        mock_exit.assert_not_called()
        assert capsys.readouterr().out == ''

    def test_event_loop_fires_on_non_excluded_event(self, tmp_path, capsys):
        """Event for a different (non-excluded) file fires normally."""
        queue_dir = tmp_path / 'queue'
        queue_dir.mkdir()
        excluded_esc = Escalation(
            id='esc-22-1', task_id='22', agent_role='orchestrator',
            severity='blocking', category='task_failure', summary='excluded',
        )
        live_esc = Escalation(
            id='esc-22-2', task_id='22', agent_role='orchestrator',
            severity='blocking', category='task_failure', summary='live',
        )
        (queue_dir / f'{excluded_esc.id}.json').write_text(excluded_esc.to_json())
        (queue_dir / f'{live_esc.id}.json').write_text(live_esc.to_json())

        mock_event = MagicMock()
        mock_event.name = f'{live_esc.id}.json'

        with (
            patch('escalation.watcher.INotify') as MockINotify,
            patch('escalation.watcher.sys.exit') as mock_exit,
            patch('escalation.watcher.sys.argv', [
                'watcher', '--queue-dir', str(queue_dir),
                '--exclude-id', excluded_esc.id,
            ]),
            patch('escalation.watcher._initial_scan', return_value=None),
        ):
            mock_inotify = MockINotify.return_value
            mock_inotify.read.side_effect = [[mock_event], KeyboardInterrupt]

            from escalation.watcher import main

            with contextlib.suppress(KeyboardInterrupt):
                main()

        mock_exit.assert_called_once_with(0)
        assert live_esc.id in capsys.readouterr().out


class TestMainLoop:
    """CLI exit behavior after first match."""

    def test_exits_after_first(self, tmp_path, blocking_escalation: Escalation):
        """main() calls sys.exit(0) after first match (via inotify event loop)."""
        queue_dir = tmp_path / 'queue'
        queue_dir.mkdir()

        # Pre-create the escalation file so inotify fires on it
        esc_path = queue_dir / f'{blocking_escalation.id}.json'
        esc_path.write_text(blocking_escalation.to_json())

        # Mock inotify to yield one event then stop
        mock_event = MagicMock()
        mock_event.name = f'{blocking_escalation.id}.json'

        with (
            patch('escalation.watcher.INotify') as MockINotify,
            patch('escalation.watcher.sys.exit') as mock_exit,
            patch('escalation.watcher.sys.argv', [
                'watcher', '--queue-dir', str(queue_dir),
            ]),
            # Neutralise initial scan so we test the event loop path
            patch('escalation.watcher._initial_scan', return_value=None),
        ):
            mock_inotify = MockINotify.return_value
            # read() returns events once, then raises to break the loop
            mock_inotify.read.side_effect = [[mock_event], KeyboardInterrupt]

            from escalation.watcher import main

            with contextlib.suppress(KeyboardInterrupt):
                main()

            mock_exit.assert_called_once_with(0)


class TestTimeout:
    """--timeout arg: exit 124 on expiry; no timeout -> block indefinitely."""

    def test_timeout_exits_124_on_expiry(self, tmp_path, capsys):
        """--timeout 2 -> exit 124 when read returns empty (no events); stdout empty."""
        queue_dir = tmp_path / 'queue'
        queue_dir.mkdir()

        # Fixed monotonic sequence: start=0.0 sets deadline=2.0; loop check=0.0
        # so remaining_ms = max(0, int((2.0 - 0.0)*1000)) = 2000 -> read(timeout=2000)
        # read returns [] -> exit 124 (empty events with deadline set)
        monotonic_values = iter([0.0, 0.0])

        with (
            patch('escalation.watcher.INotify') as MockINotify,
            patch('escalation.watcher.sys.argv', [
                'watcher', '--queue-dir', str(queue_dir), '--timeout', '2',
            ]),
            patch('escalation.watcher._initial_scan', return_value=None),
            patch('escalation.watcher.time.monotonic', side_effect=monotonic_values),
        ):
            mock_inotify = MockINotify.return_value
            # read returns empty list -> no events -> timeout path
            mock_inotify.read.return_value = []

            from escalation.watcher import main

            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 124

        # stdout must be empty (no escalation emitted)
        captured = capsys.readouterr()
        assert captured.out == ''

        # read was called with timeout=2000ms
        mock_inotify.read.assert_called_once_with(timeout=2000)

    def test_no_timeout_blocks_indefinitely(self, tmp_path):
        """Without --timeout, read is called with timeout=None (block forever)."""
        queue_dir = tmp_path / 'queue'
        queue_dir.mkdir()

        with (
            patch('escalation.watcher.INotify') as MockINotify,
            patch('escalation.watcher.sys.argv', [
                'watcher', '--queue-dir', str(queue_dir),
            ]),
            patch('escalation.watcher._initial_scan', return_value=None),
        ):
            mock_inotify = MockINotify.return_value
            # No timeout -> read blocks; we break out with KeyboardInterrupt
            mock_inotify.read.side_effect = KeyboardInterrupt

            from escalation.watcher import main

            with contextlib.suppress(KeyboardInterrupt):
                main()

        mock_inotify.read.assert_called_once_with(timeout=None)

    def test_timeout_with_matching_event_exits_0(
        self, tmp_path, blocking_escalation: Escalation, capsys
    ):
        """--timeout set AND matching event arrives before deadline -> exit 0 + JSON."""
        queue_dir = tmp_path / 'queue'
        queue_dir.mkdir()

        # Pre-create the escalation file so the event loop can read it
        esc_path = queue_dir / f'{blocking_escalation.id}.json'
        esc_path.write_text(blocking_escalation.to_json())

        mock_event = MagicMock()
        mock_event.name = f'{blocking_escalation.id}.json'

        # monotonic sequence: first call sets deadline = 0.0 + 5.0 = 5.0;
        # second call in the loop -> remaining_ms = int((5.0 - 0.5) * 1000) = 4500
        monotonic_values = iter([0.0, 0.5])

        with (
            patch('escalation.watcher.INotify') as MockINotify,
            patch('escalation.watcher.sys.argv', [
                'watcher', '--queue-dir', str(queue_dir), '--timeout', '5',
            ]),
            patch('escalation.watcher._initial_scan', return_value=None),
            patch('escalation.watcher.time.monotonic', side_effect=monotonic_values),
        ):
            mock_inotify = MockINotify.return_value
            # read returns a matching event on the first call
            mock_inotify.read.return_value = [mock_event]

            from escalation.watcher import main

            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0

        # stdout contains the escalation JSON
        captured = capsys.readouterr()
        assert blocking_escalation.id in captured.out
        data = json.loads(captured.out)
        assert data['id'] == blocking_escalation.id

        # read was called with a positive timeout (not None)
        assert mock_inotify.read.call_args.kwargs['timeout'] == 4500


class TestLevelFilter:
    """--level argument filters by escalation level."""

    def _run_watcher(self, tmp_path, escalation, argv_extra=None):
        """Run main() with a single inotify event, return whether sys.exit was called."""
        queue_dir = tmp_path / 'queue'
        queue_dir.mkdir(exist_ok=True)

        esc_path = queue_dir / f'{escalation.id}.json'
        esc_path.write_text(escalation.to_json())

        mock_event = MagicMock()
        mock_event.name = f'{escalation.id}.json'

        argv = ['watcher', '--queue-dir', str(queue_dir)]
        if argv_extra:
            argv.extend(argv_extra)

        with (
            patch('escalation.watcher.INotify') as MockINotify,
            patch('escalation.watcher.sys.exit') as mock_exit,
            patch('escalation.watcher.sys.argv', argv),
            # Neutralise initial scan so we test the event loop path
            patch('escalation.watcher._initial_scan', return_value=None),
        ):
            mock_inotify = MockINotify.return_value
            mock_inotify.read.side_effect = [[mock_event], KeyboardInterrupt]

            from escalation.watcher import main

            with contextlib.suppress(KeyboardInterrupt):
                main()

            return mock_exit.called

    def test_level_filter_matches(self, tmp_path):
        """--level 0 passes level-0 escalation."""
        esc = Escalation(
            id='esc-10-1', task_id='10', agent_role='implementer',
            severity='blocking', category='task_failure', summary='fail',
            level=0,
        )
        assert self._run_watcher(tmp_path, esc, ['--level', '0']) is True

    def test_level_filter_skips_non_matching(self, tmp_path):
        """--level 0 skips level-1 escalation (no exit)."""
        esc = Escalation(
            id='esc-10-2', task_id='10', agent_role='steward',
            severity='blocking', category='task_failure', summary='reesc',
            level=1,
        )
        assert self._run_watcher(tmp_path, esc, ['--level', '0']) is False

    def test_no_level_filter_passes_all(self, tmp_path):
        """Without --level, level-1 escalation is returned (backward compat)."""
        esc = Escalation(
            id='esc-10-3', task_id='10', agent_role='steward',
            severity='blocking', category='task_failure', summary='reesc',
            level=1,
        )
        assert self._run_watcher(tmp_path, esc) is True
