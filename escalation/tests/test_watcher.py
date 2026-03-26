"""Tests for escalation.watcher — ntfy push and loop behavior."""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

import pytest

from escalation.models import Escalation
from escalation.watcher import _send_ntfy


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


class TestMainLoop:
    """CLI exit behavior after first match."""

    def test_exits_after_first(self, tmp_path, blocking_escalation: Escalation):
        """main() calls sys.exit(0) after first match."""
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
        ):
            mock_inotify = MockINotify.return_value
            # read() returns events once, then raises to break the loop
            mock_inotify.read.side_effect = [[mock_event], KeyboardInterrupt]

            from escalation.watcher import main

            with contextlib.suppress(KeyboardInterrupt):
                main()

            mock_exit.assert_called_once_with(0)


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
