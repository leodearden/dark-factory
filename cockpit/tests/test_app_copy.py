"""CockpitApp pilot tests for the 'y' copy action."""

from __future__ import annotations

import pytest
from _cockpit_session_records import make_record
from _cockpit_tree_snapshot import snapshot_tree
from orchestrator import decision_registry as dr
from orchestrator import session_records

from cockpit.app import CockpitApp


class RecordingCopyRunner:
    """A CockpitApp `copy_runner` double: records each payload, reports a scripted CopyAttempt.

    The seam CockpitApp actually holds is Callable[[str], CopyAttempt] --
    the app has no business knowing about argv, `which` or $DISPLAY (those
    are cockpit.clipboard's own injected knobs, exercised at the process
    boundary in test_clipboard.py). Recording the payload here is what pins
    the app half of the leaf signal: which text 'y' hands to the clipboard.
    """

    def __init__(self, attempt=None):
        from cockpit.clipboard import CopyAttempt, CopyOutcome

        self.payloads: list[str] = []
        self._attempt = (
            attempt if attempt is not None else CopyAttempt(CopyOutcome.COPIED, ('xclip',))
        )

    def __call__(self, text):
        self.payloads.append(text)
        return self._attempt


class TestCopyAction:
    """'y' hands the highlighted row's payload to the clipboard seam.

    These tests previously asserted on `app._clipboard`, Textual's private
    record of the last copy_to_clipboard call -- the false green task 5448
    re-opened 2517 to fix (cockpit/src/cockpit/clipboard.py's module
    docstring says why that attribute cannot answer the question). The
    replacement asserts on the injected seam (here) and on Textual's PUBLIC
    copy_to_clipboard/notify (TestCopyFallbackAndFeedback below) -- never a
    private attribute.
    """

    @pytest.mark.timeout(10)
    async def test_copy_highlighted_decision_puts_question_and_ids_on_clipboard(self, tmp_path):
        """'y' (the copy affordance, task 2517 / esc-2303-1 F4) hands the
        highlighted DecisionQueue row's question text + ids to the system-
        clipboard seam (cockpit.clipboard::copy_to_system_clipboard in
        production, a recorder here), and is strictly READ-ONLY, never
        touching sessions/ or decisions/ (mirrors test_app.py::TestWriteDiscipline's
        before/after snapshot_tree diff).
        """
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        decision = dr.DecisionRecord(
            id='dec-1',
            project='df',
            text='Which port do we bind?',
            filed_at='2026-07-07T00:00:00+00:00',
            task_id='2517',
            escalation_id='esc-42',
        )
        assert dr.write_decision(decision, root=tmp_path)

        awaiting = make_record(
            session_slug='awaiting-1',
            status=session_records.Status.AWAITING_INPUT,
            question=session_records.Question(text='Which host?', asked_at='2026-07-07T00:00:00+00:00'),
        )
        session_records.write_record(awaiting, root=tmp_path)

        backend = FakeBackend()
        copy_runner = RecordingCopyRunner()
        app = CockpitApp(
            fleet_root=tmp_path, backend=backend, poll_interval=0.05, copy_runner=copy_runner
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            assert queue.row_count == 2

            queue.move_cursor(row=queue.get_row_index('decision:dec-1'))
            await pilot.pause()

            before = snapshot_tree(tmp_path)

            await pilot.press('y')
            await pilot.pause()

            # (a) the row's question + ids reached the clipboard seam, once.
            assert len(copy_runner.payloads) == 1
            payload = copy_runner.payloads[0]
            assert 'Which port do we bind?' in payload
            assert 'esc-42' in payload
            assert '2517' in payload

            # (b) strictly read-only -- no sessions/ or decisions/ write.
            after = snapshot_tree(tmp_path)
            assert after == before

    @pytest.mark.timeout(10)
    async def test_copy_highlighted_session_puts_slug_and_question_on_clipboard(self, tmp_path):
        """'y' on a SESSION-backed row (no decision behind it) hands the
        session slug + question text to the clipboard seam. Covers the
        app-level action_copy -> highlighted SESSION row -> clipboard path
        end-to-end -- the decision-row case above and format_copy_payload's
        own pure-formatter unit tests don't exercise this branch through
        action_copy itself (reviewer_comprehensive test_coverage
        suggestion, test_app.py:911).
        """
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        awaiting = make_record(
            session_slug='awaiting-99',
            status=session_records.Status.AWAITING_INPUT,
            question=session_records.Question(text='Which region?', asked_at='2026-07-07T00:00:00+00:00'),
        )
        session_records.write_record(awaiting, root=tmp_path)

        backend = FakeBackend()
        copy_runner = RecordingCopyRunner()
        app = CockpitApp(
            fleet_root=tmp_path, backend=backend, poll_interval=0.05, copy_runner=copy_runner
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            assert queue.row_count == 1

            queue.move_cursor(row=queue.get_row_index('session:awaiting-99'))
            await pilot.pause()

            before = snapshot_tree(tmp_path)

            await pilot.press('y')
            await pilot.pause()

            # (a) the session row's question + slug reached the clipboard seam.
            assert len(copy_runner.payloads) == 1
            payload = copy_runner.payloads[0]
            assert 'Which region?' in payload
            assert 'awaiting-99' in payload

            # (b) strictly read-only -- a session is never cockpit-written.
            after = snapshot_tree(tmp_path)
            assert after == before

    @pytest.mark.timeout(10)
    async def test_copy_with_empty_queue_is_a_fail_soft_no_op(self, tmp_path):
        """'y' against an EMPTY queue -- highlighted_key() returns None,
        mirroring action_drop/action_defer's own fail-soft guard -- must not
        crash and must never reach the clipboard seam at all (no stray
        subprocess, no misleading toast for a row that doesn't exist).
        """
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        backend = FakeBackend()
        copy_runner = RecordingCopyRunner()
        app = CockpitApp(
            fleet_root=tmp_path, backend=backend, poll_interval=0.05, copy_runner=copy_runner
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            assert queue.row_count == 0

            await pilot.press('y')
            await pilot.pause()

            assert copy_runner.payloads == []


class RecordingOsc52:
    """Spy for Textual's PUBLIC App.copy_to_clipboard -- i.e. the OSC 52 leg.

    A documented public method, replaced on the instance: a
    collaborator-boundary assertion, not a reach into internals. The
    attribute it used to set, `app._clipboard`, is precisely what cannot
    answer "did the escape actually get written, and did it reach a
    clipboard" -- see TestCopyAction's class docstring.
    """

    def __init__(self):
        self.texts: list[str] = []

    def __call__(self, text):
        self.texts.append(text)


class RecordingNotify:
    """Spy for Textual's PUBLIC App.notify; records (message, severity, title) per toast.

    Takes **kwargs rather than notify's exact keyword list so a Textual
    release adding a parameter widens this spy for free. The message is
    recorded because it is the deliverable: the operator being told what
    happened and through which mechanism. Its WORDING is pinned once, in
    test_clipboard.py::TestCopyFeedback; what these tests pin is that
    action_copy forwards that exact string rather than some other one.
    """

    def __init__(self):
        self.calls: list[tuple[str, str, str | None]] = []

    def __call__(self, message, **kwargs):
        self.calls.append(
            (message, kwargs.get('severity', 'information'), kwargs.get('title'))
        )


class ExplodingCopyRunner:
    """A copy_runner that raises: the seam is caller-injectable, so action_copy guards it.

    The bundled copy_to_system_clipboard never raises, but a keypress
    handler must not depend on that promise being kept by whatever the
    constructor was handed (PRD §2 -- a view, never a dependency).
    """

    def __init__(self):
        self.payloads: list[str] = []

    def __call__(self, text):
        self.payloads.append(text)
        raise RuntimeError('clipboard helper exploded')


def _write_copyable_decision(tmp_path):
    """One highlightable decision row, mirroring TestCopyAction's fixture."""
    decision = dr.DecisionRecord(
        id='dec-1',
        project='df',
        text='Which port do we bind?',
        filed_at='2026-07-07T00:00:00+00:00',
        task_id='2517',
        escalation_id='esc-42',
    )
    assert dr.write_decision(decision, root=tmp_path)


def _spy_on_clipboard_surface(monkeypatch, app):
    """Replace Textual's PUBLIC copy_to_clipboard/notify with recorders; return both."""
    osc52 = RecordingOsc52()
    notifications = RecordingNotify()
    monkeypatch.setattr(app, 'copy_to_clipboard', osc52)
    monkeypatch.setattr(app, 'notify', notifications)
    return osc52, notifications


class TestCopyFallbackAndFeedback:
    """action_copy's policy: when OSC 52 still runs, and what the operator is told.

    The incident's primary complaint was that 'y' produced no signal in
    either direction, which is how a total no-op survived a whole task
    cycle. Both paths now toast, and the toast names the mechanism.
    """

    @pytest.mark.timeout(10)
    async def test_a_successful_local_copy_skips_osc52_and_toasts_information(
        self, tmp_path, monkeypatch
    ):
        """A local helper took the payload -- no wasted escape write, and a success toast.

        The toast must carry copy_feedback's own message, naming the
        mechanism that ran: a regression passing the payload (or a message
        with the helper name dropped) would otherwise leave every test in
        this class green while telling the operator nothing useful.
        """
        from cockpit.backends import FakeBackend
        from cockpit.clipboard import CopyAttempt, CopyOutcome, copy_feedback
        from cockpit.panes.decision_queue import DecisionQueue

        _write_copyable_decision(tmp_path)

        attempt = CopyAttempt(CopyOutcome.COPIED, ('xclip',))
        app = CockpitApp(
            fleet_root=tmp_path,
            backend=FakeBackend(),
            poll_interval=0.05,
            copy_runner=RecordingCopyRunner(attempt),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            queue.move_cursor(row=queue.get_row_index('decision:dec-1'))
            await pilot.pause()

            osc52, notifications = _spy_on_clipboard_surface(monkeypatch, app)

            await pilot.press('y')
            await pilot.pause()

            assert osc52.texts == []
            assert notifications.calls == [
                (copy_feedback(attempt).message, 'information', 'Copy')
            ]
            assert 'xclip' in notifications.calls[0][0]

    @pytest.mark.timeout(10)
    async def test_no_local_helper_falls_back_to_osc52_and_warns(self, tmp_path, monkeypatch):
        """The over-SSH case: the same payload goes out as OSC 52, and the toast says so."""
        from cockpit.backends import FakeBackend
        from cockpit.clipboard import CopyAttempt, CopyOutcome, copy_feedback
        from cockpit.panes.decision_queue import DecisionQueue

        _write_copyable_decision(tmp_path)

        attempt = CopyAttempt(CopyOutcome.NO_HELPER)
        copy_runner = RecordingCopyRunner(attempt)
        app = CockpitApp(
            fleet_root=tmp_path,
            backend=FakeBackend(),
            poll_interval=0.05,
            copy_runner=copy_runner,
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            queue.move_cursor(row=queue.get_row_index('decision:dec-1'))
            await pilot.pause()

            osc52, notifications = _spy_on_clipboard_surface(monkeypatch, app)

            await pilot.press('y')
            await pilot.pause()

            assert osc52.texts == copy_runner.payloads
            assert len(notifications.calls) == 1
            message, severity, _title = notifications.calls[0]
            assert severity == 'warning'
            assert message == copy_feedback(attempt).message
            assert 'OSC 52' in message

    @pytest.mark.timeout(10)
    async def test_a_raising_copy_runner_still_falls_back_and_warns(self, tmp_path, monkeypatch):
        """An exception out of the injected seam must not take the cockpit down.

        The keypress completes, the operator still gets the OSC 52 fallback
        with the same payload, and the toast reports the failure rather than
        silently claiming success.
        """
        from cockpit.backends import FakeBackend
        from cockpit.clipboard import CopyAttempt, CopyOutcome, copy_feedback
        from cockpit.panes.decision_queue import DecisionQueue

        _write_copyable_decision(tmp_path)

        copy_runner = ExplodingCopyRunner()
        app = CockpitApp(
            fleet_root=tmp_path,
            backend=FakeBackend(),
            poll_interval=0.05,
            copy_runner=copy_runner,
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            queue.move_cursor(row=queue.get_row_index('decision:dec-1'))
            await pilot.pause()

            osc52, notifications = _spy_on_clipboard_surface(monkeypatch, app)

            await pilot.press('y')
            await pilot.pause()

            assert copy_runner.payloads
            assert osc52.texts == copy_runner.payloads
            assert len(notifications.calls) == 1
            message, severity, _title = notifications.calls[0]
            assert severity == 'warning'
            # The guard degrades to a HELPER_FAILED carrying no argv, so the
            # toast reads as prose rather than naming a helper that never ran.
            assert message == copy_feedback(CopyAttempt(CopyOutcome.HELPER_FAILED)).message
            assert 'OSC 52' in message

    @pytest.mark.timeout(10)
    async def test_empty_queue_neither_falls_back_nor_toasts(self, tmp_path, monkeypatch):
        """No highlighted row: nothing was copied, so there is nothing to say about it."""
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        app = CockpitApp(
            fleet_root=tmp_path,
            backend=FakeBackend(),
            poll_interval=0.05,
            copy_runner=RecordingCopyRunner(),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one(DecisionQueue).row_count == 0

            osc52, notifications = _spy_on_clipboard_surface(monkeypatch, app)

            await pilot.press('y')
            await pilot.pause()

            assert osc52.texts == []
            assert notifications.calls == []
