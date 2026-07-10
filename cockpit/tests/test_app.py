"""Tests for cockpit.app — CockpitApp: the C5a TUI skeleton (session table + detail pane + poll).

Textual headless pilot tests (App.run_test / Pilot.pause) drive the few
integration signals this task requires: the table renders scanned records on
mount, the poll timer picks up on-disk changes, row selection renders the
detail pane, and the pure-consumer write discipline holds end-to-end. The
bulk of the underlying logic (glyph/title/age/order/detail/config) is
covered by fast deterministic unit tests elsewhere in this package -- see
test_session_table.py / test_detail_pane.py / test_registry_reader.py /
test_ui_config.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from orchestrator import session_registry as sr
from textual.coordinate import Coordinate


def _make_record(**overrides):
    """Mirrors test_registry_reader.py's _make_record convention."""
    fields: dict = {
        'session_slug': 'unblock-df-2085-4242',
        'status': sr.Status.RUNNING,
        'title': 'unblock:df#2085 slug',
        'role': 'unblock',
        'project': 'df',
        'task_id': '2085',
        'start_ts': '2026-07-07T00:00:00+00:00',
    }
    fields.update(overrides)
    return sr.SessionRecord(**fields)


def _snapshot_tree(base: Path) -> dict[str, tuple[int, bytes]]:
    """Map every file under *base* (relative path -> (mtime_ns, bytes)).

    Used to prove the cockpit's write discipline by diffing this snapshot
    across an app run: an absent path is not a KeyError, it's just missing
    from the mapping, so a deleted-or-never-written file shows up as a plain
    dict-equality mismatch rather than needing special-casing.
    """
    if not base.is_dir():
        return {}
    return {
        str(path.relative_to(base)): (path.stat().st_mtime_ns, path.read_bytes())
        for path in base.rglob('*')
        if path.is_file()
    }


class TestInitialRender:
    @pytest.mark.timeout(10)
    async def test_seeded_records_render_as_rows(self, tmp_path):
        from cockpit.app import CockpitApp
        from cockpit.panes.session_table import SessionTable

        r1 = _make_record(session_slug='a-1', status=sr.Status.RUNNING)
        r2 = _make_record(session_slug='b-2', status=sr.Status.AWAITING_INPUT)
        for r in (r1, r2):
            sr.write_record(r, root=tmp_path)

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            table = app.query_one(SessionTable)
            assert table.row_count == 2

            row = table.get_row('a-1')
            assert 'unblock:df#2085' in row


class TestPollRefresh:
    @pytest.mark.timeout(10)
    async def test_new_record_appears_and_orders_blocked_first(self, tmp_path):
        """A record written to disk after mount appears within one poll window,
        and a newly-blocked record is ordered first (blocked-first, timer-driven)."""
        from cockpit.app import CockpitApp
        from cockpit.panes.session_table import SessionTable

        running = _make_record(session_slug='running-1', status=sr.Status.RUNNING)
        sr.write_record(running, root=tmp_path)

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)
            assert table.row_count == 1

            blocked = _make_record(session_slug='blocked-1', status=sr.Status.AWAITING_INPUT)
            sr.write_record(blocked, root=tmp_path)

            await pilot.pause(0.2)

            assert table.row_count == 2
            first_row_key = table.coordinate_to_cell_key(Coordinate(0, 0)).row_key
            assert first_row_key.value == 'blocked-1'

    @pytest.mark.timeout(10)
    async def test_mutated_status_reflected_after_poll(self, tmp_path):
        """An existing record's status change on disk is picked up by the next poll tick."""
        from cockpit.app import CockpitApp
        from cockpit.panes.session_table import SessionTable, state_glyph

        running = _make_record(session_slug='running-1', status=sr.Status.RUNNING)
        sr.write_record(running, root=tmp_path)

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)
            assert state_glyph(sr.Status.RUNNING) in table.get_row('running-1')

            sr.update_status('running-1', root=tmp_path, status=sr.Status.IDLE)

            await pilot.pause(0.2)

            row = table.get_row('running-1')
            assert state_glyph(sr.Status.IDLE) in row
            assert state_glyph(sr.Status.RUNNING) not in row


class TestRowSelectionDetail:
    @pytest.mark.timeout(10)
    async def test_selecting_row_renders_detail(self, tmp_path):
        from cockpit.app import CockpitApp
        from cockpit.panes.detail_pane import DetailPane
        from cockpit.panes.session_table import SessionTable

        # other-1 is blocked, so it (not target-1) is the default row-0
        # cursor -- the assertions below only pass if selecting target-1
        # actually drives the detail pane, not just whatever's highlighted
        # by default after mount.
        other = _make_record(session_slug='other-1', status=sr.Status.AWAITING_INPUT)
        target = _make_record(
            session_slug='target-1',
            status=sr.Status.RUNNING,
            task_id='2085',
            escalation_id='esc-99',
            question=sr.Question(text='Which port?', asked_at='2026-07-07T00:00:00+00:00'),
        )
        for r in (other, target):
            sr.write_record(r, root=tmp_path)

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)

            table.move_cursor(row=table.get_row_index('target-1'))
            await pilot.pause()

            detail = app.query_one(DetailPane)
            assert 'Which port?' in detail.rendered_text
            assert '2085' in detail.rendered_text
            assert 'esc-99' in detail.rendered_text


class TestWriteDiscipline:
    @pytest.mark.timeout(10)
    async def test_cockpit_writes_only_its_own_ui_config(self, tmp_path):
        """PRD §2/§5 hard invariant: C5a is a pure consumer of the session and
        decision registries. It may create/update its OWN cockpit-ui.json, but
        running the app end-to-end (mount, a poll tick, a row selection) must
        never create, modify, or delete a sessions/ or decisions/ file --
        those belong to spawn-claude.sh/T4-T7/C8/C5b, never the C5a view."""
        from cockpit.app import CockpitApp
        from cockpit.panes.session_table import SessionTable

        record = _make_record(session_slug='write-disc-1', status=sr.Status.RUNNING)
        sr.write_record(record, root=tmp_path)

        decision = sr.DecisionRecord(
            id='dec-1',
            project='df',
            text='Which port?',
            filed_at='2026-07-07T00:00:00+00:00',
        )
        assert sr.write_decision(decision, root=tmp_path)

        before = _snapshot_tree(tmp_path)
        # sanity: prove the snapshot actually captured the seeded files, so a
        # later "unchanged" assertion isn't vacuously true over an empty dict.
        assert any(path.startswith('sessions/') for path in before)
        assert any(path.startswith('decisions/') for path in before)

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)

            await pilot.pause(0.2)  # let at least one poll tick land

            table.move_cursor(row=table.get_row_index('write-disc-1'))
            await pilot.pause()

        after = _snapshot_tree(tmp_path)

        for path, value in before.items():
            assert after.get(path) == value, f'{path} was created/modified/removed by the cockpit'

        new_or_modified = set(after) - set(before)
        assert new_or_modified == {'cockpit-ui.json'}, (
            f'expected only cockpit-ui.json as a new path, got {new_or_modified}'
        )
