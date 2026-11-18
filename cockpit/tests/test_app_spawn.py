"""CockpitApp pilot tests for the spawn bar picker and the spawn-tree modal."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from _cockpit_session_records import make_record
from orchestrator import session_records

from cockpit.app import CockpitApp


class TestSpawnBar:
    @pytest.mark.timeout(10)
    async def test_new_session_key_pushes_spawn_screen(self, tmp_path):
        """'n' (PRD §9 C5b spawn bar) pushes a SpawnScreen project/role/prompt picker."""
        from cockpit.panes.spawn_bar import SpawnScreen

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05, spawn_runner=lambda argv: None)
        async with app.run_test() as pilot:
            await pilot.pause()

            await pilot.press('n')
            await pilot.pause()

            assert isinstance(app.screen, SpawnScreen)

    @pytest.mark.timeout(10)
    async def test_spawn_session_invokes_runner_with_exact_argv(self, tmp_path):
        """spawn_session(project_root, role, prompt) -- the spawn bar's leaf
        signal (PRD §9 C5b) -- builds spawn-claude.sh's exact positional
        argv (build_spawn_argv) from the picked project (cwd), a
        role-derived title, and the prompt, then hands it to the injected
        spawn_runner. No real terminal/process is launched here (the
        runner is faked) -- driven directly per this task's plan, rather
        than through SpawnScreen's own Input/Select widgets.
        """
        from cockpit.panes.spawn_bar import build_spawn_argv

        spawned: list[list[str]] = []
        spawn_script = '/repo/skills/spawn/spawn-claude.sh'

        app = CockpitApp(
            fleet_root=tmp_path,
            poll_interval=0.05,
            spawn_runner=spawned.append,
            spawn_script=spawn_script,
        )
        async with app.run_test() as pilot:
            await pilot.pause()

            project_root = '/home/leo/src/dark-factory'
            role = 'unblock'
            prompt = 'Please look at this'
            app.spawn_session(project_root, role, prompt)
            await pilot.pause()

        assert len(spawned) == 1
        expected_title = f'{role}:{Path(project_root).name}'
        assert spawned[0] == build_spawn_argv(spawn_script, project_root, expected_title, prompt)

    @pytest.mark.timeout(10)
    async def test_spawn_session_forwards_skip_perms_true(self, tmp_path):
        """spawn_session(..., skip_perms=True) (F9 fix, task 2518) threads
        through to build_spawn_argv, so argv[2] is the 'true' literal
        spawn-claude.sh's positional contract expects."""
        spawned: list[list[str]] = []
        spawn_script = '/repo/skills/spawn/spawn-claude.sh'

        app = CockpitApp(
            fleet_root=tmp_path,
            poll_interval=0.05,
            spawn_runner=spawned.append,
            spawn_script=spawn_script,
        )
        async with app.run_test() as pilot:
            await pilot.pause()

            app.spawn_session('/home/leo/src/dark-factory', 'unblock', 'Please look at this', skip_perms=True)
            await pilot.pause()

        assert len(spawned) == 1
        assert spawned[0][2] == 'true'

    @pytest.mark.timeout(10)
    async def test_spawn_session_forwards_skip_perms_false(self, tmp_path):
        """spawn_session(..., skip_perms=False) (F9 fix, task 2518) threads
        through to build_spawn_argv, so argv[2] is the 'false' literal."""
        spawned: list[list[str]] = []
        spawn_script = '/repo/skills/spawn/spawn-claude.sh'

        app = CockpitApp(
            fleet_root=tmp_path,
            poll_interval=0.05,
            spawn_runner=spawned.append,
            spawn_script=spawn_script,
        )
        async with app.run_test() as pilot:
            await pilot.pause()

            app.spawn_session('/home/leo/src/dark-factory', 'unblock', 'Please look at this', skip_perms=False)
            await pilot.pause()

        assert len(spawned) == 1
        assert spawned[0][2] == 'false'

    @pytest.mark.timeout(10)
    async def test_spawn_screen_checkbox_seeded_from_true_default_and_submitted(self, tmp_path):
        """SpawnScreen's skip-perms Checkbox (F9 fix, task 2518) is seeded
        from the injected default_skip_perms ctor param, and submitting
        passes the checkbox's value through as the on_submit callback's
        4th arg."""
        from textual.widgets import Checkbox

        from cockpit.panes.spawn_bar import SpawnScreen

        calls: list[tuple[str, str, str, bool]] = []

        def on_submit(project_root: str, role: str, prompt: str, skip_perms: bool) -> None:
            calls.append((project_root, role, prompt, skip_perms))

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05, spawn_runner=lambda argv: None)
        async with app.run_test() as pilot:
            await pilot.pause()

            app.push_screen(
                SpawnScreen(
                    ['/home/leo/src/dark-factory'],
                    on_submit,
                    default_skip_perms=True,
                )
            )
            await pilot.pause()

            checkbox = app.screen.query_one('#spawn-skip-perms', Checkbox)
            assert checkbox.value is True

            await pilot.click('#spawn-submit')
            await pilot.pause()

        assert calls == [('/home/leo/src/dark-factory', 'unblock', '', True)]

    @pytest.mark.timeout(10)
    async def test_spawn_screen_checkbox_toggled_on_overrides_false_default(self, tmp_path):
        """Toggling SpawnScreen's skip-perms Checkbox on overrides a False
        injected default -- the per-spawn toggle wins over the operator's
        env-configured default (F9 fix, task 2518)."""
        from textual.widgets import Checkbox

        from cockpit.panes.spawn_bar import SpawnScreen

        calls: list[tuple[str, str, str, bool]] = []

        def on_submit(project_root: str, role: str, prompt: str, skip_perms: bool) -> None:
            calls.append((project_root, role, prompt, skip_perms))

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05, spawn_runner=lambda argv: None)
        async with app.run_test() as pilot:
            await pilot.pause()

            app.push_screen(
                SpawnScreen(
                    ['/home/leo/src/dark-factory'],
                    on_submit,
                    default_skip_perms=False,
                )
            )
            await pilot.pause()

            checkbox = app.screen.query_one('#spawn-skip-perms', Checkbox)
            assert checkbox.value is False
            checkbox.value = True
            await pilot.pause()

            await pilot.click('#spawn-submit')
            await pilot.pause()

        assert len(calls) == 1
        assert calls[0][3] is True

    @pytest.mark.timeout(10)
    async def test_action_new_session_seeds_skip_perms_checkbox_from_env_true(self, tmp_path, monkeypatch):
        """End-to-end F9 fix (task 2518): with $CLAUDE_SPAWN_SKIP_PERMS set,
        'n' (action_new_session) pushes a SpawnScreen whose skip-perms
        Checkbox is already True, and submitting drives
        env -> screen -> spawn_session -> build_spawn_argv -> runner,
        landing the 'true' literal spawn-claude.sh's positional contract
        expects at argv[2]."""
        from textual.widgets import Checkbox, Input

        from cockpit.panes.spawn_bar import SpawnScreen

        monkeypatch.setenv('CLAUDE_SPAWN_SKIP_PERMS', 'true')

        spawned: list[list[str]] = []
        spawn_script = '/repo/skills/spawn/spawn-claude.sh'

        app = CockpitApp(
            fleet_root=tmp_path,
            poll_interval=0.05,
            spawn_runner=spawned.append,
            spawn_script=spawn_script,
        )
        async with app.run_test() as pilot:
            await pilot.pause()

            await pilot.press('n')
            await pilot.pause()

            assert isinstance(app.screen, SpawnScreen)
            checkbox = app.screen.query_one('#spawn-skip-perms', Checkbox)
            assert checkbox.value is True

            app.screen.query_one('#spawn-project', Input).value = '/home/leo/src/dark-factory'
            app.screen.query_one('#spawn-role', Input).value = 'unblock'
            app.screen.query_one('#spawn-prompt', Input).value = 'Please look at this'
            await pilot.pause()

            await pilot.click('#spawn-submit')
            await pilot.pause()

        assert len(spawned) == 1
        assert spawned[0][2] == 'true'

    @pytest.mark.timeout(10)
    async def test_action_new_session_seeds_skip_perms_checkbox_false_when_env_unset(self, tmp_path, monkeypatch):
        """Complementary env-unset case (F9 fix, task 2518): with
        $CLAUDE_SPAWN_SKIP_PERMS unset, 'n' pushes a SpawnScreen whose
        skip-perms Checkbox is seeded False -- the safe default."""
        from textual.widgets import Checkbox

        from cockpit.panes.spawn_bar import SpawnScreen

        monkeypatch.delenv('CLAUDE_SPAWN_SKIP_PERMS', raising=False)

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05, spawn_runner=lambda argv: None)
        async with app.run_test() as pilot:
            await pilot.pause()

            await pilot.press('n')
            await pilot.pause()

            assert isinstance(app.screen, SpawnScreen)
            checkbox = app.screen.query_one('#spawn-skip-perms', Checkbox)
            assert checkbox.value is False


class TestSpawnTree:
    @pytest.mark.timeout(10)
    async def test_toggle_key_pushes_tree_screen_with_structure(self, tmp_path):
        """'t' (Fleet Cockpit C9a spawn-tree toggle) pushes a SpawnTreeScreen
        rendering the parent_session_id parent→child session-tree structure."""
        from cockpit.panes.spawn_tree import SpawnTree, SpawnTreeScreen

        parent = make_record(session_slug='parent-1', parent_session_id=None)
        child = make_record(session_slug='child-1', parent_session_id='parent-1')
        for r in (parent, child):
            session_records.write_record(r, root=tmp_path)

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            await pilot.press('t')
            await pilot.pause()

            assert isinstance(app.screen, SpawnTreeScreen)
            tree = app.screen.query_one(SpawnTree)
            top_level_slugs = {node.data for node in tree.root.children}
            assert 'parent-1' in top_level_slugs

            parent_node = next(node for node in tree.root.children if node.data == 'parent-1')
            child_slugs = {node.data for node in parent_node.children}
            assert child_slugs == {'child-1'}

    @pytest.mark.timeout(10)
    async def test_outstanding_children_are_highlighted(self, tmp_path):
        """Outstanding (non-terminal) children are visually marked in the
        rendered tree (Fleet Cockpit C9a, PRD §9) -- pins the "outstanding
        children highlighted" half of the leaf signal at the widget level."""
        from cockpit.panes.spawn_tree import _OUTSTANDING_MARKER, SpawnTree, SpawnTreeScreen

        parent = make_record(session_slug='parent-1', parent_session_id=None)
        running_child = make_record(
            session_slug='child-running',
            parent_session_id='parent-1',
            status=session_records.Status.RUNNING,
        )
        exited_child = make_record(
            session_slug='child-exited',
            parent_session_id='parent-1',
            status=session_records.Status.EXITED,
        )
        for r in (parent, running_child, exited_child):
            session_records.write_record(r, root=tmp_path)

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            await pilot.press('t')
            await pilot.pause()

            assert isinstance(app.screen, SpawnTreeScreen)
            tree = app.screen.query_one(SpawnTree)
            parent_node = next(node for node in tree.root.children if node.data == 'parent-1')
            nodes_by_slug = {node.data: node for node in parent_node.children}

            running_label = str(nodes_by_slug['child-running'].label)
            exited_label = str(nodes_by_slug['child-exited'].label)

            assert _OUTSTANDING_MARKER in running_label
            assert _OUTSTANDING_MARKER not in exited_label


class TestSpawnTreeEnterFocus:
    @pytest.mark.timeout(10)
    async def test_enter_focuses_the_highlighted_child_node(self, tmp_path):
        """Enter on the spawn tree's CHILD node (deliberately not the
        parent/row-0) resolves that child's own Display and routes it to
        the focus backend -- the C9a leaf signal (PRD §9), mirroring
        test_app.py::TestEnterFocus's decision-queue contract at the tree-toggle level.
        """
        from cockpit.backends import DisplayTarget, FakeBackend
        from cockpit.panes.spawn_tree import SpawnTree, SpawnTreeScreen

        parent_display = session_records.Display(kind='wm', wm_title='parent title')
        child_display = session_records.Display(kind='wm', wm_title='child title')
        parent = make_record(
            session_slug='parent-1', parent_session_id=None, display=parent_display
        )
        child = make_record(
            session_slug='child-1', parent_session_id='parent-1', display=child_display
        )
        for r in (parent, child):
            session_records.write_record(r, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            await pilot.press('t')
            await pilot.pause()

            assert isinstance(app.screen, SpawnTreeScreen)
            tree = app.screen.query_one(SpawnTree)
            parent_node = next(node for node in tree.root.children if node.data == 'parent-1')
            child_node = next(node for node in parent_node.children if node.data == 'child-1')

            tree.move_cursor(child_node)
            await pilot.pause()

            await pilot.press('enter')
            await pilot.pause()

            assert backend.focus_calls == [DisplayTarget(kind='wm', wm_title='child title')]

    @pytest.mark.timeout(10)
    async def test_enter_on_child_with_no_display_is_fail_soft(self, tmp_path):
        """A child with no Display resolves to no focus target -- pressing
        Enter on it must not focus anything and must not raise (PRD §2)."""
        from cockpit.backends import FakeBackend
        from cockpit.panes.spawn_tree import SpawnTree, SpawnTreeScreen

        parent = make_record(session_slug='parent-1', parent_session_id=None)
        child = make_record(
            session_slug='child-1', parent_session_id='parent-1', display=None
        )
        for r in (parent, child):
            session_records.write_record(r, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            await pilot.press('t')
            await pilot.pause()

            assert isinstance(app.screen, SpawnTreeScreen)
            tree = app.screen.query_one(SpawnTree)
            parent_node = next(node for node in tree.root.children if node.data == 'parent-1')
            child_node = next(node for node in parent_node.children if node.data == 'child-1')

            tree.move_cursor(child_node)
            await pilot.pause()

            await pilot.press('enter')
            await pilot.pause()

            assert backend.focus_calls == []

    @pytest.mark.timeout(10)
    async def test_focus_slug_missing_from_snapshot_is_fail_soft(self, tmp_path):
        """A slug that isn't in the snapshot passed to _focus_slug (e.g. the
        opened tree's own snapshot, once stale) resolves to no record at all
        -- the `record is None` branch -- and must not raise or focus
        anything (PRD §2), mirroring the no-Display fail-soft case above."""
        from cockpit.backends import FakeBackend

        some_record = make_record(session_slug='parent-1', parent_session_id=None)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test():
            app._focus_slug('nonexistent-slug', [some_record])

            assert backend.focus_calls == []

    @pytest.mark.timeout(10)
    async def test_enter_resolves_against_the_opened_snapshot_not_live_records(self, tmp_path):
        """Enter must resolve against the snapshot action_toggle_tree closed
        over when the tree was opened, not whatever self._records has been
        reassigned to by an intervening refresh_registry poll tick -- see
        action_toggle_tree's and _focus_slug's docstrings. The reassignment
        is reproduced via a *real* refresh_registry() scan (not a hand-poked
        app._records assignment): the child's on-disk record is removed
        first, so the scan legitimately drops it and self._records
        genuinely no longer contains the child by the time Enter is
        pressed. This keeps the guard deterministic -- unlike overwriting
        app._records directly, which the still-running poll_interval=0.05
        background poller could silently re-populate from the (unchanged)
        on-disk child before Enter lands, since nothing about that
        alternate approach stops disk from still backing the child.
        Enter must still focus the original child's Display, exactly as if
        the poll tick had never landed."""
        from cockpit.backends import DisplayTarget, FakeBackend
        from cockpit.panes.spawn_tree import SpawnTree, SpawnTreeScreen

        parent_display = session_records.Display(kind='wm', wm_title='parent title')
        child_display = session_records.Display(kind='wm', wm_title='child title')
        parent = make_record(
            session_slug='parent-1', parent_session_id=None, display=parent_display
        )
        child = make_record(
            session_slug='child-1', parent_session_id='parent-1', display=child_display
        )
        for r in (parent, child):
            session_records.write_record(r, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            await pilot.press('t')
            await pilot.pause()

            assert isinstance(app.screen, SpawnTreeScreen)
            tree = app.screen.query_one(SpawnTree)
            parent_node = next(node for node in tree.root.children if node.data == 'parent-1')
            child_node = next(node for node in parent_node.children if node.data == 'child-1')

            tree.move_cursor(child_node)
            await pilot.pause()

            # Simulate an intervening refresh_registry poll tick that
            # replaces self._records wholesale with a set no longer
            # containing the highlighted child -- deterministically: the
            # child's record.json is gone from disk, so this scan (like
            # any later background poll tick) can only ever find parent-1.
            shutil.rmtree(session_records.record_path_for_slug('child-1', root=tmp_path).parent)
            app.refresh_registry()
            await pilot.pause()
            # sanity: the simulated poll tick actually dropped the child,
            # so the assertion below isn't vacuously true.
            assert [r.session_slug for r in app._records] == ['parent-1']

            await pilot.press('enter')
            await pilot.pause()

            assert backend.focus_calls == [DisplayTarget(kind='wm', wm_title='child title')]


class TestSpawnTreeToggle:
    @pytest.mark.timeout(10)
    async def test_toggle_key_closes_an_already_open_tree(self, tmp_path):
        """'t' toggles: pressing it again while the tree is open closes it
        (back to the default screen) rather than leaving it open or
        pushing a second modal on top (Fleet Cockpit C9a, PRD §9)."""
        from cockpit.panes.spawn_tree import SpawnTreeScreen

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            await pilot.press('t')
            await pilot.pause()
            assert isinstance(app.screen, SpawnTreeScreen)

            await pilot.press('t')
            await pilot.pause()
            assert not isinstance(app.screen, SpawnTreeScreen)

    @pytest.mark.timeout(10)
    async def test_escape_closes_the_open_tree(self, tmp_path):
        """'escape' also dismisses the open spawn-tree modal, mirroring
        SpawnScreen.action_cancel's own escape-to-dismiss binding."""
        from cockpit.panes.spawn_tree import SpawnTreeScreen

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            await pilot.press('t')
            await pilot.pause()
            assert isinstance(app.screen, SpawnTreeScreen)

            await pilot.press('escape')
            await pilot.pause()
            assert not isinstance(app.screen, SpawnTreeScreen)
