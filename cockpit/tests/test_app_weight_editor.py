"""CockpitApp pilot tests for the 'w' weight editor and the priorities file it persists."""

from __future__ import annotations

from datetime import datetime

import pytest
from orchestrator import decision_registry as dr

from cockpit.app import CockpitApp


class TestWeightEditorReordersAndPersists:
    @pytest.mark.timeout(10)
    async def test_apply_priorities_reorders_live_and_persists(self, tmp_path):
        """apply_priorities (PRD §9 C9b weight editor) is this task's leaf
        signal: bumping a project weight reorders the DecisionQueue
        IMMEDIATELY -- no poll tick, no explicit refresh_registry() call --
        and persists the change to priorities.yaml via C3's save_priorities.

        Driven directly (mirrors
        test_app_spawn.py::TestSpawnBar::test_spawn_session_invokes_runner_with_exact_argv
        driving app.spawn_session), ahead of the 'w' keybinding/screen wiring
        added later in this task. Two decisions in projects 'alpha'/'beta'
        with identical filed_at and boost=0 score equally under
        Priorities.default() (project_weights={}), so the initial order is
        decided purely by the (decision:alpha < decision:beta) key tiebreak
        -- the "another factor" a project-weight bump on 'beta' alone must
        overcome to flip the order.
        """
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue
        from cockpit.panes.weight_editor import merge_weight_edits
        from cockpit.priority import Priorities, load_priorities

        fixed_now = datetime.fromisoformat('2026-07-07T00:00:00+00:00')

        alpha = dr.DecisionRecord(
            id='alpha',
            project='alpha',
            text='Alpha?',
            filed_at='2026-07-07T00:00:00+00:00',
            manual_boost=0,
        )
        beta = dr.DecisionRecord(
            id='beta',
            project='beta',
            text='Beta?',
            filed_at='2026-07-07T00:00:00+00:00',
            manual_boost=0,
        )
        for d in (alpha, beta):
            assert dr.write_decision(d, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(
            fleet_root=tmp_path,
            backend=backend,
            poll_interval=0.05,
            now_fn=lambda: fixed_now,
            priorities=Priorities.default(),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            assert queue.row_count == 2
            # sanity: equal score -> tiebroken by key, alpha before beta.
            assert queue.get_row_index('decision:alpha') < queue.get_row_index('decision:beta')

            new_priorities = merge_weight_edits(
                app._priorities, category_edits={}, project_edits={'beta': '100'}
            )
            app.apply_priorities(new_priorities)
            await pilot.pause()

            # (a) live reorder -- no poll tick, no explicit
            # refresh_registry() call anywhere in this test.
            assert queue.get_row_index('decision:beta') < queue.get_row_index('decision:alpha')

        # (b) persisted to priorities.yaml -- checked after unmount so this
        # also proves the write landed on disk, not just in memory.
        persisted = load_priorities(tmp_path / 'priorities.yaml')
        assert persisted.project_weights['beta'] == 100.0

    @pytest.mark.timeout(10)
    async def test_reorder_survives_a_silently_failed_persist(self, tmp_path, monkeypatch):
        """apply_priorities' persist half is fail-soft, by contract, all the
        way down: save_priorities never raises (see
        TestSavePriorities::test_write_failure_is_fail_soft_and_warns /
        test_non_os_error_during_write_is_fail_soft_and_warns in
        test_priorities_config.py). This pins the SAME guarantee at the app
        boundary apply_priorities itself relies on: self._priorities is
        already swapped and save_priorities is called before
        _rebuild_queue() runs, so if a persist attempt were ever to become a
        silent no-op (the worst case save_priorities' own fail-soft
        contract allows), the live reorder must still happen with no
        exception escaping apply_priorities -- an operator's in-session view
        must never be held hostage by a write fault (PRD §2). No production
        code change -- cockpit.app.save_priorities is monkeypatched here to
        a no-op stand-in for the worst case, not to prove a NEW behavior.
        """
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue
        from cockpit.panes.weight_editor import merge_weight_edits
        from cockpit.priority import Priorities

        fixed_now = datetime.fromisoformat('2026-07-07T00:00:00+00:00')

        alpha = dr.DecisionRecord(
            id='alpha',
            project='alpha',
            text='Alpha?',
            filed_at='2026-07-07T00:00:00+00:00',
            manual_boost=0,
        )
        beta = dr.DecisionRecord(
            id='beta',
            project='beta',
            text='Beta?',
            filed_at='2026-07-07T00:00:00+00:00',
            manual_boost=0,
        )
        for d in (alpha, beta):
            assert dr.write_decision(d, root=tmp_path)

        monkeypatch.setattr('cockpit.app.save_priorities', lambda *args, **kwargs: None)

        backend = FakeBackend()
        app = CockpitApp(
            fleet_root=tmp_path,
            backend=backend,
            poll_interval=0.05,
            now_fn=lambda: fixed_now,
            priorities=Priorities.default(),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            assert queue.get_row_index('decision:alpha') < queue.get_row_index('decision:beta')

            new_priorities = merge_weight_edits(
                app._priorities, category_edits={}, project_edits={'beta': '100'}
            )
            app.apply_priorities(new_priorities)
            await pilot.pause()

            # Live reorder happened despite the no-op'd save_priorities --
            # apply_priorities never propagated an exception (this pilot
            # session would have errored out already if it had).
            assert queue.get_row_index('decision:beta') < queue.get_row_index('decision:alpha')

        # Nothing was ever written -- the no-op stand-in is a faithful
        # stand-in for a real, silently-swallowed write fault.
        assert not (tmp_path / 'priorities.yaml').exists()


class TestPrioritiesPathResolution:
    @pytest.mark.timeout(10)
    async def test_load_and_save_target_the_same_fleet_root_relative_path(self, tmp_path):
        """apply_priorities persists to self._priorities_path, derived as
        resolve_fleet_root(self.fleet_root)/'priorities.yaml' (see
        TestWeightEditorReordersAndPersists above for the write half of this
        contract). This pins the READ half: __init__'s in-app default is
        LOADED from that exact same fleet_root-relative path -- not the
        global ~/.claude/fleet default load_priorities() falls back to --
        so an explicit fleet_root=tmp_path CockpitApp is fully
        self-contained under tmp_path for both read and write. Regression
        test for a reviewer-flagged incidental behavior change (see
        app.py's design_decisions).
        """
        from dataclasses import replace

        from orchestrator.fleet_files import fleet_root as resolve_fleet_root

        from cockpit.priority import Priorities, save_priorities

        seeded = replace(Priorities.default(), project_weights={'df': 7.0})
        save_priorities(seeded, resolve_fleet_root(tmp_path) / 'priorities.yaml')

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)

        assert app._priorities_path == resolve_fleet_root(tmp_path) / 'priorities.yaml'
        # The seeded marker key is read back CANONICAL: load_priorities
        # folds project_weights keys (task 3812). Which file was read --
        # the point of this test -- is unaffected: the 7.0 value is the
        # marker, and the global default file has no such entry at all.
        assert app._priorities.project_weights == {'dark_factory': 7.0}


class TestWeightEditor:
    @pytest.mark.timeout(10)
    async def test_w_key_pushes_weight_editor_screen(self, tmp_path):
        """'w' (PRD §9 C9b weight editor) pushes a WeightEditorScreen (mirrors
        test_app_spawn.py::TestSpawnBar::test_new_session_key_pushes_spawn_screen)."""
        from cockpit.panes.weight_editor import WeightEditorScreen

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            await pilot.press('w')
            await pilot.pause()

            assert isinstance(app.screen, WeightEditorScreen)

    @pytest.mark.timeout(10)
    async def test_submit_reads_edited_input_into_new_priorities(self, tmp_path):
        """compose()->_submit() end-to-end against REAL widgets -- proves the
        positional cat-<index>/proj-<index> ids compose() assigns are the
        same ones _submit()'s query_one lookups read back (a name-derived id
        would BadIdentifier-crash for a name with '.'/'/'/':'/spaces), and
        that only the edited project Input reaches the on_submit callback's
        new Priorities -- an untouched project Input (still showing its
        seeded fallback) must NOT materialize a project_weights entry.
        """
        from textual.widgets import Input

        from cockpit.panes.weight_editor import WeightEditorScreen
        from cockpit.priority import Priorities

        received: list[Priorities] = []
        base = Priorities.default()

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            screen = WeightEditorScreen(base, ['alpha', 'beta'], received.append)
            await app.push_screen(screen)
            await pilot.pause()

            # project_names == ['alpha', 'beta'] -> compose() assigns
            # id 'proj-1' to 'beta' (enumerate order); 'alpha' (proj-0) is
            # left untouched at its seeded fallback.
            beta_input = screen.query_one('#proj-1', Input)
            beta_input.value = '42'
            await pilot.pause()

            await pilot.click('#weight-submit')
            await pilot.pause()

        assert len(received) == 1
        assert received[0].project_weights == {'beta': 42.0}
        assert received[0].category_weights == base.category_weights

    @pytest.mark.timeout(10)
    async def test_submit_reads_edited_category_input_into_new_priorities(self, tmp_path):
        """compose()->_submit() for a CATEGORY Input, symmetric with the
        project-Input coverage above (test_submit_reads_edited_input_into_new_priorities).

        category_edits uses the same value-vs-seeded skip as project_edits
        (see WeightEditorScreen._submit), so this also pins that an
        untouched category Input is skipped -- merge_weight_edits' base
        copy already carries its value forward unchanged -- while the
        edited one takes the new value.
        """
        from textual.widgets import Input

        from cockpit.panes.weight_editor import WeightEditorScreen
        from cockpit.priority import Priorities

        received: list[Priorities] = []
        base = Priorities.default()

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            screen = WeightEditorScreen(base, [], received.append)
            await app.push_screen(screen)
            await pilot.pause()

            # base.category_weights == {'security': 2.0, 'bug': 1.0, 'feature': 0.5,
            # 'chore': 0.2} (dict insertion order) -> compose() assigns id
            # 'cat-1' to 'bug'.
            bug_input = screen.query_one('#cat-1', Input)
            bug_input.value = '9.0'
            await pilot.pause()

            await pilot.click('#weight-submit')
            await pilot.pause()

        assert len(received) == 1
        expected_category_weights = dict(base.category_weights)
        expected_category_weights['bug'] = 9.0
        assert received[0].category_weights == expected_category_weights
        assert received[0].project_weights == {}

    @pytest.mark.timeout(10)
    async def test_escape_cancels_without_invoking_on_submit(self, tmp_path):
        """Escape (action_cancel) dismisses WITHOUT submitting -- mirrors
        test_app_spawn.py::TestSpawnTreeToggle::test_escape_closes_the_open_tree's
        escape-to-dismiss coverage for SpawnTreeScreen. Pushed with an injected on_submit collecting into
        *received* (same shape as the submit-path tests above), so this
        pins BOTH halves of the cancel contract at once: the screen is
        popped, AND on_submit is never called -- a regression that made
        cancel accidentally submit (or fail to dismiss) would leave
        *received* non-empty, or leave the screen on top, respectively.
        """
        from cockpit.panes.weight_editor import WeightEditorScreen
        from cockpit.priority import Priorities

        received: list[Priorities] = []
        base = Priorities.default()

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            screen = WeightEditorScreen(base, ['alpha', 'beta'], received.append)
            await app.push_screen(screen)
            await pilot.pause()
            assert isinstance(app.screen, WeightEditorScreen)

            await pilot.press('escape')
            await pilot.pause()

            assert not isinstance(app.screen, WeightEditorScreen)

        assert received == []

    @pytest.mark.timeout(10)
    async def test_submit_keeps_prior_value_for_a_rejected_category_edit(self, tmp_path):
        """A category Input made unparseable (garbage text) is silently
        dropped -- merge_weight_edits keeps *base*'s value for that key,
        never raises (see merge_weight_edits' own docstring: 'a weight is
        cleared by typing 0, not by blanking a field'). This pins that
        fail-soft contract through the FULL compose->_submit pipeline (not
        just the pure-helper coverage in TestMergeWeightEdits), symmetric
        with test_submit_reads_edited_category_input_into_new_priorities'
        coverage of the valid-edit path.

        Visible operator feedback for a rejected edit (e.g. re-seeding the
        Input to its prior value, a style change) was considered and
        intentionally deferred: _submit() dismisses the screen immediately
        after building new_priorities, so any Input-value reset would land
        on an already-torn-down widget and never be seen, and this
        narrowly-scoped persistence task is not the place to introduce a
        new Textual validation/notification pattern with no existing
        precedent in this codebase. See WeightEditorScreen._submit's
        docstring.
        """
        from textual.widgets import Input

        from cockpit.panes.weight_editor import WeightEditorScreen
        from cockpit.priority import Priorities

        received: list[Priorities] = []
        base = Priorities.default()

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            screen = WeightEditorScreen(base, [], received.append)
            await app.push_screen(screen)
            await pilot.pause()

            # base.category_weights == {'security': 2.0, 'bug': 1.0,
            # 'feature': 0.5, 'chore': 0.2} (dict insertion order) ->
            # compose() assigns id 'cat-1' to 'bug'.
            bug_input = screen.query_one('#cat-1', Input)
            bug_input.value = 'not-a-number'
            await pilot.pause()

            await pilot.click('#weight-submit')
            await pilot.pause()

        assert len(received) == 1
        assert received[0].category_weights == base.category_weights
        assert received[0].project_weights == {}

    @pytest.mark.timeout(10)
    async def test_submit_skips_project_input_retyped_to_its_seeded_value(self, tmp_path):
        """A project Input explicitly retyped to exactly its already-displayed
        seeded value must NOT materialize a project_weights entry either --
        this directly pins the `value == seeded` half of _submit()'s skip
        branch (`if value is None or value == seeded: continue`), symmetric
        with the `value is None` half pinned by
        test_submit_keeps_prior_value_for_a_rejected_category_edit above.

        test_submit_reads_edited_input_into_new_priorities' untouched
        'alpha' Input happens to hit this same branch too (an Input's value
        never changes from its compose()-time seed if nothing sets it), but
        only incidentally -- this test exercises the comparison directly by
        assigning the Input's value explicitly, the same observable state a
        real retype-then-revert would leave it in.
        """
        from textual.widgets import Input

        from cockpit.panes.weight_editor import WeightEditorScreen
        from cockpit.priority import Priorities

        received: list[Priorities] = []
        base = Priorities.default()

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            screen = WeightEditorScreen(base, ['alpha'], received.append)
            await app.push_screen(screen)
            await pilot.pause()

            # project_names == ['alpha'] -> compose() assigns id 'proj-0',
            # seeded from base.project_weights.get('alpha', base.defaults.project)
            # == base.defaults.project (base.project_weights is {}).
            seeded = base.project_weights.get('alpha', base.defaults.project)
            alpha_input = screen.query_one('#proj-0', Input)
            alpha_input.value = str(seeded)
            await pilot.pause()

            await pilot.click('#weight-submit')
            await pilot.pause()

        assert len(received) == 1
        assert received[0].project_weights == {}

    @pytest.mark.timeout(10)
    async def test_duplicate_project_names_are_deduped_without_clobbering(self, tmp_path):
        """WeightEditorScreen is constructed directly with an arbitrary
        Sequence[str] of project names -- known_projects() (the production
        caller) always returns an already-deduped sorted union, but the
        screen itself must not depend on that. compose() enumerates
        self._project_names to assign each row's id/seed, and _submit()
        re-enumerates it into a NAME-keyed project_edits dict, so before
        __init__ deduped defensively, a duplicate name would have rendered
        two indistinguishable rows and then let the later-index row's read
        silently clobber the earlier one's entry in that dict. This pins
        both halves: only ONE row per distinct name exists after dedup, and
        editing it reaches on_submit intact.
        """
        from textual.widgets import Input

        from cockpit.panes.weight_editor import WeightEditorScreen
        from cockpit.priority import Priorities

        received: list[Priorities] = []
        base = Priorities.default()

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            screen = WeightEditorScreen(base, ['alpha', 'alpha', 'beta'], received.append)
            await app.push_screen(screen)
            await pilot.pause()

            assert isinstance(app.screen, WeightEditorScreen)

            # Deduped (order-preserving) to ['alpha', 'beta'] -> exactly two
            # project rows exist; a 'proj-2' would only exist if the
            # duplicate 'alpha' had NOT been collapsed.
            assert screen.query_one('#proj-0', Input).placeholder == 'alpha'
            assert screen.query_one('#proj-1', Input).placeholder == 'beta'
            assert len(screen.query('#proj-2')) == 0

            beta_input = screen.query_one('#proj-1', Input)
            beta_input.value = '7.0'
            await pilot.pause()

            await pilot.click('#weight-submit')
            await pilot.pause()

        assert len(received) == 1
        assert received[0].project_weights == {'beta': 7.0}
