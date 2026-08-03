"""Contract tests for ``conftest.py`` correctness.

This module guards six invariants that would silently regress under refactoring
and are NOT already covered by other tests:

1. **sys.path ordering / module resolution** — ``conftest.py`` must insert
   worktree-local source directories onto ``sys.path`` *before* any
   ``from orchestrator`` or ``from shared`` import, so that worktree-local
   code is the version actually loaded (verified behaviorally via ``__file__``).

2. **Top-level ``spec_set`` wiring** — the top-level ``MagicMock`` in
   ``mock_orch_config`` must use ``spec_set=pydantic_spec(OrchestratorConfig)``
   so that typos raise ``AttributeError`` immediately.  Downstream harness
   consumers only assign *valid* top-level fields, so they would pass silently
   even if ``spec_set`` were dropped — this test does not.

3. **Sub-section ``spec_set`` wiring** — each sub-section of
   ``mock_orch_config`` (usage_cap, review, sandbox, fused_memory, escalation)
   must be ``spec_set``'d so that typos raise ``AttributeError`` immediately
   rather than silently creating phantom attributes.

4. **``@property`` descriptor exposure** — ``pydantic_spec`` must expose
   user-defined ``@property`` descriptors (e.g. ``overrides_db_path``) in
   the proxy class so ``spec_set`` accepts both read and write.
   BaseModel-inherited properties (``model_extra``, ``model_fields_set``, …)
   and BaseModel methods (``model_dump``, …) must remain excluded.

5. **``mock_orch_config.overrides_db_path`` is a real ``Path``** — the
   fixture must set a concrete ``Path`` under ``tmp_path`` for
   ``overrides_db_path`` so ``OverrideStore.from_config(config)`` can call
   ``.parent.mkdir()`` and ``sqlite3.connect(str(...))`` without crashing.

6. **``make_steward`` owns its worktree** — the shared steward factory must
   root every worktree it builds strictly *below* the test's ``tmp_path``,
   whether the caller supplies one or not, so pytest's retention policy
   reclaims both the worktree and the ``.task-meta`` sibling the steward
   derives from it, and two default builds in one test must not collide.
   See ``TestMakeStewardFixture``.

Tests of plain attribute defaults (e.g. ``mock.usage_cap.enabled is False``)
are deliberately omitted — they would just duplicate literals from
``conftest.py`` two lines away.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest


def test_syspath_block_precedes_guarded_imports():
    """Worktree-local source is the version of ``orchestrator.config`` / ``shared.config_models`` loaded.

    Verifies the *behaviour* produced by conftest.py's sys.path ordering: the
    modules resolved at import time must come from the worktree-local src
    directories, not from an installed-package copy.

    If conftest.py's sys.path.insert block were moved *after* the guarded
    imports, Python would resolve to the installed-package version and this
    test would fail with a path mismatch.
    """
    import shared.config_models

    import orchestrator.config

    _src = (Path(__file__).parent.parent / 'src').resolve()
    _shared_src = (Path(__file__).parent.parent.parent / 'shared' / 'src').resolve()

    orch_file = Path(orchestrator.config.__file__).resolve()
    shared_file = Path(shared.config_models.__file__).resolve()

    assert orch_file.is_relative_to(_src), (
        f'orchestrator.config loaded from {orch_file} — expected a path under '
        f'{_src}. Check that the sys.path.insert block in conftest.py runs '
        f'before the guarded imports.'
    )
    assert shared_file.is_relative_to(_shared_src), (
        f'shared.config_models loaded from {shared_file} — expected a path '
        f'under {_shared_src}. Check that the sys.path.insert block in '
        f'conftest.py runs before the guarded imports.'
    )


def test_toplevel_typo_rejected(mock_orch_config):
    """Top-level ``spec_set`` wiring rejects unknown attribute names.

    Guards against a refactor that drops ``spec_set=pydantic_spec(OrchestratorConfig)``
    from the top-level ``MagicMock`` (e.g. switching to plain ``MagicMock()``
    or ``MagicMock(spec=...)``).  The 7 downstream harness consumers only
    assign *valid* top-level fields, so they would pass silently even if
    ``spec_set`` were removed — this test does not.
    """
    with pytest.raises(AttributeError):
        mock_orch_config.projcet_root = 'anything'


@pytest.mark.parametrize('attr_path', [
    ['usage_cap', 'enabld'],
    ['review', 'enabld'],
    ['sandbox', 'bakcend'],
    ['fused_memory', 'projcet_id'],
    ['escalation', 'hsot'],
])
def test_subsection_typo_rejected(mock_orch_config, attr_path):
    """Typos on spec_set'd sub-sections raise AttributeError on assignment.

    Guards the sub-section ``spec_set`` wiring on ``mock_orch_config``.  If a
    refactor accidentally drops ``spec_set=`` from a sub-section, typos would
    silently create phantom attributes instead of raising.  The 7 downstream
    harness consumers only set *known* top-level fields, so they would not
    catch a sub-section wiring regression — this test does.
    """
    obj = mock_orch_config
    for attr in attr_path[:-1]:
        obj = getattr(obj, attr)
    with pytest.raises(AttributeError):
        setattr(obj, attr_path[-1], 'anything')


def test_pydantic_spec_exposes_user_property_descriptors():
    """pydantic_spec exposes @property descriptors declared on the user's pydantic model.

    Regression pin for the bug introduced by task 1313: ``Harness.__init__:218``
    calls ``OverrideStore.from_config(config)``, which dereferences
    ``config.overrides_db_path`` — a ``@property`` on ``OrchestratorConfig``
    (config.py:741).  Without this fix, ``MagicMock(spec_set=pydantic_spec(
    OrchestratorConfig))`` rejects both read and write of ``overrides_db_path``
    with ``AttributeError``.

    This test FAILS on pre-fix pydantic_spec (model_fields only).
    """
    from _orch_helpers import pydantic_spec

    from orchestrator.config import OrchestratorConfig

    spec = pydantic_spec(OrchestratorConfig)
    m = MagicMock(spec_set=spec)
    # overrides_db_path is a @property on OrchestratorConfig — not in model_fields.
    _ = m.overrides_db_path            # read must not raise AttributeError
    m.overrides_db_path = Path('/x')   # write must not raise AttributeError


def test_pydantic_spec_excludes_basemodel_inherited_members():
    """pydantic_spec still rejects BaseModel methods and inherited properties.

    Preserves the invariant established by task 1064: the proxy class created
    by pydantic_spec must NOT expose BaseModel API surface (model_dump,
    model_validate, model_extra, model_fields_set, …).  Exposing those would
    let tests write ``mock.model_dump = ...`` without error, silently hiding
    bugs in consumers that call real pydantic methods.

    This test PASSES on current (pre-fix) pydantic_spec because the spec proxy
    only contains model_fields names — none of which are BaseModel API.  After
    step-2's fix (broader @property enumeration with BaseModel-inherited
    filtering), it must continue to pass.

    Both read *and* write are checked so that a regression that accidentally
    removes the ``_basemodel_attrs`` filter (e.g. a misspelling of the variable
    name) is caught on the write path as well as the read path.
    """
    from _orch_helpers import pydantic_spec

    from orchestrator.config import OrchestratorConfig

    m = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    # --- read-side checks ---
    with pytest.raises(AttributeError):
        _ = m.model_dump            # BaseModel method
    with pytest.raises(AttributeError):
        _ = m.model_extra           # BaseModel @property
    with pytest.raises(AttributeError):
        _ = m.model_fields_set      # BaseModel @property
    # --- write-side checks (guard against _basemodel_attrs filter regression) ---
    with pytest.raises(AttributeError):
        m.model_dump = MagicMock()      # BaseModel method — write must also raise
    with pytest.raises(AttributeError):
        m.model_validate = MagicMock()  # BaseModel classmethod — write must also raise


def test_pydantic_spec_exposes_user_defined_methods():
    """pydantic_spec exposes user-defined regular methods on the pydantic model.

    Regression pin for the ~18 ad-hoc ``_spec.for_module = None`` patches
    scattered across orchestrator tests.  Without this exposure,
    ``MagicMock(spec_set=pydantic_spec(OrchestratorConfig))`` rejects any
    access to ``OrchestratorConfig.for_module`` — a regular method at
    config.py:991 that is not in ``model_fields`` and is not a ``@property``.
    The patches were needed purely to work around the spec_set gap; once
    ``pydantic_spec`` includes user-defined methods, they are unnecessary.

    This test FAILS on pre-fix ``pydantic_spec`` (model_fields + properties
    only) with ``AttributeError: Mock object has no attribute 'for_module'``.
    """
    from _orch_helpers import pydantic_spec

    from orchestrator.config import OrchestratorConfig

    spec = pydantic_spec(OrchestratorConfig)
    m = MagicMock(spec_set=spec)
    # for_module is a regular method on OrchestratorConfig — not in model_fields.
    _ = m.for_module('mod_a')          # read/call must not raise AttributeError
    m.for_module = MagicMock()         # write must not raise AttributeError


def test_pydantic_spec_exposes_private_attributes():
    """pydantic_spec exposes Pydantic v2 PrivateAttr members on the pydantic model.

    Regression pin for the ``_spec._module_configs = None`` patch in
    ``test_merge_queue.py``.  ``OrchestratorConfig._module_configs`` is a
    Pydantic v2 ``PrivateAttr(default=None)`` (config.py:971) declared via
    Pydantic's ``__private_attributes__`` registry, NOT via ``model_fields``.
    Without this exposure, ``MagicMock(spec_set=pydantic_spec(OrchestratorConfig))``
    rejects access to ``_module_configs``.

    This test FAILS on the state after step-2 (methods exposed but PrivateAttrs
    still missing) with ``AttributeError: Mock object has no attribute
    '_module_configs'``.
    """
    from _orch_helpers import pydantic_spec

    from orchestrator.config import OrchestratorConfig

    spec = pydantic_spec(OrchestratorConfig)
    m = MagicMock(spec_set=spec)
    # _module_configs is a PrivateAttr on OrchestratorConfig — not in model_fields.
    _ = m._module_configs              # read must not raise AttributeError
    m._module_configs = {}             # write must not raise AttributeError


class TestInitHarnessStateForTest:
    """Contract tests for the ``_init_harness_state_for_test`` helper.

    Guards two invariants:

    1. **Digest counters initialised** — after ``Harness.__new__(Harness)``
       followed by ``_init_harness_state_for_test(h)``, the four task-1327
       AFK-hardening digest counters exist at their ``Harness.__init__``
       defaults.  Without the helper the attributes are absent and
       ``_maybe_write_digest`` raises ``AttributeError`` (now surfaced by the
       narrowed catch-all added in step-4; previously silently swallowed).

    2. **Safe on already-initialised harness** — calling the helper a second
       time on a harness that already has the four counters set does NOT raise.
       Idempotence on *pre-existing values* is NOT required (the helper
       unconditionally overwrites with defaults), but it must not crash so that
       stacked helpers remain safe in future fixtures.
    """

    def test_digest_counters_set_to_init_defaults(self, tmp_path) -> None:
        """Four task-1327 digest counters are present at their __init__ defaults.

        This test FAILS before step-2 because ``_init_harness_state_for_test``
        does not yet exist in ``_orch_helpers``.
        """
        from _orch_helpers import _init_harness_state_for_test

        from orchestrator.harness import Harness

        h = Harness.__new__(Harness)
        _init_harness_state_for_test(h)

        assert h._escalation_event_count == 0, (
            f'_escalation_event_count expected 0, got {h._escalation_event_count!r}'
        )
        assert h._last_digest_event_count == 0, (
            f'_last_digest_event_count expected 0, got {h._last_digest_event_count!r}'
        )
        assert h._ewa_value == 0.0, (
            f'_ewa_value expected 0.0, got {h._ewa_value!r}'
        )
        assert h._last_digest_window_end_iso == '', (
            f'_last_digest_window_end_iso expected \'\', got {h._last_digest_window_end_iso!r}'
        )

    def test_helper_does_not_crash_on_already_initialised_harness(self, tmp_path) -> None:
        """Calling the helper twice does not raise.

        Idempotence on pre-existing values is NOT guaranteed (the helper may
        reset them to defaults), but the call must succeed without exception so
        that future fixtures can safely stack helpers.
        """
        from _orch_helpers import _init_harness_state_for_test

        from orchestrator.harness import Harness

        h = Harness.__new__(Harness)
        _init_harness_state_for_test(h)    # first call — sets defaults
        _init_harness_state_for_test(h)    # second call — must not raise


def test_mock_orch_config_overrides_db_path_default(mock_orch_config, tmp_path):
    """mock_orch_config seeds overrides_db_path with a real Path under tmp_path.

    Guards the fixture-level cleanup (step-4): ``Harness.__init__`` calls
    ``OverrideStore.from_config(config)`` unconditionally, which does
    ``self.db_path.parent.mkdir(...)`` and ``sqlite3.connect(str(self.db_path))``.
    A child ``MagicMock`` value for ``overrides_db_path`` would let ``.parent``
    and ``.mkdir()`` silently no-op but then cause ``sqlite3.connect`` to open a
    file named ``'<MagicMock …>'`` under cwd, leaking filesystem state in CI.

    This test FAILS on pre-fix conftest.py (no default set for overrides_db_path).
    """
    db_path = mock_orch_config.overrides_db_path
    assert isinstance(db_path, Path), (
        f'expected overrides_db_path to be a Path, got {type(db_path).__name__!r}'
    )
    assert db_path.is_relative_to(tmp_path), (
        f'expected overrides_db_path under tmp_path={tmp_path}, got {db_path}'
    )
    assert db_path.suffix == '.db', (
        f'expected overrides_db_path to have .db suffix, got {db_path.name!r}'
    )


class TestMakeStewardFixture:
    """Contract tests for the ``make_steward`` conftest fixture-factory.

    ``make_steward`` merges the two near-identical ``_make_steward`` copies that
    used to live in ``test_suggestion_triage.py`` and
    ``test_workflow_state_machine_boundary.py`` (task 3461).  It is not the only
    steward factory in the suite — ``test_steward.py`` and
    ``test_out_of_band_routing.py`` still carry their own; see the fixture
    docstring in ``conftest.py``.

    Because it closes over ``tmp_path`` it can *own* the worktree directory
    rather than merely documenting a convention, which is what these tests pin:

    - the default worktree is fixture-owned, created, and distinct per call, so
      repeated default builds cannot share ``.task/`` or the ``.task-meta``
      sibling that holds ``verdicts/triage.json``;
    - a caller-supplied worktree is rejected unless it is strictly below
      ``tmp_path``, checked *before* any ``mkdir``;
    - ``config_overrides`` is applied last (the row-9 ``steward_max_attempts=1``
      divergence rides on this channel — it is the only field the two former
      factories genuinely disagreed on).

    Per this module's exclusion policy (see the module docstring), the scalar
    config defaults and the collaborators' canned return values are NOT asserted
    here: those literals live 20 lines away in ``conftest.py``, and the real
    regression detector for them is the migrated call sites themselves.
    """

    def test_default_worktree_is_created_under_tmp_path(self, make_steward, tmp_path):
        """``make_steward()`` with no arguments yields a created, tmp_path-rooted worktree.

        The steward's pre-flight auto-escalates "Worktree missing" on a path
        that is not a directory, and the row-9 call site reads ``.task/``, so
        both must exist.  Strictly-below-``tmp_path`` is the retention
        invariant: the steward derives its artifacts root as
        ``<worktree.parent>/.task-meta/<worktree.name>`` (see
        ``orchestrator/config.py`` ``TASK_META_DIRNAME`` and
        ``artifacts.TaskArtifacts.meta_root_for``), so a worktree AT
        ``tmp_path`` would put that sibling outside the dir pytest reclaims.
        """
        from orchestrator.steward import TaskSteward

        steward = make_steward()

        assert isinstance(steward, TaskSteward), (
            f'expected a TaskSteward, got {type(steward).__name__!r}'
        )
        wt = steward.worktree
        assert wt.is_dir(), f'expected make_steward() to create {wt}, but it is not a directory'
        assert wt.resolve() != tmp_path.resolve(), (
            f'default worktree must be strictly below tmp_path, got {wt} == tmp_path'
        )
        assert wt.resolve().is_relative_to(tmp_path.resolve()), (
            f'expected default worktree under tmp_path={tmp_path}, got {wt}'
        )
        assert (wt / '.task').is_dir(), (
            f'expected a .task subdirectory under {wt} — row 9 of '
            f'test_workflow_state_machine_boundary.py reads it'
        )

    def test_project_root_is_a_real_path_inside_the_sandbox(self, make_steward, tmp_path):
        """``config.project_root`` is a real ``Path`` under ``tmp_path``, not a bare mock.

        The sandbox invariant this refactor fixed: the retired triage factory
        set the ``/tmp/project`` literal, which pointed outside the test's tmp
        dir, so anything the steward wrote relative to it escaped pytest's
        retention sweep.  A ``MagicMock`` here would also silently satisfy every
        ``/``-join in the steward without producing a real directory.

        The scalar config defaults (``steward_max_attempts``, ``models.*``,
        ``escalation.port``, …) are deliberately NOT asserted — they would
        restate ``conftest.py``'s literals without detecting a regression.
        """
        project_root = make_steward().config.project_root
        assert isinstance(project_root, Path), (
            f'expected project_root to be a real Path, got {type(project_root).__name__!r}'
        )
        assert project_root.resolve().is_relative_to(tmp_path.resolve()), (
            f'expected project_root under tmp_path={tmp_path}, got {project_root} — '
            f'the old /tmp/project literal pointed outside the test sandbox'
        )

    def test_config_overrides_applied_after_defaults(self, make_steward):
        """``config_overrides`` wins over the defaults, and does not leak between builds.

        This is the row-9 boundary variant's only real divergence from the
        triage variant (``steward_max_attempts`` 1 vs 3), so it is the
        regression pin for the override channel that lets one factory serve
        both.  A sibling default-built steward must still report 3.
        """
        overridden = make_steward(config_overrides={'steward_max_attempts': 1})
        default = make_steward()

        assert overridden.config.steward_max_attempts == 1, (
            'config_overrides must be applied AFTER the defaults'
        )
        assert default.config.steward_max_attempts == 3, (
            'an override on one steward must not leak into a sibling build'
        )

    def test_config_mock_is_spec_set(self, make_steward):
        """The config mock keeps ``spec_set=pydantic_spec(OrchestratorConfig)``.

        Guards the move into conftest.py: both former factories built their
        config with ``MagicMock(spec_set=pydantic_spec(OrchestratorConfig))``,
        so a typo'd field name raised ``AttributeError`` on read and write.  A
        refactor that dropped ``spec_set=`` would silently create phantom
        attributes instead, and every existing call site would still pass.
        """
        steward = make_steward()
        with pytest.raises(AttributeError):
            steward.config.steward_max_attemptz = 1

    @pytest.mark.asyncio
    async def test_briefing_initial_prompt_is_awaitable(self, make_steward):
        """``briefing.build_steward_initial_prompt`` is awaitable, not a plain ``MagicMock``.

        Awaitability is a real contract rather than a value: the merged factory
        builds ``briefing`` as an ``AsyncMock`` but then *reassigns* this
        attribute, and reassigning it with a bare ``MagicMock`` would make the
        steward's ``await`` raise ``TypeError`` at every call site.  Its return
        literal, and the mcp/queue canned returns, are not asserted — those are
        conftest literals restated.
        """
        steward = make_steward()

        assert await steward.briefing.build_steward_initial_prompt(steward.task) is not None

    # --- enforcement: a caller-supplied worktree must be strictly below tmp_path ---

    def test_rejects_worktree_equal_to_tmp_path(self, make_steward, tmp_path):
        """``worktree=tmp_path`` is rejected — it leaks ``.task-meta`` out of the sandbox.

        This is the exact escape hatch task 3366 could only document.  The
        steward's artifacts root is a SIBLING of the worktree
        (``<worktree.parent>/.task-meta/<worktree.name>`` — see
        ``TASK_META_DIRNAME`` in ``orchestrator/config.py`` and
        ``TaskArtifacts.meta_root_for`` in ``orchestrator/artifacts.py``), so a
        worktree AT ``tmp_path`` puts that sibling at
        ``tmp_path.parent/.task-meta/<numbered-root>`` — outside the directory
        pytest's retention policy reclaims.
        """
        with pytest.raises(AssertionError, match='strictly below'):
            make_steward(worktree=tmp_path)

    def test_rejects_worktree_outside_tmp_path(self, make_steward, tmp_path):
        """A worktree outside ``tmp_path`` is rejected BEFORE the directory is created.

        The mkdir-ordering half matters: a guard that fired after ``mkdir``
        would litter the very location it is defending, leaving an unreclaimed
        directory next to the test's tmp dir on every failure.
        """
        escaped = tmp_path.parent / 'escaped-wt'
        assert not escaped.exists(), f'test precondition: {escaped} must not pre-exist'

        with pytest.raises(AssertionError, match='strictly below'):
            make_steward(worktree=escaped)

        assert not escaped.exists(), (
            f'the guard must reject {escaped} BEFORE mkdir — it was created anyway'
        )

    def test_two_stewards_in_one_test_are_both_accepted(self, make_steward, tmp_path):
        """Two distinct sub-paths of ``tmp_path`` both pass — the no-false-positive control.

        This case is what ruled out the cheaper module-level approximation
        ("the worktree must not already exist"): a test that needs two stewards
        must be able to build both.
        """
        a = make_steward(worktree=tmp_path / 'a')
        b = make_steward(worktree=tmp_path / 'b')

        assert a.worktree != b.worktree
        assert a.worktree.is_dir() and b.worktree.is_dir()

    def test_repeated_default_builds_are_isolated(self, make_steward, tmp_path):
        """Two default builds get *different* fixture-owned worktrees.

        Sharing one directory would be exactly the shared mutable state the
        fixture otherwise defends against: both stewards would write the same
        ``.task/`` and, via the ``<worktree.parent>/.task-meta/<worktree.name>``
        sibling, the same ``verdicts/triage.json`` — whose staleness semantics
        ``test_pre_triage_clears_stale_verdict_before_spawn``
        (test_suggestion_triage.py) depends on.  No call site builds two default
        stewards today; this pins isolation before one does.
        """
        first, second = make_steward(), make_steward()

        assert second.worktree != first.worktree, (
            'repeated default make_steward() builds must not share a worktree'
        )
        assert first.worktree.is_dir() and second.worktree.is_dir()
        for wt in (first.worktree, second.worktree):
            assert wt.resolve().is_relative_to(tmp_path.resolve()), (
                f'every default worktree must stay under tmp_path={tmp_path}, got {wt}'
            )
            assert (wt / '.task').is_dir(), f'expected a .task subdirectory under {wt}'

    # --- pass-through kwargs: task / event_store / cost_store (task 3514) ---

    def test_task_kwarg_is_passed_through_and_drives_task_id(self, make_steward):
        """``task=`` reaches ``steward.task`` verbatim and ``task_id`` derives from it.

        The folded-in ``test_out_of_band_routing.py`` call sites vary
        ``task['metadata']`` (``dispatch_count``, ``model_overrides``) and assert
        on what reaches ``resolve_and_record_route``, so a hardcoded task dict
        cannot serve them.  ``task_id`` must follow ``task['id']`` rather than
        staying pinned at the default — a steward whose ``task_id`` disagreed
        with ``task['id']`` would mis-key every artifact it writes.
        """
        task = {'id': '7', 'title': 't', 'description': 'd', 'metadata': {'dispatch_count': 2}}

        steward = make_steward(task=task)

        assert steward.task is task, (
            'make_steward(task=...) must pass the caller dict through by identity, '
            'so post-construction mutation of task["metadata"] is visible to the steward'
        )
        assert steward.task_id == '7', (
            f"expected task_id to derive from task['id'], got {steward.task_id!r}"
        )

    def test_event_store_is_passed_through(self, make_steward):
        """``event_store=`` reaches ``steward.event_store`` by identity."""
        sentinel = MagicMock()

        steward = make_steward(event_store=sentinel)

        assert steward.event_store is sentinel, (
            'make_steward(event_store=...) must forward the recorder verbatim — '
            'the routing tests assert on events it records'
        )

    def test_cost_store_is_passed_through(self, make_steward):
        """``cost_store=`` reaches ``steward.cost_store`` by identity."""
        sentinel = MagicMock()

        steward = make_steward(cost_store=sentinel)

        assert steward.cost_store is sentinel, (
            'make_steward(cost_store=...) must forward the store verbatim'
        )

    def test_passthrough_kwargs_default_to_todays_behaviour(self, make_steward):
        """Omitting the new kwargs leaves the pre-3514 defaults exactly as they were.

        The three already-migrated call sites pass none of them, so each new
        kwarg must be strictly additive: no store injected, and the default
        ``task_id`` unchanged.
        """
        steward = make_steward()

        assert steward.event_store is None, (
            'make_steward() with no event_store= must leave it unset (None)'
        )
        assert steward.cost_store is None, (
            'make_steward() with no cost_store= must leave it unset (None)'
        )
        assert steward.task_id == '42', (
            f'the default task_id must not move, got {steward.task_id!r}'
        )

    # --- config defaults: the KNOWN GAPS conftest used to list (task 3514) ---

    def test_routing_config_is_resolvable_not_magicmock(self, make_steward):
        """``config.routing.*`` carries real containers, i.e. ``stamp_stock_routing_config`` ran.

        Not cosmetic: the folded-in ``test_out_of_band_routing.py`` tests that do
        NOT patch the invoke seam reach the real ``resolve_route``, which does
        membership and dict ops (``candidate not in ...allowed_models``,
        ``...per_model_daily_ceiling_usd.get(...)``, ``for rule in ...rules``)
        that a bare child ``MagicMock`` satisfies falsely and then fails on
        deeper.  The stock *values* are ``_orch_helpers.py``'s literals and are
        deliberately not restated here — only that real containers arrived.
        """
        routing = make_steward().config.routing

        assert isinstance(routing.allowed_models, list), (
            f'expected routing.allowed_models to be a real list, got '
            f'{type(routing.allowed_models).__name__!r} — stamp_stock_routing_config did not run'
        )
        assert isinstance(routing.ladder, list), (
            f'expected routing.ladder to be a real list, got {type(routing.ladder).__name__!r}'
        )
        assert isinstance(routing.per_model_daily_ceiling_usd, dict), (
            f'expected routing.per_model_daily_ceiling_usd to be a real dict, got '
            f'{type(routing.per_model_daily_ceiling_usd).__name__!r}'
        )
        assert isinstance(routing.rules, list), (
            f'expected routing.rules to be a real list, got {type(routing.rules).__name__!r}'
        )

    def test_timeout_config_defaults_are_real_floats(self, make_steward):
        """The spawn/completion timeouts are real numbers, not child ``MagicMock``s.

        The steward compares these against elapsed wall-clock; a ``MagicMock``
        silently satisfies the comparison operator and yields a confusing
        ``TypeError`` (or a wrong branch) far from the missing default.
        """
        config = make_steward().config

        assert isinstance(config.timeouts.steward, float), (
            f'expected timeouts.steward to be a real float, got '
            f'{type(config.timeouts.steward).__name__!r}'
        )
        assert isinstance(config.steward_completion_timeout, float), (
            f'expected steward_completion_timeout to be a real float, got '
            f'{type(config.steward_completion_timeout).__name__!r}'
        )

    def test_fused_memory_config_defaults_are_real_strings(self, make_steward):
        """``config.fused_memory.url`` / ``.project_id`` are real strings.

        They are interpolated into the MCP config the steward hands its agent, so
        a ``MagicMock`` here reaches JSON serialization before it fails.
        """
        fused_memory = make_steward().config.fused_memory

        assert isinstance(fused_memory.url, str), (
            f'expected fused_memory.url to be a real str, got {type(fused_memory.url).__name__!r}'
        )
        assert isinstance(fused_memory.project_id, str), (
            f'expected fused_memory.project_id to be a real str, got '
            f'{type(fused_memory.project_id).__name__!r}'
        )

    # --- collaborator + worktree defaults, and the queue-override channel (3514) ---

    def test_queue_dir_is_a_real_created_dir_and_unique_per_build(
        self, make_steward, tmp_path,
    ):
        """The mock queue's ``queue_dir`` is real, created, sandboxed, and per-build.

        ``steward.py`` reads ``escalation_queue.queue_dir`` into the escalation-
        watcher argv, so it is real mutable state — the same reason
        ``test_repeated_default_builds_are_isolated`` pins the worktree.  Two
        default builds sharing one queue dir would be exactly that shared state,
        so it is derived from the fixture-owned (already per-build) worktree
        rather than a flat ``tmp_path/'escalations'`` literal.
        """
        first, second = make_steward(), make_steward()

        for steward in (first, second):
            qd = steward.escalation_queue.queue_dir
            assert isinstance(qd, Path), (
                f'expected queue_dir to be a real Path, got {type(qd).__name__!r}'
            )
            assert qd.is_dir(), f'expected make_steward to create queue_dir {qd}'
            assert qd.resolve().is_relative_to(tmp_path.resolve()), (
                f'queue_dir must stay under tmp_path={tmp_path} so pytest reclaims it, got {qd}'
            )

        assert first.escalation_queue.queue_dir != second.escalation_queue.queue_dir, (
            'repeated default make_steward() builds must not share a queue_dir — '
            'the escalation-watcher argv is built from it'
        )

    def test_mcp_url_is_a_real_string(self, make_steward):
        """``mcp.url`` is a real ``str``, not an unconfigured child ``MagicMock``."""
        url = make_steward().mcp.url
        assert isinstance(url, str), f'expected mcp.url to be a real str, got {type(url).__name__!r}'

    @pytest.mark.asyncio
    async def test_briefing_continuation_prompt_is_awaitable(self, make_steward):
        """``briefing.build_steward_continuation_prompt`` is awaitable, not a plain ``MagicMock``.

        Same reassignment hazard as ``test_briefing_initial_prompt_is_awaitable``:
        ``briefing`` is an ``AsyncMock``, but this attribute is *reassigned*, and
        reassigning it with a bare ``MagicMock`` would make the steward's
        ``await`` raise ``TypeError`` at the re-escalation call site.
        """
        steward = make_steward()

        assert await steward.briefing.build_steward_continuation_prompt(steward.task) is not None

    def test_default_worktree_task_dir_is_seeded(self, make_steward):
        """The fixture-owned worktree's ``.task/`` carries parseable metadata + plan files.

        Union-of-defaults with the folded-in ``test_steward.py`` worktree fixture.
        ``steward.py`` reads neither file, so this is inert for existing
        consumers — it exists so a call site that *does* read them (the retired
        fixture's reason for writing them) finds valid JSON rather than a
        missing path.
        """
        import json

        task_dir = make_steward().worktree / '.task'

        for name in ('metadata.json', 'plan.json'):
            path = task_dir / name
            assert path.is_file(), f'expected make_steward to seed {path}'
            json.loads(path.read_text())  # raises if not valid JSON

    def test_caller_supplied_escalation_queue_is_used_verbatim(self, make_steward, tmp_path):
        """``escalation_queue=`` replaces the mock queue and keeps its own directory.

        ``test_steward.py``'s ``steward_with_real_queue`` substitutes a real
        filesystem-backed ``EscalationQueue`` to assert on-disk write locations.
        The fixture must not build a mock queue at all in that case, and must not
        stamp its own derived ``queue_dir`` over the one the caller's queue owns.
        """
        from escalation.queue import EscalationQueue

        own_dir = tmp_path / 'caller-escalations'
        real_queue = EscalationQueue(own_dir)

        steward = make_steward(escalation_queue=real_queue)

        assert steward.escalation_queue is real_queue, (
            'make_steward(escalation_queue=...) must use the caller queue verbatim'
        )
        assert real_queue.queue_dir == own_dir, (
            f'the fixture must not re-stamp queue_dir on a caller-supplied queue — '
            f'expected {own_dir}, got {real_queue.queue_dir}'
        )
