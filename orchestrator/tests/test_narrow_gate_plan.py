"""What the narrow merge gate runs: narrow_gate_plan's command set (task 5411)."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest
from _serial_recovery_helpers import ORCH_SHAPED_PYPROJECT

from orchestrator.config import ModuleConfig
from orchestrator.narrow_gate_class import GateMiss, MergeGateResolution, MissReason
from orchestrator.narrow_gate_plan import NarrowGateDecision, narrow_gate_plan
from orchestrator.verify_artifact_consumers import BEHAVIOURAL_ARTIFACTS
from orchestrator.verify_cmd import parse_config_command
from orchestrator.verify_plan import (
    PlannedRun,
    ScopeKind,
    VerifyPlan,
    planned_run_leg,
    scope_prefix_to_keyword,
    whole_tree_guard_family,
)

ORCH = ModuleConfig(
    prefix='orchestrator',
    test_command='cd orchestrator && uv run pytest',
    lint_command='cd orchestrator && uv run ruff check',
    type_check_command='cd orchestrator && uv run pyright',
)
SHARED = ModuleConfig(
    prefix='shared',
    test_command='cd shared && uv run pytest',
    lint_command='cd shared && uv run ruff check',
    type_check_command='cd shared && uv run pyright',
)
TESTS_SCRIPTS = ModuleConfig(
    prefix='tests/scripts',
    test_command='uv run --project shared pytest tests/scripts/',
    lint_command='uv run --project shared ruff check tests/scripts/',
    type_check_command='uv run --project shared pyright tests/scripts/',
)

# Seams stand in for verify.py's worktree lister and reader. Never build a lister by
# globbing a directory: the whole-tree census guard would read this module as a scanner.
_GUARD_SOURCE = (
    'import pytest\n'
    'from _orch_helpers import WHOLE_TREE_SCAN_TEST_TIMEOUT\n'
    'pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)\n'
)
_DESELECTED_GUARD_SOURCE = (
    'import pytest\n'
    'from _orch_helpers import WHOLE_TREE_SCAN_TEST_TIMEOUT\n'
    'pytestmark = [pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT), pytest.mark.warm_lane_bash]\n'
)
GUARD_A = 'orchestrator/tests/test_guard_a.py'
GUARD_B = 'orchestrator/tests/test_guard_b.py'
PLAIN = 'orchestrator/tests/test_plain.py'
SHARED_GUARD = 'shared/tests/test_shared_guard.py'
SCRIPTS_PLAIN = 'tests/scripts/test_plain_script.py'

_CONTENTS = {
    GUARD_A: _GUARD_SOURCE,
    GUARD_B: _GUARD_SOURCE,
    PLAIN: 'def test_plain():\n    pass\n',
    SHARED_GUARD: _GUARD_SOURCE,
    SCRIPTS_PLAIN: 'def test_plain():\n    pass\n',
}
_TESTS_BY_PREFIX = {
    'orchestrator': [GUARD_A, GUARD_B, PLAIN],
    'shared': [SHARED_GUARD],
    'tests/scripts': [SCRIPTS_PLAIN],
}


def _reader(extra: dict[str, str] | None = None) -> Callable[[str], str | None]:
    contents = {**_CONTENTS, **(extra or {})}
    return contents.get


def _list_pkg_tests(prefix: str) -> list[str]:
    return list(_TESTS_BY_PREFIX.get(prefix, []))


def _plan(
    resolution: MergeGateResolution,
    paths: Sequence[str],
    modules: Sequence[ModuleConfig],
    reader: Callable[[str], str | None] | None = None,
) -> VerifyPlan | None:
    return narrow_gate_plan(
        resolution, paths, modules, list_pkg_tests=_list_pkg_tests, worktree_reader=reader or _reader(),
    )


def _run(plan: VerifyPlan | None, prefix: str, tool: str) -> PlannedRun:
    assert plan is not None
    [run] = [r for r in plan.runs if r.module_prefix == prefix and r.reason.startswith(f'{tool}:')]
    return run


def _consumers_of(artifact: str) -> tuple[str, ...]:
    [entry] = [e for e in BEHAVIOURAL_ARTIFACTS if artifact in e.globs]
    return entry.test_files


class TestDocsGate:

    def test_each_module_runs_its_guard_family_file_scoped(self) -> None:
        plan = _plan(MergeGateResolution.DOCS, ['README.md', 'docs/a.md'], [ORCH, SHARED])

        orch = _run(plan, 'orchestrator', 'pytest')
        assert orch.scope_kind is ScopeKind.FILE_SCOPED
        assert orch.scoped_targets == (GUARD_A, GUARD_B)
        assert orch.cmd == scope_prefix_to_keyword('cd orchestrator && uv run pytest', 'pytest', [GUARD_A, GUARD_B])
        assert orch.reason.startswith('pytest: docs_gate:')
        shared = _run(plan, 'shared', 'pytest')
        assert (shared.scope_kind, shared.scoped_targets) == (ScopeKind.FILE_SCOPED, (SHARED_GUARD,))

    @pytest.mark.parametrize('tool', ['lint', 'pyright'])
    def test_lint_and_pyright_are_skipped_with_a_docs_gate_reason(self, tool: str) -> None:
        plan = _plan(MergeGateResolution.DOCS, ['README.md'], [ORCH, SHARED])

        for prefix in ('orchestrator', 'shared'):
            run = _run(plan, prefix, tool)
            assert (run.scope_kind, run.cmd) == (ScopeKind.SKIPPED, None)
            assert run.reason.startswith(f'{tool}: docs_gate:')

    def test_a_touched_behavioural_artifact_adds_its_consumer_tests(self) -> None:
        artifact = 'skills/unblock-auto/SKILL.md'

        run = _run(_plan(MergeGateResolution.DOCS, [artifact], [ORCH]), 'orchestrator', 'pytest')

        expected = tuple(sorted({GUARD_A, GUARD_B, *_consumers_of(artifact)}))
        assert run.scope_kind is ScopeKind.FILE_SCOPED
        assert run.scoped_targets == expected
        assert run.cmd == scope_prefix_to_keyword('cd orchestrator && uv run pytest', 'pytest', list(expected))
        assert artifact in run.reason

    def test_a_full_suites_demand_runs_that_modules_full_suite(self) -> None:
        plan = _plan(MergeGateResolution.DOCS, ['dark-factory-orchestrator.yaml'], [ORCH, TESTS_SCRIPTS])

        scripts = _run(plan, 'tests/scripts', 'pytest')
        assert scripts.scope_kind is ScopeKind.FULL_SUITE
        assert scripts.cmd == parse_config_command('uv run --project shared pytest tests/scripts/')
        assert scripts.scoped_targets == ()
        assert _run(plan, 'orchestrator', 'pytest').scoped_targets == (GUARD_A, GUARD_B)

    def test_a_module_with_no_family_and_no_demand_skips_pytest(self) -> None:
        plan = _plan(MergeGateResolution.DOCS, ['README.md'], [ORCH, TESTS_SCRIPTS])

        run = _run(plan, 'tests/scripts', 'pytest')
        assert (run.scope_kind, run.cmd, run.scoped_targets) == (ScopeKind.SKIPPED, None, ())
        assert run.reason.startswith('pytest: docs_gate:')

    def test_a_narrowed_selection_wholly_deselected_by_the_modules_marker_is_skipped(self) -> None:
        reader = _reader({
            'orchestrator/pyproject.toml': ORCH_SHAPED_PYPROJECT,
            GUARD_A: _DESELECTED_GUARD_SOURCE,
            GUARD_B: _DESELECTED_GUARD_SOURCE,
        })

        run = _run(_plan(MergeGateResolution.DOCS, ['README.md'], [ORCH, SHARED], reader), 'orchestrator', 'pytest')

        assert (run.scope_kind, run.cmd, run.scoped_targets) == (ScopeKind.SKIPPED, None, ())
        assert "'not warm_lane_bash'" in run.reason


class TestCommentsGate:

    _PY = ['orchestrator/tests/test_b.py', 'orchestrator/src/orchestrator/a.py']

    def test_a_module_owning_a_changed_py_scopes_lint_and_pyright_to_it(self) -> None:
        plan = _plan(MergeGateResolution.COMMENTS, ['README.md', *self._PY], [ORCH, SHARED])

        expected = sorted(self._PY)
        lint = _run(plan, 'orchestrator', 'lint')
        assert (lint.scope_kind, lint.scoped_targets) == (ScopeKind.FILE_SCOPED, tuple(expected))
        assert lint.cmd == scope_prefix_to_keyword('cd orchestrator && uv run ruff check', 'ruff check', expected)
        assert lint.reason.startswith('lint: comments_gate:')
        pyright = _run(plan, 'orchestrator', 'pyright')
        assert (pyright.scope_kind, pyright.scoped_targets) == (ScopeKind.FILE_SCOPED, tuple(expected))
        assert pyright.cmd == scope_prefix_to_keyword('cd orchestrator && uv run pyright', 'pyright', expected)

    def test_the_pytest_slot_is_the_docs_gates(self) -> None:
        plan = _plan(MergeGateResolution.COMMENTS, self._PY, [ORCH])

        run = _run(plan, 'orchestrator', 'pytest')
        assert (run.scope_kind, run.scoped_targets) == (ScopeKind.FILE_SCOPED, (GUARD_A, GUARD_B))
        assert run.reason.startswith('pytest: comments_gate:')

    @pytest.mark.parametrize('tool', ['lint', 'pyright'])
    def test_a_module_owning_no_changed_py_keeps_lint_and_pyright_skipped(self, tool: str) -> None:
        run = _run(_plan(MergeGateResolution.COMMENTS, self._PY, [ORCH, SHARED]), 'shared', tool)

        assert (run.scope_kind, run.cmd) == (ScopeKind.SKIPPED, None)
        assert run.reason.startswith(f'{tool}: comments_gate:')

    def test_a_py_owned_by_no_module_has_no_narrow_plan(self) -> None:
        assert _plan(MergeGateResolution.COMMENTS, ['setup.py', *self._PY], [ORCH, SHARED]) is None


class TestPlanShape:

    def test_an_empty_command_set_is_never_narrow(self) -> None:
        assert _plan(MergeGateResolution.DOCS, ['README.md'], [TESTS_SCRIPTS]) is None
        assert _plan(MergeGateResolution.DOCS, ['README.md'], []) is None

    def test_the_full_gate_has_no_narrow_plan_to_build(self) -> None:
        with pytest.raises(ValueError, match='full_gate'):
            _plan(MergeGateResolution.FULL, ['README.md'], [ORCH])

    @pytest.mark.parametrize('resolution', [MergeGateResolution.DOCS, MergeGateResolution.COMMENTS])
    def test_every_run_keeps_its_leg_and_the_scoped_targets_invariant(self, resolution: MergeGateResolution) -> None:
        plan = _plan(resolution, ['README.md', 'orchestrator/src/orchestrator/a.py'], [ORCH, SHARED, TESTS_SCRIPTS])

        assert plan is not None
        assert len(plan.runs) == 9
        for run in plan.runs:
            assert planned_run_leg(run) is not None, run.reason
            assert (run.scope_kind is ScopeKind.FILE_SCOPED) == bool(run.scoped_targets), run


class TestWholeTreeGuardFamily:

    def test_it_is_public_and_sorted(self) -> None:
        assert whole_tree_guard_family('orchestrator', _list_pkg_tests, _reader()) == (GUARD_A, GUARD_B)

    def test_no_lister_enumerates_nothing(self) -> None:
        assert whole_tree_guard_family('orchestrator', None, _reader()) == ()


class TestNarrowGateDecision:

    _MISS = GateMiss(MissReason.NOT_ALLOWLISTED, 'pyproject.toml')

    def test_full_carries_its_miss_and_no_plan(self) -> None:
        decision = NarrowGateDecision.full(self._MISS, ['pyproject.toml'])

        assert (decision.resolution, decision.miss, decision.plan) == (MergeGateResolution.FULL, self._MISS, None)
        assert decision.changed_paths == ('pyproject.toml',)

    def test_a_narrow_decision_carries_its_plan(self) -> None:
        plan = _plan(MergeGateResolution.DOCS, ['README.md'], [ORCH])
        assert plan is not None

        decision = NarrowGateDecision(MergeGateResolution.DOCS, None, plan, ('README.md',))

        assert decision.plan is plan

    @pytest.mark.parametrize(('resolution', 'miss', 'with_plan'), [
        (MergeGateResolution.DOCS, None, False),
        (MergeGateResolution.COMMENTS, _MISS, True),
        (MergeGateResolution.FULL, None, False),
        (MergeGateResolution.FULL, _MISS, True),
    ])
    def test_an_inconsistent_decision_is_refused(
        self, resolution: MergeGateResolution, miss: GateMiss | None, with_plan: bool,
    ) -> None:
        plan = _plan(MergeGateResolution.DOCS, ['README.md'], [ORCH]) if with_plan else None

        with pytest.raises(ValueError, match=resolution.value):
            NarrowGateDecision(resolution, miss, plan)
