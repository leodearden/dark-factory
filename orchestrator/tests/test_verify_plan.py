"""Tests for orchestrator.verify_plan — derive_verify_plan() + FileKind.

Task γ of the verify-plan PRD (plans/verify-plan-prd.md §Contract·derive_verify_plan).
Unifies the twice-fixed scope decision (scope_module_config + _build_fallback_config)
behind a single pure ``derive_verify_plan()`` and a ``FileKind`` enum, so file
classification happens exactly once instead of being reimplemented per call site.

No source stub exists yet — every test in this module is RED until
orchestrator/src/orchestrator/verify_plan.py is created (step-2 onward).

GOLDEN fixtures below reconstruct the historical incident diffs from the cited
fix commits (all git-verified present on this branch) rather than inventing
arbitrary file lists — see PRD resolved-decision 6.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from orchestrator import verify
from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.verify_cmd import parse_config_command
from orchestrator.verify_plan import (
    FileKind,
    PlannedRun,
    ScopeKind,
    VerifyPlan,
    _is_collectable_test_file,
    _is_conftest,
    _is_test_file,
    classify_file,
    derive_verify_plan,
)

# ---------------------------------------------------------------------------
# GOLDEN incident fixtures
# ---------------------------------------------------------------------------

# task-1077: conftest.py must trigger the full unscoped suite, never be passed
# directly to pytest as a target (pytest >= 9 exits 1 "no tests ran" on a bare
# conftest target). The same fix landed twice — scope_module_config
# (d7504d432d) and _build_fallback_config (cb7277926d) — the exact "same bug
# fixed in both functions" class derive_verify_plan closes by construction.
ROOT_CONFTEST_DIFF: list[str] = ['orchestrator/tests/conftest.py']

# task-1852: a non-test data module under tests/ — a test-tree member but NOT
# pytest-collectable (passing it to pytest produces rc=5 "no tests ran").
# Fixed twice: scope_module_config (4fbed6c4fb, has_test_data -> full suite)
# and _build_fallback_config (7c9b316260, bare-fallback -> SKIPPED/None).
DATA_MODULE_DIFF: list[str] = ['shared/tests/silent_fallthrough_allowlist.py']

# A Protocol-defining source file (D2): file-scoped pyright cannot verify
# cross-file Protocol conformance, so a STRUCTURAL file must widen pyright to
# the unscoped package-wide command in BOTH the module and fallback paths —
# the latent gap _build_fallback_config never closed (only scope_module_config
# widened for this case).
STRUCTURAL_DIFF: list[str] = ['orchestrator/src/orchestrator/interfaces.py']

# An all-INERT diff (no .py/.rs at all) for the plan-level flags: every
# module-path/fallback branch would no-op on this, so derive_verify_plan must
# short-circuit to a TRIVIAL PlannedRun rather than fabricate a pytest run.
_ALL_INERT_DIFF: list[str] = ['docs/README.md', 'scripts/deploy.yaml']

# Canned file contents for the dict-backed fake worktree_reader below. Only
# STRUCTURAL_DIFF's file has real (Protocol-bearing) content; every other
# path — including ROOT_CONFTEST_DIFF/DATA_MODULE_DIFF's files, and any path
# absent from this dict entirely — reads back as None, which classify_file
# must treat as "STRUCTURAL simply not detected", never an error.
_FAKE_FILE_CONTENTS: dict[str, str] = {
    STRUCTURAL_DIFF[0]: 'class Foo(Protocol):\n    def method(self) -> None: ...\n',
}


def fake_worktree_reader(path: str) -> str | None:
    """Dict-backed stand-in for real file I/O (``Callable[[str], str | None]``).

    Keeps derive_verify_plan pure and unit-testable without touching a real
    filesystem: returns the canned Protocol content for STRUCTURAL_DIFF's
    file, else None.
    """
    return _FAKE_FILE_CONTENTS.get(path)


# ---------------------------------------------------------------------------
# FileKind / classify_file (step-1: RED)
# ---------------------------------------------------------------------------


class TestFileKindMembers:
    """FileKind is a plain Enum with exactly the six classification kinds."""

    def test_members_present(self):
        names = {member.name for member in FileKind}
        assert names == {
            'CONFTEST', 'COLLECTABLE_TEST', 'TEST_DATA', 'STRUCTURAL', 'SOURCE', 'INERT',
        }


class TestClassifyFile:
    """classify_file(path, content) -> FileKind runs the classification ladder exactly once.

    Precedence: CONFTEST > COLLECTABLE_TEST > TEST_DATA > STRUCTURAL > SOURCE > INERT.
    """

    # -- one representative path per FileKind ---------------------------------

    def test_conftest_under_subdirectory(self):
        assert classify_file('orchestrator/tests/conftest.py', None) is FileKind.CONFTEST

    def test_conftest_at_root(self):
        assert classify_file('conftest.py', None) is FileKind.CONFTEST

    def test_collectable_test_prefix(self):
        assert classify_file('a/test_x.py', None) is FileKind.COLLECTABLE_TEST

    def test_collectable_test_suffix(self):
        assert classify_file('a/x_test.py', None) is FileKind.COLLECTABLE_TEST

    def test_data_module_under_tests_dir(self):
        """Task-1852 golden: not conftest, not collectable, but a test-tree member."""
        assert classify_file(DATA_MODULE_DIFF[0], None) is FileKind.TEST_DATA

    def test_structural_protocol_source_file(self):
        content = _FAKE_FILE_CONTENTS[STRUCTURAL_DIFF[0]]
        assert classify_file(STRUCTURAL_DIFF[0], content) is FileKind.STRUCTURAL

    def test_structural_typeddict_source_file(self):
        content = 'class Bar(TypedDict):\n    name: str\n'
        assert classify_file('orchestrator/src/orchestrator/types.py', content) is FileKind.STRUCTURAL

    def test_plain_source_file(self):
        content = 'def do_thing(x: int) -> str:\n    return str(x)\n'
        assert classify_file('orchestrator/src/orchestrator/utils.py', content) is FileKind.SOURCE

    def test_non_python_path_is_inert(self):
        assert classify_file('docs/README.md', None) is FileKind.INERT
        assert classify_file('scripts/deploy.yaml', None) is FileKind.INERT
        assert classify_file('crates/foo/src/lib.rs', None) is FileKind.INERT

    # -- precedence assertions -------------------------------------------------

    def test_test_data_beats_structural(self):
        """A data module under tests/ that ALSO defines a Protocol stays TEST_DATA.

        TEST_DATA must outrank STRUCTURAL so a Protocol-defining data module
        under tests/ still full-suites (D1) rather than merely widening pyright.
        """
        content = 'class Foo(Protocol):\n    def method(self) -> None: ...\n'
        assert classify_file(DATA_MODULE_DIFF[0], content) is FileKind.TEST_DATA

    def test_conftest_beats_test_data(self):
        """conftest.py under tests/ classifies CONFTEST, never TEST_DATA."""
        assert classify_file('shared/tests/conftest.py', None) is FileKind.CONFTEST

    def test_none_content_never_raises_and_skips_structural(self):
        """content=None must never raise — STRUCTURAL is simply not detected."""
        result = classify_file('orchestrator/src/orchestrator/foo.py', None)
        assert result is FileKind.SOURCE


# ---------------------------------------------------------------------------
# Derived predicates: _is_conftest / _is_collectable_test_file / _is_test_file
# (step-3: RED)
# ---------------------------------------------------------------------------

# One representative path per FileKind, covering conftest at multiple depths,
# both collectable-test naming conventions, a collectable test nested under
# tests/ (precedence check), a data module under tests/, and plain source.
_PREDICATE_PATH_TABLE: list[str] = [
    'conftest.py',
    'a/conftest.py',
    'a/b/conftest.py',
    'orchestrator/tests/conftest.py',
    'shared/tests/conftest.py',
    'test_foo.py',
    'a/test_foo.py',
    'foo_test.py',
    'a/foo_test.py',
    'a/tests/test_foo.py',
    DATA_MODULE_DIFF[0],
    'tests/data.py',
    'a/tests/helpers.py',
    'orchestrator/src/orchestrator/utils.py',
    'foo.py',
    'a/b/c/plain_module.py',
]


class TestDerivedPredicates:
    """_is_conftest / _is_collectable_test_file / _is_test_file are DERIVED from classify_file.

    Never recombined: each predicate is pinned as an exact FileKind membership
    check, and each must stay behaviorally equivalent to the legacy verify.py
    predicate it replaces.
    """

    # -- behavioral equivalence to the legacy verify.py predicates ------------

    def test_is_conftest_matches_legacy(self):
        for path in _PREDICATE_PATH_TABLE:
            assert _is_conftest(path) == verify._is_conftest(path), path

    def test_is_collectable_test_file_matches_legacy(self):
        for path in _PREDICATE_PATH_TABLE:
            assert _is_collectable_test_file(path) == verify._is_collectable_test_file(path), path

    def test_is_test_file_matches_legacy(self):
        for path in _PREDICATE_PATH_TABLE:
            assert _is_test_file(path) == verify._is_test_file(path), path

    # -- explicit narrow/broad pin against FileKind membership ----------------

    def test_is_conftest_is_exact_classify_file_membership(self):
        for path in _PREDICATE_PATH_TABLE:
            expected = classify_file(path, None) is FileKind.CONFTEST
            assert _is_conftest(path) == expected, path

    def test_is_collectable_test_file_is_narrow_membership(self):
        """NARROW: COLLECTABLE_TEST only."""
        for path in _PREDICATE_PATH_TABLE:
            expected = classify_file(path, None) is FileKind.COLLECTABLE_TEST
            assert _is_collectable_test_file(path) == expected, path

    def test_is_test_file_is_broad_membership(self):
        """BROAD: COLLECTABLE_TEST ∪ TEST_DATA, excludes conftest."""
        for path in _PREDICATE_PATH_TABLE:
            expected = classify_file(path, None) in (FileKind.COLLECTABLE_TEST, FileKind.TEST_DATA)
            assert _is_test_file(path) == expected, path


# ---------------------------------------------------------------------------
# Plan datatypes: ScopeKind / PlannedRun / VerifyPlan (step-5: RED)
# ---------------------------------------------------------------------------


class TestScopeKind:
    """ScopeKind is a StrEnum with exactly the four scope outcomes."""

    def test_members_present(self):
        names = {member.name for member in ScopeKind}
        assert names == {'FULL_SUITE', 'FILE_SCOPED', 'SKIPPED', 'TRIVIAL'}

    def test_serialises_to_lowercase_str_value(self):
        assert str(ScopeKind.FULL_SUITE) == 'full_suite'
        assert str(ScopeKind.FILE_SCOPED) == 'file_scoped'
        assert str(ScopeKind.SKIPPED) == 'skipped'
        assert str(ScopeKind.TRIVIAL) == 'trivial'


class TestPlannedRun:
    """PlannedRun is a frozen dataclass: one (module, tool) slot's outcome."""

    def test_is_frozen(self):
        run = PlannedRun(
            module_prefix='orchestrator', cmd=None, scope_kind=ScopeKind.TRIVIAL, reason='x',
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            run.reason = 'y'  # type: ignore[misc]

    def test_to_dict_with_real_cmd_is_json_native(self):
        """D3: a PlannedRun carrying a real VerifyCmd serialises cmd to a dict."""
        cmd = parse_config_command('pytest -q a/test_x.py')
        run = PlannedRun(
            module_prefix='orchestrator',
            cmd=cmd,
            scope_kind=ScopeKind.FILE_SCOPED,
            reason='file-scoped collectable test',
        )
        d = run.to_dict()
        assert d == {
            'module_prefix': 'orchestrator',
            'cmd': {
                'tool': 'pytest',
                'uv_project': None,
                'cwd_rel': None,
                'base_flags': ['-q'],
                'targets': ['a/test_x.py'],
                'env': {},
                'wrappers': [],
                'raw': None,
            },
            'scope_kind': 'file_scoped',
            'reason': 'file-scoped collectable test',
        }
        json.dumps(d)  # D3: must not raise.

    def test_skipped_run_serialises_cmd_null_with_nonempty_reason(self):
        run = PlannedRun(
            module_prefix='shared',
            cmd=None,
            scope_kind=ScopeKind.SKIPPED,
            reason='task-1852: bare-fallback data module, no real suite to run',
        )
        d = run.to_dict()
        assert d['cmd'] is None
        assert d['scope_kind'] == 'skipped'
        assert d['reason'] == 'task-1852: bare-fallback data module, no real suite to run'
        json.dumps(d)  # D3: must not raise.


class TestVerifyPlan:
    """VerifyPlan is a frozen dataclass: the full set of planned runs plus flags."""

    def test_is_frozen(self):
        plan = VerifyPlan(runs=())
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.needs_pipeline_guard_check = True  # type: ignore[misc]

    def test_default_needs_pipeline_guard_check_is_false(self):
        assert VerifyPlan(runs=()).needs_pipeline_guard_check is False

    def test_to_dict_is_json_native_dict(self):
        cmd = parse_config_command('pytest -q')
        plan = VerifyPlan(
            runs=(
                PlannedRun('orchestrator', cmd, ScopeKind.FULL_SUITE, 'conftest touched'),
                PlannedRun('shared', None, ScopeKind.SKIPPED, 'no files under prefix'),
            ),
            needs_pipeline_guard_check=True,
        )
        d = plan.to_dict()
        assert isinstance(d, dict)
        assert d['needs_pipeline_guard_check'] is True
        assert len(d['runs']) == 2
        assert d['runs'][0]['scope_kind'] == 'full_suite'
        assert d['runs'][1] == {
            'module_prefix': 'shared', 'cmd': None, 'scope_kind': 'skipped',
            'reason': 'no files under prefix',
        }
        # D3: round-trips through the real json module byte-for-byte as data.
        assert json.loads(json.dumps(d)) == d


# ---------------------------------------------------------------------------
# derive_verify_plan: module-config branch (step-7: RED)
# ---------------------------------------------------------------------------


def _run_for(plan: VerifyPlan, prefix: str, tool_word: str) -> PlannedRun | None:
    """Find *prefix*'s PlannedRun whose reason names *tool_word* (e.g. ``'pytest:'``).

    Tool identity is recoverable from ``cmd.tool`` for a non-SKIPPED run, but a
    SKIPPED slot carries ``cmd=None`` (D3's "explicit reasoned skip, never a
    dropped command") — so ``derive_verify_plan`` always prefixes each
    per-tool ``PlannedRun.reason`` with its tool name, keeping the reason the
    tool-identity signal of last resort.
    """
    return next(
        (r for r in plan.runs if r.module_prefix == prefix and r.reason.startswith(tool_word)),
        None,
    )


class TestDeriveVerifyPlanModulePath:
    """derive_verify_plan(existing_files, module_configs, config, worktree_reader).

    Module-config branch: module_configs is non-empty.
    """

    def test_root_conftest_full_suites_pytest(self):
        """GOLDEN task-1077 (d7504d432d): conftest.py -> unscoped full-suite pytest."""
        mc = ModuleConfig(
            prefix='orchestrator',
            test_command=(
                'uv run --project orchestrator --directory orchestrator '
                'pytest tests/ --tb=short -q'
            ),
            lint_command='uv run --directory orchestrator ruff check src/',
        )
        plan = derive_verify_plan(ROOT_CONFTEST_DIFF, [mc], None, fake_worktree_reader)
        run = _run_for(plan, 'orchestrator', 'pytest:')
        assert run is not None
        assert run.scope_kind is ScopeKind.FULL_SUITE
        # Verbatim unscoped test_command — structural equality against the
        # same parse_config_command transform sidesteps render()'s documented
        # cwd_rel-as-leading-`cd` normalisation (not always byte-identical to
        # a --directory-form input — see verify_cmd.render's docstring).
        assert mc.test_command is not None
        assert run.cmd == parse_config_command(mc.test_command)
        assert 'conftest' in run.reason.lower()

    def test_structural_file_full_suites_pyright_and_skips_pytest(self):
        """GOLDEN D2 module-side: a Protocol source file widens pyright, skips pytest."""
        mc = ModuleConfig(
            prefix='orchestrator',
            test_command='uv run --project orchestrator --directory orchestrator pytest tests/',
            lint_command='uv run --directory orchestrator ruff check src/',
            type_check_command=(
                'uv run --project orchestrator --directory orchestrator pyright src/ tests/'
            ),
        )
        plan = derive_verify_plan(STRUCTURAL_DIFF, [mc], None, fake_worktree_reader)

        pyright_run = _run_for(plan, 'orchestrator', 'pyright:')
        assert pyright_run is not None
        assert pyright_run.scope_kind is ScopeKind.FULL_SUITE
        assert mc.type_check_command is not None
        assert pyright_run.cmd == parse_config_command(mc.type_check_command)
        assert STRUCTURAL_DIFF[0] in pyright_run.reason

        pytest_run = _run_for(plan, 'orchestrator', 'pytest:')
        assert pytest_run is not None
        assert pytest_run.scope_kind is ScopeKind.SKIPPED

    def test_lone_collectable_test_file_scopes_pytest(self):
        """Control: a real test file alone must produce a FILE_SCOPED pytest run."""
        mc = ModuleConfig(
            prefix='shared',
            test_command='uv run --directory shared pytest tests/',
            lint_command='uv run --directory shared ruff check src/',
        )
        plan = derive_verify_plan(['shared/tests/test_x.py'], [mc], None, fake_worktree_reader)
        run = _run_for(plan, 'shared', 'pytest:')
        assert run is not None
        assert run.scope_kind is ScopeKind.FILE_SCOPED
        assert run.cmd is not None
        assert 'shared/tests/test_x.py' in run.cmd.targets

    def test_no_matching_files_contributes_only_skipped_runs(self):
        """Control: a module with zero matching files contributes no non-SKIPPED runs."""
        mc = ModuleConfig(
            prefix='shared',
            test_command='uv run --directory shared pytest tests/',
            lint_command='uv run --directory shared ruff check src/',
            type_check_command='uv run --directory shared pyright src/',
        )
        plan = derive_verify_plan(['fused-memory/src/foo.py'], [mc], None, fake_worktree_reader)
        module_runs = [r for r in plan.runs if r.module_prefix == 'shared']
        assert module_runs
        assert all(r.scope_kind is ScopeKind.SKIPPED for r in module_runs)


# ---------------------------------------------------------------------------
# derive_verify_plan: fallback branch (step-9: RED)
# ---------------------------------------------------------------------------


class TestDeriveVerifyPlanFallbackPath:
    """derive_verify_plan(existing_files, [], config, worktree_reader).

    Fallback branch: module_configs is empty. A synthetic ``'__fallback__'``
    module (mirrors ``_build_fallback_config``'s ``prefix='__fallback__'``
    sentinel) is derived straight from *config*'s global commands, unifying
    the SAME D1/D2 rules as the module-config branch with one reconciliation:
    CONFTEST/TEST_DATA only widen to FULL_SUITE when a real suite exists
    (module path's ``mc.test_command``, or here a non-default configured
    ``config.test_command``); the bare-``'pytest'``-default + TEST_DATA-only
    case has no real suite to run, so it degrades to an explicit reasoned
    SKIPPED rather than fabricating a run that would rc=5 (task-1852 golden).
    CONFTEST always full-suites regardless — a directory target is always
    safe to run, even against the bare ``pytest`` default.
    """

    def test_data_module_bare_pytest_default_skips_with_reason(self):
        """GOLDEN task-1852 (7c9b316260): bare-fallback data module -> explicit SKIPPED.

        config.test_command == 'pytest' (the bare default) means there is no
        real suite to fall back to — the twice-fixed bug's second fix made
        _build_fallback_config yield test_cmd=None here rather than risk
        rc=5 ("no tests ran"). derive_verify_plan upgrades the silent None to
        an explicit reasoned SKIPPED PlannedRun — explicitly NOT silent.
        """
        config = OrchestratorConfig(project_root=Path('/fake'), test_command='pytest')
        plan = derive_verify_plan(DATA_MODULE_DIFF, [], config, fake_worktree_reader)
        run = _run_for(plan, '__fallback__', 'pytest:')
        assert run is not None
        assert run.scope_kind is ScopeKind.SKIPPED
        assert run.reason
        assert DATA_MODULE_DIFF[0] in run.reason
        assert '1852' in run.reason

    def test_data_module_with_configured_suite_full_suites(self):
        """Reconciliation pin: a non-default configured suite -> FULL_SUITE, not SKIPPED.

        Distinct from the bare-default case above: when a real suite exists,
        using it is strictly safer than skipping (mirrors
        _build_fallback_config's
        test_data_module_with_nonconfigured_pytest_uses_full_suite).
        """
        config = OrchestratorConfig(
            project_root=Path('/fake'), test_command="uv run --extra dev pytest -m 'not slow'",
        )
        plan = derive_verify_plan(DATA_MODULE_DIFF, [], config, fake_worktree_reader)
        run = _run_for(plan, '__fallback__', 'pytest:')
        assert run is not None
        assert run.scope_kind is ScopeKind.FULL_SUITE
        assert run.cmd == parse_config_command(config.test_command)

    def test_structural_file_full_suites_pyright(self):
        """GOLDEN D2 fallback gap: _build_fallback_config never widens; derive_verify_plan does.

        A Protocol source file must widen pyright to the unscoped command in
        the fallback path too, matching the module-path behavior — pytest
        for the same source-only diff stays SKIPPED (nothing collectable).
        """
        config = OrchestratorConfig(project_root=Path('/fake'), test_command='pytest')
        plan = derive_verify_plan(STRUCTURAL_DIFF, [], config, fake_worktree_reader)

        pyright_run = _run_for(plan, '__fallback__', 'pyright:')
        assert pyright_run is not None
        assert pyright_run.scope_kind is ScopeKind.FULL_SUITE
        assert pyright_run.cmd == parse_config_command(config.type_check_command)
        assert STRUCTURAL_DIFF[0] in pyright_run.reason

        pytest_run = _run_for(plan, '__fallback__', 'pytest:')
        assert pytest_run is not None
        assert pytest_run.scope_kind is ScopeKind.SKIPPED

    def test_fallback_conftest_targets_parent_directory(self):
        """GOLDEN fallback conftest (reconstructs cb7277926d).

        'a/conftest.py' -> pytest targets parent directory 'a', never the
        conftest file itself (pytest >= 9 exits 1 "no tests ran" on a bare
        conftest target) — full suite, runnable command.
        """
        config = OrchestratorConfig(project_root=Path('/fake'), test_command='pytest')
        plan = derive_verify_plan(['a/conftest.py'], [], config, fake_worktree_reader)
        run = _run_for(plan, '__fallback__', 'pytest:')
        assert run is not None
        assert run.scope_kind is ScopeKind.FULL_SUITE
        assert run.cmd is not None
        assert 'a' in run.cmd.targets
        assert not any('conftest.py' in t for t in run.cmd.targets)

    def test_root_conftest_maps_to_dot(self):
        """A root-level conftest.py must target '.', never itself (root -> '.')."""
        config = OrchestratorConfig(project_root=Path('/fake'), test_command='pytest')
        plan = derive_verify_plan(['conftest.py'], [], config, fake_worktree_reader)
        run = _run_for(plan, '__fallback__', 'pytest:')
        assert run is not None
        assert run.scope_kind is ScopeKind.FULL_SUITE
        assert run.cmd is not None
        assert run.cmd.targets == ('.',)


# ---------------------------------------------------------------------------
# Plan-level flags: needs_pipeline_guard_check / TRIVIAL (step-11: RED)
# ---------------------------------------------------------------------------


class TestPlanFlags:
    """VerifyPlan's plan-level flags: needs_pipeline_guard_check and TRIVIAL."""

    # -- (a) needs_pipeline_guard_check ---------------------------------------

    def test_needs_pipeline_guard_check_true_for_merge_role(self):
        """An all-INERT diff in a merge-role context flags the caller to check the guard.

        derive_verify_plan never executes _verify_pipeline_guard_requires_full_gate
        itself (that's an impure subprocess call) — it only records that the
        caller must run it before trusting this plan's trivial verdict.
        """
        plan = derive_verify_plan(_ALL_INERT_DIFF, [], None, fake_worktree_reader, role='merge')
        assert plan.needs_pipeline_guard_check is True

    def test_needs_pipeline_guard_check_false_for_task_role(self):
        """The SAME all-INERT diff outside merge role never needs the guard check."""
        plan = derive_verify_plan(_ALL_INERT_DIFF, [], None, fake_worktree_reader, role='task')
        assert plan.needs_pipeline_guard_check is False

    def test_needs_pipeline_guard_check_defaults_false(self):
        """role defaults to 'task' — omitting it must not accidentally opt into the guard."""
        plan = derive_verify_plan(_ALL_INERT_DIFF, [], None, fake_worktree_reader)
        assert plan.needs_pipeline_guard_check is False

    # -- (b) TRIVIAL -------------------------------------------------------------

    def test_all_inert_diff_emits_trivial_run_not_fabricated_pytest(self):
        """An all-INERT diff must not fabricate a pytest run — it marks TRIVIAL."""
        plan = derive_verify_plan(_ALL_INERT_DIFF, [], None, fake_worktree_reader)
        assert len(plan.runs) == 1
        assert plan.runs[0].scope_kind is ScopeKind.TRIVIAL
        assert plan.runs[0].cmd is None
        assert plan.runs[0].reason

    def test_all_inert_diff_with_module_configs_still_trivial(self):
        """The TRIVIAL short-circuit applies before ever branching on module_configs."""
        mc = ModuleConfig(
            prefix='orchestrator',
            test_command='uv run --directory orchestrator pytest tests/',
            lint_command='uv run --directory orchestrator ruff check src/',
        )
        plan = derive_verify_plan(_ALL_INERT_DIFF, [mc], None, fake_worktree_reader)
        assert len(plan.runs) == 1
        assert plan.runs[0].scope_kind is ScopeKind.TRIVIAL

    # -- (c) D1-with-suite pin (module path) --------------------------------------

    def test_module_path_test_data_full_suites_distinct_from_fallback_skip(self):
        """A TEST_DATA file in the MODULE path always has mc.test_command as a real
        suite, so it FULL_SUITEs — never SKIPPED, unlike the bare-fallback golden
        (TestDeriveVerifyPlanFallbackPath.test_data_module_bare_pytest_default_skips_with_reason).
        Locks the step-8 module-path reconciliation so it can't regress silently.
        """
        mc = ModuleConfig(
            prefix='shared',
            test_command='uv run --directory shared pytest tests/',
            lint_command='uv run --directory shared ruff check src/',
        )
        plan = derive_verify_plan(DATA_MODULE_DIFF, [mc], None, fake_worktree_reader)
        run = _run_for(plan, 'shared', 'pytest:')
        assert run is not None
        assert run.scope_kind is ScopeKind.FULL_SUITE
        assert mc.test_command is not None
        assert run.cmd == parse_config_command(mc.test_command)
