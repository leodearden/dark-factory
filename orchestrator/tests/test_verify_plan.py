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
from _verify_config_corpus import (
    FM_LINT_COMMAND,
    MODULE_LINT_COMMANDS,
    ROOT_LINT_COMMAND,
    ROOT_TEST_COMMAND,
    ROOT_TYPE_CHECK_COMMAND,
)

from orchestrator import verify
from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.verify_cmd import parse_config_command, render
from orchestrator.verify_plan import (
    FileKind,
    PlannedRun,
    ScopeKind,
    VerifyPlan,
    _is_collectable_test_file,
    _is_conftest,
    _is_test_file,
    _scope_prefix_to_keyword,
    classify_file,
    derive_verify_plan,
    reverse_dependent_test_targets,
)

# The real config command strings this suite exercises live in
# `_verify_config_corpus.py` (one definition site, shared with test_verify_cmd.py
# and test_verify_scope_kappa.py); `test_verify_config_corpus.py` checks them
# against the live YAML — the executable form of the "verified byte-identical"
# claim this comment block used to make in prose.


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

# task λ (2589): a plain SOURCE .py file under a module prefix — not a
# conftest, not COLLECTABLE_TEST, not TEST_DATA, and (with no
# type_check_command configured) never read for STRUCTURAL content either.
# Pre-λ this hit _derive_module_runs' pytest else-branch and always produced
# a SKIPPED "no collectable test files touched" regardless of role — the
# task-role pytest floor (R3) makes role='task' run the owning module's full
# test_command instead. Synthetic path (not a historical-incident golden),
# matching the existing invented-path convention for control-shaped tests
# (e.g. 'shared/tests/test_x.py', 'fused-memory/src/foo.py' below).
SOURCE_ONLY_DIFF: list[str] = ['orchestrator/src/orchestrator/some_module.py']

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
            scoped_targets=('a/test_x.py',),
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
            'scoped_targets': ['a/test_x.py'],
        }
        json.dumps(d)  # D3: must not raise.

    def test_scoped_targets_defaults_to_empty_tuple(self):
        """The field is defaulted, so every existing 4-arg construction stays valid.

        Empty is the CORRECT value for a SKIPPED slot — nothing ran, so
        nothing was narrowed (task 3219's "non-empty iff FILE_SCOPED").
        """
        run = PlannedRun('m', None, ScopeKind.SKIPPED, 'x')
        assert run.scoped_targets == ()

    def test_to_dict_records_scoped_targets_as_json_native_list(self):
        """D3: the tuple serialises as a plain list, per _verify_cmd_to_dict's convention."""
        run = PlannedRun(
            module_prefix='orchestrator',
            cmd=parse_config_command('pytest -q'),
            scope_kind=ScopeKind.FILE_SCOPED,
            reason='pytest: file-scoped to touched test file(s)',
            scoped_targets=('a/test_x.py', 'a/test_y.py'),
        )
        d = run.to_dict()
        assert d['scoped_targets'] == ['a/test_x.py', 'a/test_y.py']
        assert isinstance(d['scoped_targets'], list)
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
            'reason': 'no files under prefix', 'scoped_targets': [],
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

    def test_structural_file_full_suites_pyright_and_pytest_at_task_role(self):
        """GOLDEN D2 module-side, migrated by the task-role pytest floor (λ, task 2589
        R3): a Protocol source file widens pyright (D2, role-independent) and — at the
        default role='task' — now also full-suites pytest via the floor, instead of the
        pre-λ SKIPPED. See test_structural_file_full_suites_pyright_and_skips_pytest_at_merge_role
        below for the preserved legacy SKIPPED shape at role='merge' (R4)."""
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
        assert pytest_run.scope_kind is ScopeKind.FULL_SUITE
        assert mc.test_command is not None
        assert pytest_run.cmd == parse_config_command(mc.test_command)

    def test_structural_file_full_suites_pyright_and_skips_pytest_at_merge_role(self):
        """R4 rollback golden: the SAME structural diff at role='merge' preserves the
        pre-λ legacy shape — pytest stays SKIPPED. The task-role floor (R3) never
        widens the merge gate; that widening is the separate, knob-gated
        merge_verify_breadth='full' path."""
        mc = ModuleConfig(
            prefix='orchestrator',
            test_command='uv run --project orchestrator --directory orchestrator pytest tests/',
            lint_command='uv run --directory orchestrator ruff check src/',
            type_check_command=(
                'uv run --project orchestrator --directory orchestrator pyright src/ tests/'
            ),
        )
        config = OrchestratorConfig(project_root=Path('/fake'), merge_verify_breadth='scoped')
        plan = derive_verify_plan(
            STRUCTURAL_DIFF, [mc], config, fake_worktree_reader, role='merge',
        )

        pyright_run = _run_for(plan, 'orchestrator', 'pyright:')
        assert pyright_run is not None
        assert pyright_run.scope_kind is ScopeKind.FULL_SUITE
        assert mc.type_check_command is not None
        assert pyright_run.cmd == parse_config_command(mc.type_check_command)

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
# derive_verify_plan: task-role pytest floor (λ, task 2589 step-3: RED)
# ---------------------------------------------------------------------------


class TestDeriveVerifyPlanTaskRoleFloor:
    """λ (task 2589), R3: the task-role pytest floor.

    Pre-λ, _derive_module_runs' pytest else-branch always emitted a SKIPPED
    "no collectable test files touched" for a source-only diff, regardless
    of role — zero pytest signal at task verify for the single most common
    diff shape. role='task' now runs the owning module's full test_command
    instead; role='merge' (and the fallback branch, out of scope here) keep
    the legacy SKIPPED shape (R4 — pinned by the merge+scoped counterparts
    added alongside the migrated goldens in TestDeriveVerifyPlanModulePath).
    """

    def test_source_only_diff_full_suites_pytest_at_task_role(self):
        """(a) A source-only diff full-suites the owning module's pytest at
        role='task', and the reason names both the role and the "sibling
        modules NOT run" coverage signpost."""
        mc = ModuleConfig(
            prefix='orchestrator',
            test_command='uv run --directory orchestrator pytest tests/',
            lint_command='uv run --directory orchestrator ruff check src/',
        )
        plan = derive_verify_plan(
            SOURCE_ONLY_DIFF, [mc], None, fake_worktree_reader, role='task',
        )
        run = _run_for(plan, 'orchestrator', 'pytest:')
        assert run is not None
        assert run.scope_kind is ScopeKind.FULL_SUITE
        assert mc.test_command is not None
        assert run.cmd == parse_config_command(mc.test_command)
        assert 'task' in run.reason.lower()
        assert 'not run' in run.reason.lower()

    def test_touched_test_only_diff_stays_file_scoped_at_task_role(self):
        """(b) A real collectable test file keeps FILE_SCOPED selection — the
        floor only fires on the pytest else-branch (no touched test file),
        never overriding the existing collectable-test selection."""
        mc = ModuleConfig(
            prefix='shared',
            test_command='uv run --directory shared pytest tests/',
            lint_command='uv run --directory shared ruff check src/',
        )
        plan = derive_verify_plan(
            ['shared/tests/test_x.py'], [mc], None, fake_worktree_reader, role='task',
        )
        run = _run_for(plan, 'shared', 'pytest:')
        assert run is not None
        assert run.scope_kind is ScopeKind.FILE_SCOPED
        assert run.cmd is not None
        assert 'shared/tests/test_x.py' in run.cmd.targets

    def test_multi_module_source_only_diff_floors_only_owning_modules(self):
        """(c) Each owning module full-suites its own pytest; a THIRD
        registered module NOT touched by the diff contributes only SKIPPED
        runs — the floor never widens beyond the modules actually touched
        (R1)."""
        mc_a = ModuleConfig(
            prefix='orchestrator',
            test_command='uv run --directory orchestrator pytest tests/',
            lint_command='uv run --directory orchestrator ruff check src/',
        )
        mc_b = ModuleConfig(
            prefix='shared',
            test_command='uv run --directory shared pytest tests/',
            lint_command='uv run --directory shared ruff check src/',
        )
        mc_c = ModuleConfig(
            prefix='escalation',
            test_command='uv run --directory escalation pytest tests/',
            lint_command='uv run --directory escalation ruff check src/',
        )
        files = [
            SOURCE_ONLY_DIFF[0],
            'shared/src/shared/another_module.py',
        ]
        plan = derive_verify_plan(
            files, [mc_a, mc_b, mc_c], None, fake_worktree_reader, role='task',
        )

        run_a = _run_for(plan, 'orchestrator', 'pytest:')
        assert run_a is not None
        assert run_a.scope_kind is ScopeKind.FULL_SUITE

        run_b = _run_for(plan, 'shared', 'pytest:')
        assert run_b is not None
        assert run_b.scope_kind is ScopeKind.FULL_SUITE

        module_c_runs = [r for r in plan.runs if r.module_prefix == 'escalation']
        assert module_c_runs
        assert all(r.scope_kind is ScopeKind.SKIPPED for r in module_c_runs)

    def test_structural_only_diff_floors_pytest_and_widens_pyright_at_task_role(self):
        """(d) A structural-only diff full-suites BOTH pytest (the floor —
        STRUCTURAL counts as source, non-test .py) AND pyright (existing D2,
        unaffected by the floor)."""
        mc = ModuleConfig(
            prefix='orchestrator',
            test_command='uv run --directory orchestrator pytest tests/',
            lint_command='uv run --directory orchestrator ruff check src/',
            type_check_command=(
                'uv run --project orchestrator --directory orchestrator pyright src/ tests/'
            ),
        )
        plan = derive_verify_plan(
            STRUCTURAL_DIFF, [mc], None, fake_worktree_reader, role='task',
        )

        pytest_run = _run_for(plan, 'orchestrator', 'pytest:')
        assert pytest_run is not None
        assert pytest_run.scope_kind is ScopeKind.FULL_SUITE
        assert mc.test_command is not None
        assert pytest_run.cmd == parse_config_command(mc.test_command)

        pyright_run = _run_for(plan, 'orchestrator', 'pyright:')
        assert pyright_run is not None
        assert pyright_run.scope_kind is ScopeKind.FULL_SUITE
        assert mc.type_check_command is not None
        assert pyright_run.cmd == parse_config_command(mc.type_check_command)


# ---------------------------------------------------------------------------
# derive_verify_plan: merge role + merge_verify_breadth fork (λ, task 2589 step-5: RED)
# ---------------------------------------------------------------------------


class TestDeriveVerifyPlanMergeBreadth:
    """λ (task 2589), R1/R2/R4: the broad merge gate's breadth knob.

    role='merge' + config.merge_verify_breadth='full' full-suites EVERY PASSED
    module's EVERY configured command (pytest+ruff+pyright) — even a module the
    diff never touches — closing the "only the touched modules are protected"
    gap the task-role floor deliberately does not close (R1: the floor never
    widens beyond owning modules; only the merge+full gate does). breadth=
    'scoped' (the shipped default) keeps role='merge' byte-identical to the
    legacy _derive_module_runs shape (R4 — the gate's rollback path).
    """

    # -- (a) role='merge' + breadth='full' ------------------------------------

    def test_merge_full_breadth_full_suites_every_passed_module(self):
        """(a) EVERY passed module full-suites each configured command, including
        a module the diff never touches at all; a module missing a command gets
        an explicit SKIPPED for that tool, never a fabricated run."""
        mc_a = ModuleConfig(
            prefix='orchestrator',
            test_command='uv run --directory orchestrator pytest tests/',
            lint_command='uv run --directory orchestrator ruff check src/',
            type_check_command='uv run --directory orchestrator pyright src/',
        )
        mc_b = ModuleConfig(
            prefix='shared',
            test_command='uv run --directory shared pytest tests/',
            lint_command='uv run --directory shared ruff check src/',
            # No type_check_command configured -> pyright must SKIP, never fabricate.
        )
        config = OrchestratorConfig(project_root=Path('/fake'), merge_verify_breadth='full')

        plan = derive_verify_plan(
            SOURCE_ONLY_DIFF, [mc_a, mc_b], config, fake_worktree_reader, role='merge',
        )

        # mc_a: touched by the diff -- still FULL_SUITE (never file-scoped) at
        # this breadth, for all three tools.
        pytest_a = _run_for(plan, 'orchestrator', 'pytest:')
        assert pytest_a is not None
        assert pytest_a.scope_kind is ScopeKind.FULL_SUITE
        assert mc_a.test_command is not None
        assert pytest_a.cmd == parse_config_command(mc_a.test_command)
        assert 'merge' in pytest_a.reason.lower()
        assert 'full' in pytest_a.reason.lower() or 'registered' in pytest_a.reason.lower()

        lint_a = _run_for(plan, 'orchestrator', 'lint:')
        assert lint_a is not None
        assert lint_a.scope_kind is ScopeKind.FULL_SUITE
        assert mc_a.lint_command is not None
        assert lint_a.cmd == parse_config_command(mc_a.lint_command)
        assert 'merge' in lint_a.reason.lower()

        pyright_a = _run_for(plan, 'orchestrator', 'pyright:')
        assert pyright_a is not None
        assert pyright_a.scope_kind is ScopeKind.FULL_SUITE
        assert mc_a.type_check_command is not None
        assert pyright_a.cmd == parse_config_command(mc_a.type_check_command)
        assert 'merge' in pyright_a.reason.lower()

        # mc_b: NOT touched by the diff at all -- still FULL_SUITE (R1: the
        # broad merge gate covers every REGISTERED module passed to it, not
        # just the modules the diff happens to touch).
        pytest_b = _run_for(plan, 'shared', 'pytest:')
        assert pytest_b is not None
        assert pytest_b.scope_kind is ScopeKind.FULL_SUITE
        assert mc_b.test_command is not None
        assert pytest_b.cmd == parse_config_command(mc_b.test_command)

        lint_b = _run_for(plan, 'shared', 'lint:')
        assert lint_b is not None
        assert lint_b.scope_kind is ScopeKind.FULL_SUITE
        assert mc_b.lint_command is not None
        assert lint_b.cmd == parse_config_command(mc_b.lint_command)

        # mc_b has no type_check_command configured -> explicit SKIPPED, never
        # a fabricated pyright run.
        pyright_b = _run_for(plan, 'shared', 'pyright:')
        assert pyright_b is not None
        assert pyright_b.scope_kind is ScopeKind.SKIPPED
        assert pyright_b.cmd is None

    # -- (b) role='merge' + breadth='scoped' (R4 rollback golden) ------------

    def test_merge_scoped_breadth_source_only_matches_legacy_skipped(self):
        """(b) source-only diff, breadth='scoped' (config=None) -> pytest SKIPPED,
        byte-identical to the pre-λ legacy _derive_module_runs shape."""
        mc = ModuleConfig(
            prefix='orchestrator',
            test_command='uv run --directory orchestrator pytest tests/',
            lint_command='uv run --directory orchestrator ruff check src/',
        )
        plan = derive_verify_plan(
            SOURCE_ONLY_DIFF, [mc], None, fake_worktree_reader, role='merge',
        )
        run = _run_for(plan, 'orchestrator', 'pytest:')
        assert run is not None
        assert run.scope_kind is ScopeKind.SKIPPED
        assert run.cmd is None
        assert run.reason == 'pytest: no collectable test files touched — nothing to run'

    def test_merge_scoped_breadth_collectable_test_matches_legacy_file_scoped(self):
        """(b) a real touched test file, breadth='scoped' -> pytest FILE_SCOPED,
        byte-identical to the pre-λ legacy shape."""
        mc = ModuleConfig(
            prefix='shared',
            test_command='uv run --directory shared pytest tests/',
            lint_command='uv run --directory shared ruff check src/',
        )
        plan = derive_verify_plan(
            ['shared/tests/test_x.py'], [mc], None, fake_worktree_reader, role='merge',
        )
        run = _run_for(plan, 'shared', 'pytest:')
        assert run is not None
        assert run.scope_kind is ScopeKind.FILE_SCOPED
        assert run.cmd is not None
        assert 'shared/tests/test_x.py' in run.cmd.targets

    def test_merge_scoped_breadth_structural_matches_legacy_pyright_widen_pytest_skip(self):
        """(b) structural-only diff, breadth='scoped' -> pyright FULL_SUITE (D2,
        role/breadth-independent) AND pytest SKIPPED (the floor never fires for
        role='merge'), byte-identical to the pre-λ legacy shape."""
        mc = ModuleConfig(
            prefix='orchestrator',
            test_command='uv run --project orchestrator --directory orchestrator pytest tests/',
            lint_command='uv run --directory orchestrator ruff check src/',
            type_check_command=(
                'uv run --project orchestrator --directory orchestrator pyright src/ tests/'
            ),
        )
        plan = derive_verify_plan(
            STRUCTURAL_DIFF, [mc], None, fake_worktree_reader, role='merge',
        )

        pyright_run = _run_for(plan, 'orchestrator', 'pyright:')
        assert pyright_run is not None
        assert pyright_run.scope_kind is ScopeKind.FULL_SUITE
        assert mc.type_check_command is not None
        assert pyright_run.cmd == parse_config_command(mc.type_check_command)

        pytest_run = _run_for(plan, 'orchestrator', 'pytest:')
        assert pytest_run is not None
        assert pytest_run.scope_kind is ScopeKind.SKIPPED

    def test_merge_scoped_breadth_conftest_matches_legacy_full_suite(self):
        """(b) conftest diff, breadth='scoped' -> pytest FULL_SUITE (D1,
        role/breadth-independent), byte-identical to the pre-λ legacy shape."""
        mc = ModuleConfig(
            prefix='orchestrator',
            test_command=(
                'uv run --project orchestrator --directory orchestrator '
                'pytest tests/ --tb=short -q'
            ),
            lint_command='uv run --directory orchestrator ruff check src/',
        )
        plan = derive_verify_plan(
            ROOT_CONFTEST_DIFF, [mc], None, fake_worktree_reader, role='merge',
        )
        run = _run_for(plan, 'orchestrator', 'pytest:')
        assert run is not None
        assert run.scope_kind is ScopeKind.FULL_SUITE
        assert mc.test_command is not None
        assert run.cmd == parse_config_command(mc.test_command)
        assert 'conftest' in run.reason.lower()

    # -- (c) TRIVIAL short-circuit is breadth-independent (R2) ----------------

    def test_docs_only_diff_stays_trivial_at_merge_full_breadth(self):
        """(c) An all-INERT diff never fabricates FULL_SUITE runs even under
        role='merge' + breadth='full' — the TRIVIAL short-circuit in
        derive_verify_plan runs before the module-config branch is ever
        reached, so it is unconditionally breadth-independent."""
        mc = ModuleConfig(
            prefix='orchestrator',
            test_command='uv run --directory orchestrator pytest tests/',
        )
        config = OrchestratorConfig(project_root=Path('/fake'), merge_verify_breadth='full')
        plan = derive_verify_plan(
            _ALL_INERT_DIFF, [mc], config, fake_worktree_reader, role='merge',
        )
        assert len(plan.runs) == 1
        assert plan.runs[0].scope_kind is ScopeKind.TRIVIAL
        assert plan.needs_pipeline_guard_check is True


# ---------------------------------------------------------------------------
# The two scopers: &&-chain trailing-clause preservation (task 3061)
# ---------------------------------------------------------------------------

# Every real config command, paired with the keyword its slot scopes on. Used
# as the lockstep corpus below; `_scope_to_keyword` and `_scope_prefix_to_keyword`
# must agree on ALL of them, for EVERY keyword — including the mismatched
# pairings, which exercise the keyword-absent and keyword-in-a-later-segment
# rejections.
_REAL_CONFIG_COMMANDS: list[tuple[str, str]] = [
    *((f'{module}-lint', cmd) for module, cmd in MODULE_LINT_COMMANDS.items()),
    ('fm-lint', FM_LINT_COMMAND),
    ('root-lint', ROOT_LINT_COMMAND),
    ('root-type-check', ROOT_TYPE_CHECK_COMMAND),
    ('root-test', ROOT_TEST_COMMAND),
    ('ruff-trailing-value-flag', 'ruff check src/ --select E'),
    ('opaque-mypy', 'mypy src/'),
    ('no-op-true', 'true'),
]

# Synthetic hazards, not real configs: an `&&` nested inside a shell construct
# that `_NON_AND_CHAIN_TOKENS`' token-equality check cannot see. Carried through
# the same lockstep sweep as the real commands, and pinned individually by
# `test_shell_construct_is_never_split_mid_construct` below.
_SHELL_CONSTRUCT_HAZARDS: list[tuple[str, str]] = [
    ('subst-dollar-paren', 'ruff check $(git ls-files && echo x) && python3 y.py'),
    ('subst-backtick', 'ruff check `ls && echo x` && python3 y.py'),
    ('unspaced-subshell', '(ruff check src/ && echo x) && python3 y.py'),
    ('subst-in-double-quotes', 'ruff check "$(ls && echo x)" && python3 y.py'),
]

_LOCKSTEP_KEYWORDS = ('ruff check', 'pyright', 'pytest')

_LOCKSTEP_CASES = [
    (raw, keyword, f'{name}::{keyword.replace(" ", "-")}')
    for name, raw in (*_REAL_CONFIG_COMMANDS, *_SHELL_CONSTRUCT_HAZARDS)
    for keyword in _LOCKSTEP_KEYWORDS
]


class TestScoperTrailingClausePreservation:
    """A sibling-checker clause chained after the scoped tool must SURVIVE.

    Task 3061: every subproject's ``lint_command`` chains a
    ``python3 fused-memory/scripts/check_*.py <dir>`` gate after ``ruff
    check``. Both scopers truncated at the keyword, so scoped pre-merge
    verify silently ran only ruff — the bare-MagicMock and asyncmock-
    assertion gates were invisible until post-merge (task 2920's violation
    landed exactly this way).

    A cwd-sequenced same-tool fan-out must KEEP being truncated, though —
    see ``test_root_type_check_fan_out_still_truncates`` for why that is a
    correctness requirement, not merely a scoping preference.
    """

    _FILES = ['fused-memory/tests/test_harness.py']

    def test_fused_memory_lint_chain_scopes_ruff_and_keeps_both_checkers(self):
        """The task's headline case, as a full literal golden.

        The ruff clause is narrowed to the one touched file (its unscoped
        ``src/ tests/`` targets are gone); both sibling checkers survive
        byte-identically, still pointed at the whole ``fused-memory/tests``
        directory they assert an invariant over.
        """
        scoped = verify._scope_to_keyword(FM_LINT_COMMAND, 'ruff check', self._FILES)
        assert scoped == (
            'uv run --project fused-memory ruff check fused-memory/tests/test_harness.py'
            ' && python3 fused-memory/scripts/check_bare_magicmock_config.py fused-memory/tests'
            ' && python3 fused-memory/scripts/check_asyncmock_assertion_style.py fused-memory/tests'
        )
        assert 'src/ tests/' not in scoped
        for checker in (
            '&& python3 fused-memory/scripts/check_bare_magicmock_config.py fused-memory/tests',
            '&& python3 fused-memory/scripts/check_asyncmock_assertion_style.py fused-memory/tests',
        ):
            assert checker in scoped, f'{checker!r} must survive verbatim'
            assert checker in FM_LINT_COMMAND, 'the slice asserted above must be verbatim'

    def test_root_lint_chain_scopes_ruff_and_keeps_the_checker(self):
        """dark-factory-orchestrator.yaml:50 — the fallback path's own command."""
        scoped = verify._scope_to_keyword(ROOT_LINT_COMMAND, 'ruff check', self._FILES)
        assert scoped == (
            'uv run ruff check fused-memory/tests/test_harness.py'
            ' && python3 fused-memory/scripts/check_bare_magicmock_config.py shared/tests'
            ' escalation/tests fused-memory/tests orchestrator/tests dashboard/tests'
        )

    def test_trailing_value_flag_still_truncates_at_the_keyword(self):
        """The flag-misread guard is untouched: intra-segment truncation still applies.

        ``--select E``'s value would be misread as an extra ruff target by
        ``scope_to`` if the whole segment were parsed, so truncating at the
        keyword WITHIN the matched segment stays exactly as it was.
        """
        scoped = verify._scope_to_keyword('ruff check src/ --select E', 'ruff check', ['a.py'])
        assert scoped == 'ruff check a.py'
        assert '--select' not in scoped

    @pytest.mark.parametrize(
        'raw', ['mypy src/', 'true'], ids=['opaque-mypy', 'no-op-true'],
    )
    def test_keyword_absent_returns_byte_identical(self, raw):
        assert verify._scope_to_keyword(raw, 'ruff check', self._FILES) == raw

    def test_root_type_check_fan_out_still_truncates(self):
        """HAZARD GUARD: a cwd-sequenced same-tool fan-out must NOT keep its tail.

        ``dark-factory-orchestrator.yaml:51`` is ``cd fused-memory && npx
        pyright && cd ../orchestrator && npx pyright && cd ../dashboard &&
        npx pyright``. Preserving that tail would (a) run pyright fully
        UNSCOPED over two more subprojects, defeating scoping entirely, and
        (b) break correctness — scoping applies ``strip_cwd``, which removes
        the leading ``cd fused-memory``, so a surviving ``cd ../orchestrator``
        would resolve relative to the worktree ROOT and escape the repo.
        """
        scoped = verify._scope_to_keyword(ROOT_TYPE_CHECK_COMMAND, 'pyright', self._FILES)
        assert scoped == 'npx pyright fused-memory/tests/test_harness.py'
        assert 'cd ../orchestrator' not in scoped
        assert 'cd ../dashboard' not in scoped

    def test_root_test_fan_out_still_truncates(self):
        """Same hazard for ``dark-factory-orchestrator.yaml:41``'s 8-segment fan-out."""
        scoped = verify._scope_to_keyword(ROOT_TEST_COMMAND, 'pytest', self._FILES)
        assert scoped == 'uv run pytest fused-memory/tests/test_harness.py'
        assert 'cd ../escalation' not in scoped
        assert 'cockpit' not in scoped

    @pytest.mark.parametrize(
        'raw',
        [raw for _, raw in _SHELL_CONSTRUCT_HAZARDS],
        ids=[name for name, _ in _SHELL_CONSTRUCT_HAZARDS],
    )
    def test_shell_construct_is_never_split_mid_construct(self, raw):
        """HAZARD GUARD: an `&&` inside `$(...)`/backticks/`(...)` is not a chain op.

        The tail gate's token check is EQUALITY based, so it sees `(` only
        when ``shlex`` isolates it as its own whitespace-separated token. A
        command substitution or an unspaced subshell hides its `&&` from that
        check while ``split_top_level_and`` — which tracks quote state only —
        still splits there. Carrying a tail out of one
        truncates the head mid-construct and emits an unbalanced shell string
        (a stray `)`, an unpaired backtick), which bash rejects outright: a
        spurious RED verify, strictly worse than the missed sibling checker
        this feature exists to fix, and a NEW failure mode rather than a
        pre-existing one. The character-level grouping check must reject
        these, restoring the byte-identical pre-feature output.
        """
        scoped = verify._scope_to_keyword(raw, 'ruff check', self._FILES)
        assert scoped is not None  # `raw` is a str, so the None passthrough is unreachable
        expected = (
            raw
            if raw.startswith('(')  # keyword not in segment 0 -> untouched
            else f'ruff check {self._FILES[0]}'
        )
        assert scoped == expected
        assert '&& python3 y.py' not in scoped or scoped == raw
        # No dangling opener/closer survived into the emitted command.
        assert scoped.count('(') == scoped.count(')')
        assert scoped.count('`') % 2 == 0

    @pytest.mark.parametrize(
        ('raw', 'keyword'),
        [(raw, keyword) for raw, keyword, _ in _LOCKSTEP_CASES],
        ids=[test_id for _, _, test_id in _LOCKSTEP_CASES],
    )
    def test_lockstep_between_the_two_scopers(self, raw, keyword):
        """The string scoper and the VerifyCmd scoper must never disagree.

        They were historically kept in sync only by a docstring mandate —
        a convention this very bug shows was load-bearing and unenforced.
        Both now route through the shared ``split_chain_tail`` gate, so this
        asserts a property that holds by construction.
        """
        assert verify._scope_to_keyword(raw, keyword, self._FILES) == render(
            _scope_prefix_to_keyword(raw, keyword, self._FILES)
        )


# ---------------------------------------------------------------------------
# verify._reproject_str: &&-chain tail awareness (task 3061)
# ---------------------------------------------------------------------------


class TestReprojectStrChainTail:
    """_reproject_str must reproject a chained command's head, not give up on it.

    The fallback path (verify.py:2314-2321) scopes a config command with
    ``_scope_to_keyword`` and then reprojects it into ``_FALLBACK_UV_PROJECT``,
    because — per task 2036 — the depless workspace-root uv project cannot
    spawn ruff/pyright. Losing the ``--project`` injection is therefore a hard
    breakage (exit 127, command not found), not a cosmetic diff.

    Once the scoper starts preserving a trailing ``&&`` clause, the string
    handed to ``_reproject_str`` is a chain; re-parsing the whole chain
    classifies it OPAQUE and the injection is SILENTLY dropped. This class
    guards that regression BEFORE the scopers change, so no intermediate
    commit is red.

    It ALSO guards the converse hazard (task 3061 step-7). ``_reproject_str``
    is the only one of the three ``split_chain_tail`` call sites that
    re-renders a parsed head WITHOUT ``strip_cwd``, and ``render()`` re-emits
    a parsed ``cwd_rel`` as a LEADING ``cd X &&``. ``split_chain_tail``'s gate
    rejects a literal ``cd`` TOKEN precisely because a cwd shift voids every
    later segment's relative paths — but it inspects the INPUT string, where
    that shift is still spelled ``--directory X`` and is therefore invisible
    to it. So a tail may only be carried when the head has NO cwd to re-emit;
    otherwise ``_reproject_str`` must bail to the untouched original.

    That is not theoretical: all seven module ``lint_command``s are exactly
    the ``--directory <mod>`` + ``&&``-chained
    ``python3 fused-memory/scripts/check_bare_magicmock_config.py <mod>/tests``
    shape, so an introduced ``cd <mod> &&`` makes the tail resolve as
    ``<mod>/fused-memory/scripts/...`` -> exit 2 -> a spurious RED verify.
    """

    def test_chained_head_is_reprojected_and_tail_survives_verbatim(self):
        """The regression case: bare ``uv run`` head + sibling-checker tail."""
        raw = (
            'uv run ruff check f.py'
            ' && python3 fused-memory/scripts/check_bare_magicmock_config.py shared/tests'
        )
        assert verify._reproject_str(raw, verify._FALLBACK_UV_PROJECT) == (
            'uv run --project shared ruff check f.py'
            ' && python3 fused-memory/scripts/check_bare_magicmock_config.py shared/tests'
        )

    @pytest.mark.parametrize(
        ('raw', 'expected'),
        [
            ('uv run ruff check f.py', 'uv run --project shared ruff check f.py'),
            ('uv run pyright f.py', 'uv run --project shared pyright f.py'),
            (
                'uv run --project fused-memory ruff check f.py',
                'uv run --project fused-memory ruff check f.py',
            ),
            ('mypy src/', 'mypy src/'),
            ('true', 'true'),
        ],
        ids=[
            'bare-uv-run-ruff',
            'bare-uv-run-pyright',
            'explicit-project-is-a-no-op',
            'opaque-is-a-no-op',
            'no-op-command',
        ],
    )
    def test_single_clause_commands_are_byte_identical_goldens(self, raw, expected):
        """No-tail regression corpus: literal goldens, deliberately not derived."""
        assert verify._reproject_str(raw, verify._FALLBACK_UV_PROJECT) == expected

    @pytest.mark.parametrize(
        'raw',
        [ROOT_TYPE_CHECK_COMMAND, ROOT_TEST_COMMAND],
        ids=['root-type-check-cd-fan-out', 'root-test-subshell-fan-out'],
    )
    def test_gate_reject_cases_still_no_op(self, raw):
        """A cwd-sequenced / subshell fan-out is returned untouched, never truncated."""
        assert verify._reproject_str(raw, verify._FALLBACK_UV_PROJECT) == raw

    def test_none_is_passed_through(self):
        assert verify._reproject_str(None, verify._FALLBACK_UV_PROJECT) is None

    # -- cwd-shifting head + preserved tail must bail (task 3061 step-7) -----

    def test_directory_head_with_tail_bails_byte_identically(self):
        """A ``--directory`` head + tail is returned UNCHANGED, not renormalised.

        ``render()`` would turn the parsed ``cwd_rel`` back into a leading
        ``cd sub &&``, shifting the shell's cwd out from under a tail that was
        written against the worktree root.

        The bail forfeits nothing: ``reproject`` is a documented no-op when
        ``cwd_rel is not None`` (verify_cmd.py "an explicit ``--directory`` is
        already set"), so the discarded expression was a pure re-render with
        no ``--project`` injection to lose.
        """
        raw = 'uv run --directory sub pyright src/ && python3 tools/check.py d'
        assert verify._reproject_str(raw, verify._FALLBACK_UV_PROJECT) == raw

    def test_real_module_lint_command_does_not_double_its_own_path(self):
        """The realistic shape: every module ``lint_command`` is this chain.

        A leading ``cd fused-memory &&`` makes the tail's
        ``fused-memory/scripts/check_*.py`` resolve from INSIDE that directory
        — i.e. ``fused-memory/fused-memory/scripts/...`` -> exit 2 "can't open
        file" -> a spurious RED verify on a clean tree.
        """
        result = verify._reproject_str(FM_LINT_COMMAND, verify._FALLBACK_UV_PROJECT)
        assert result == FM_LINT_COMMAND
        assert 'fused-memory/fused-memory' not in (result or '')

    @pytest.mark.parametrize(
        'raw',
        [
            FM_LINT_COMMAND,
            *MODULE_LINT_COMMANDS.values(),
            ROOT_LINT_COMMAND,
            ROOT_TYPE_CHECK_COMMAND,
            ROOT_TEST_COMMAND,
            'uv run --directory sub pyright src/ && python3 tools/check.py d',
        ],
        ids=[
            'fused-memory-lint',
            *(f'{module}-lint' for module in MODULE_LINT_COMMANDS),
            'root-lint',
            'root-type-check-cd-fan-out',
            'root-test-subshell-fan-out',
            'synthetic-directory-head',
        ],
    )
    def test_never_introduces_a_leading_cd_into_a_surviving_chain(self, raw):
        """INVARIANT: a preserved tail never gains a cwd shift it lacked.

        Stated as "never INTRODUCES", because the two root fan-outs are
        gate-rejected and returned byte-identically — they legitimately keep
        the leading ``cd`` they arrived with. What must never happen is the
        rewrite ADDING one under a tail.
        """
        result = verify._reproject_str(raw, verify._FALLBACK_UV_PROJECT)
        assert result is not None
        if '&&' in result:
            assert result.startswith('cd ') == raw.startswith('cd '), (
                'a surviving tail must not gain a leading `cd` it did not arrive with'
            )

    def test_no_tail_directory_renormalisation_is_unchanged(self):
        """PRE-EXISTING and deliberately out of scope: lone ``--directory`` head.

        ``--directory sub`` -> ``cd sub &&`` is semantically equivalent for a
        head with nothing chained after it, and predates task 3061. Only the
        TAIL case is being fixed, so this golden must not move.
        """
        assert verify._reproject_str(
            'uv run --directory sub pyright src/', verify._FALLBACK_UV_PROJECT,
        ) == 'cd sub && uv run pyright src/'


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


# ---------------------------------------------------------------------------
# reverse_dependent_test_targets (task 2607, step-1/step-3: RED)
# ---------------------------------------------------------------------------
#
# Closes the merge-verify blind spot where a scoped orchestrator-only diff
# never runs escalation's coupled cross-package tests (RED-main fix-forward
# 3x: 1736->1761, 2173->2038, 2435->2604). reverse_dependent_test_targets is
# the pure decision layer: given the touched files, a hardcoded
# depended-upon -> dependents map, and injected (list_pkg_tests, read_content)
# callables, it returns which dependent packages' test files actually import
# a triggering package — no filesystem, no subprocess (established
# test_verify_plan.py idiom: hand-built dict/lambda fakes).

# escalation/tests/test_server.py's real shape: a guarded, indented import of
# orchestrator.merge_queue (escalation/tests/test_server.py:31-34).
_ESCALATION_TEST_SERVER_CONTENT = (
    'from __future__ import annotations\n'
    '\n'
    'import pytest\n'
    '\n'
    'try:\n'
    '    from orchestrator.merge_queue import SpeculativeMergeWorker\n'
    'except ImportError:\n'
    '    SpeculativeMergeWorker = None\n'
)

_ESCALATION_TEST_UNRELATED_CONTENT = (
    'from __future__ import annotations\n'
    '\n'
    'def test_unrelated():\n'
    '    assert True\n'
)


def _fake_list_pkg_tests(tests_by_pkg):
    """Dict-backed stand-in for a real ``list_pkg_tests(pkg) -> list[str]`` callable."""
    return lambda pkg: tests_by_pkg.get(pkg, [])


def _fake_read_content(content_by_path):
    """Dict-backed stand-in for a real ``read_content(path) -> str | None`` callable."""
    return lambda path: content_by_path.get(path)


class TestReverseDependentTestTargets:
    """reverse_dependent_test_targets(existing_files, reverse_dep_map, list_pkg_tests, read_content).

    -> list[tuple[str, list[str]]]: for each dependent package triggered by a
    depended-upon package's SOURCE change, the subset of that dependent's
    test files whose content actually imports the changed package.
    """

    def test_orchestrator_source_change_triggers_escalation_importing_test_only(self):
        """Happy path: only the importing test file is kept; dependents sorted."""
        existing_files = ['orchestrator/src/orchestrator/merge_queue.py']
        reverse_dep_map = {'orchestrator': frozenset({'escalation'})}
        list_pkg_tests = _fake_list_pkg_tests({
            'escalation': [
                'escalation/tests/test_server.py',
                'escalation/tests/test_unrelated.py',
            ],
        })
        read_content = _fake_read_content({
            'escalation/tests/test_server.py': _ESCALATION_TEST_SERVER_CONTENT,
            'escalation/tests/test_unrelated.py': _ESCALATION_TEST_UNRELATED_CONTENT,
        })

        result = reverse_dependent_test_targets(
            existing_files, reverse_dep_map, list_pkg_tests, read_content,
        )

        assert result == [('escalation', ['escalation/tests/test_server.py'])]

    # -- edge cases (step-3) --------------------------------------------------

    def test_non_source_orchestrator_files_do_not_trigger(self):
        """(a) trigger gate: only <pkg>/src/**.py triggers — tests/yaml/non-.py do not."""
        existing_files = [
            'orchestrator/tests/test_x.py',
            'orchestrator/orchestrator.yaml',
            'orchestrator/src/orchestrator/x.txt',
        ]
        reverse_dep_map = {'orchestrator': frozenset({'escalation'})}
        list_pkg_tests = _fake_list_pkg_tests({
            'escalation': ['escalation/tests/test_server.py'],
        })
        read_content = _fake_read_content({
            'escalation/tests/test_server.py': _ESCALATION_TEST_SERVER_CONTENT,
        })

        result = reverse_dependent_test_targets(
            existing_files, reverse_dep_map, list_pkg_tests, read_content,
        )

        assert result == []

    def test_triggered_dependent_with_no_importing_tests_is_omitted(self):
        """(b) a triggered dependent whose tests don't import the package is omitted."""
        existing_files = ['orchestrator/src/orchestrator/merge_queue.py']
        reverse_dep_map = {'orchestrator': frozenset({'escalation'})}
        list_pkg_tests = _fake_list_pkg_tests({
            'escalation': ['escalation/tests/test_unrelated.py'],
        })
        read_content = _fake_read_content({
            'escalation/tests/test_unrelated.py': _ESCALATION_TEST_UNRELATED_CONTENT,
        })

        result = reverse_dependent_test_targets(
            existing_files, reverse_dep_map, list_pkg_tests, read_content,
        )

        assert result == []

    def test_package_with_no_map_entry_yields_empty(self):
        """(c) a changed package absent from reverse_dep_map triggers nothing."""
        existing_files = ['dashboard/src/dashboard/app.py']
        reverse_dep_map = {'orchestrator': frozenset({'escalation'})}
        list_pkg_tests = _fake_list_pkg_tests({
            'escalation': ['escalation/tests/test_server.py'],
        })
        read_content = _fake_read_content({
            'escalation/tests/test_server.py': _ESCALATION_TEST_SERVER_CONTENT,
        })

        result = reverse_dependent_test_targets(
            existing_files, reverse_dep_map, list_pkg_tests, read_content,
        )

        assert result == []

    def test_multiple_triggering_files_do_not_duplicate_dependent(self):
        """(d) multiple triggering source files under one package -> one dependent entry."""
        existing_files = [
            'orchestrator/src/orchestrator/merge_queue.py',
            'orchestrator/src/orchestrator/git_ops.py',
        ]
        reverse_dep_map = {'orchestrator': frozenset({'escalation'})}
        list_pkg_tests = _fake_list_pkg_tests({
            'escalation': ['escalation/tests/test_server.py'],
        })
        read_content = _fake_read_content({
            'escalation/tests/test_server.py': _ESCALATION_TEST_SERVER_CONTENT,
        })

        result = reverse_dependent_test_targets(
            existing_files, reverse_dep_map, list_pkg_tests, read_content,
        )

        assert result == [('escalation', ['escalation/tests/test_server.py'])]


# ---------------------------------------------------------------------------
# PlannedRun.scoped_targets: the D3 plan record's scoping provenance (task 3219)
# ---------------------------------------------------------------------------

# The single-clause counterpart of MODULE_LINT_COMMANDS' chained shape — the
# UNIFORMITY control for test_unchained_lint_records_the_same_scoped_targets.
# Nothing in the live configs looks like this today (every subproject chains a
# sibling checker), which is exactly why the chained path's record loss went
# unnoticed: the pre-3219 fixtures were all single-clause.
_UNCHAINED_LINT_COMMAND = 'uv run --project fused-memory --directory fused-memory ruff check src/ tests/'

_FM_TEST_FILE = 'fused-memory/tests/test_harness.py'

# --- the invariant sweep's corpus (step-5c) --------------------------------
#
# Every case is re-homed under the ONE 'orchestrator/' prefix so a single set
# of diffs matches every ModuleConfig — the sweep varies the COMMAND SHAPE
# (which decides whether cmd ends up raw-retained, structurally scoped, or
# unscoped) against the FILE SHAPE (which decides scope_kind), and asserts the
# scoped_targets invariant survives every combination of the two.

_SWEEP_TEST_FILE = 'orchestrator/tests/test_sweep.py'

_SWEEP_MODULE_CONFIGS: list[tuple[str, ModuleConfig]] = [
    # The live 2-segment chained lint + the root's cd-chained pyright/pytest.
    ('real-chained', ModuleConfig(
        prefix='orchestrator',
        lint_command=MODULE_LINT_COMMANDS['orchestrator'],
        type_check_command=ROOT_TYPE_CHECK_COMMAND,
        test_command=ROOT_TEST_COMMAND,
    )),
    # The only 3-segment chain in the live configs (two sibling checkers).
    ('fm-three-segment-lint', ModuleConfig(
        prefix='orchestrator',
        lint_command=FM_LINT_COMMAND,
        type_check_command='npx pyright',
        test_command='uv run pytest tests/ --timeout=300',
    )),
    # The root lint chain, plus bare unchained tool commands — the structured
    # (cmd.targets-populated) control the pre-3219 fixtures were all built from.
    ('root-lint-bare-tools', ModuleConfig(
        prefix='orchestrator',
        lint_command=ROOT_LINT_COMMAND,
        type_check_command='pyright',
        test_command='pytest',
    )),
    # Commands the scoper cannot narrow at all: OPAQUE (mypy) and a keyword
    # that never appears (`true`). _scope_prefix_to_keyword returns the
    # command UNSCOPED here while _derive_module_runs still labels the slot
    # FILE_SCOPED — a pre-existing property of scope_kind that scoped_targets
    # must track consistently rather than contradict.
    ('unscopable', ModuleConfig(
        prefix='orchestrator',
        lint_command='true',
        type_check_command='mypy src/',
        test_command='pytest',
    )),
    # Partially-configured: the SKIPPED "no X configured" slots must stay empty.
    ('lint-only', ModuleConfig(prefix='orchestrator', lint_command=_UNCHAINED_LINT_COMMAND)),
]

_SWEEP_DIFFS: list[tuple[str, list[str]]] = [
    ('collectable-test', [_SWEEP_TEST_FILE]),
    ('conftest', ROOT_CONFTEST_DIFF),          # D1 -> FULL_SUITE pytest
    ('test-data', ['orchestrator/tests/fixtures_data.py']),
    ('source-only', SOURCE_ONLY_DIFF),         # task-role pytest floor (R3)
    ('structural', STRUCTURAL_DIFF),           # D2 -> FULL_SUITE pyright
    ('mixed', [_SWEEP_TEST_FILE, SOURCE_ONLY_DIFF[0]]),
    ('all-inert', _ALL_INERT_DIFF),            # -> TRIVIAL short-circuit
]

_FULL_BREADTH_CONFIG = OrchestratorConfig(
    project_root=Path('/fake'), merge_verify_breadth='full',
)
_BARE_FALLBACK_CONFIG = OrchestratorConfig(project_root=Path('/fake'), test_command='pytest')
_REAL_SUITE_FALLBACK_CONFIG = OrchestratorConfig(
    project_root=Path('/fake'),
    lint_command=ROOT_LINT_COMMAND,
    type_check_command=ROOT_TYPE_CHECK_COMMAND,
    test_command=ROOT_TEST_COMMAND,
)

# The branches derive_verify_plan can take, split by whether a ModuleConfig is
# consulted at all — the two sweeps below parametrise over different axes, so
# keeping one combined list would multiply the fallback cases by an inert
# module-config dimension and advertise coverage that does not exist.
#
# (name, config, role) — the module path at both roles, plus the knob-gated
# full-breadth merge gate (_derive_full_suite_runs, which must never file-scope).
_SWEEP_MODULE_BRANCHES: list[tuple[str, OrchestratorConfig | None, str]] = [
    ('module/task', None, 'task'),
    ('module/merge', None, 'merge'),
    ('module/merge-full-breadth', _FULL_BREADTH_CONFIG, 'merge'),
]

# (name, config, role) — the fallback path at both the bare-pytest default and
# a real configured suite. module_configs is [] here by definition.
_SWEEP_FALLBACK_BRANCHES: list[tuple[str, OrchestratorConfig, str]] = [
    ('fallback/bare-default', _BARE_FALLBACK_CONFIG, 'task'),
    ('fallback/real-suite', _REAL_SUITE_FALLBACK_CONFIG, 'merge'),
]


def _assert_scoped_targets_invariant(plan, files: list[str], label: str) -> None:
    """Shared body of the two invariant sweeps — three properties per plan.

    1. **The invariant** — ``scoped_targets`` is non-empty EXACTLY when
       ``scope_kind is FILE_SCOPED``. FULL_SUITE (D1/D2 widening, and the
       whole ``merge_verify_breadth='full'`` gate) is deliberately unscoped;
       SKIPPED/TRIVIAL never ran.
    2. **No invented paths** — the record is always a subset of the diff it
       was derived from, so it can never drift into naming a file the plan
       never saw.
    3. **D3 stays intact** — the key serialises JSON-natively and the whole
       plan dict survives a JSON round-trip unchanged, which is what actually
       reaches ``VerifyResult.plan``.
    """
    for run in plan.runs:
        assert bool(run.scoped_targets) == (run.scope_kind is ScopeKind.FILE_SCOPED), (
            f'{label}: {run.reason!r} is '
            f'{run.scope_kind} but scoped_targets={run.scoped_targets!r}'
        )
        assert set(run.scoped_targets) <= set(files), label
        assert run.to_dict()['scoped_targets'] == list(run.scoped_targets), label

    as_dict = plan.to_dict()
    assert json.loads(json.dumps(as_dict)) == as_dict, label


class TestPlanRecordScopedTargets:
    """PlannedRun.scoped_targets records WHICH files a FILE_SCOPED slot narrowed to.

    Task 3219. ``cmd.targets`` cannot answer that question once ``cmd`` is
    raw-retained, which is the shape the tail-preserving chained-lint accept
    path (_scope_prefix_to_keyword) produces for EVERY subproject — each
    one's lint_command chains a sibling checker. The scoping itself was and
    remains correct; only the machine-readable record was lost.
    """

    def test_chained_lint_records_scoped_targets(self):
        """HEADLINE: the chained shape is preserved verbatim AND now records its targets.

        The first three assertions pin that this fix changes NOTHING about
        what executes — the raw-retained tail-preserving shape, P3, and the
        task-3061 sibling-checker preservation are all untouched. The last
        two are the actual deliverable: the D3 record can now answer "which
        files was this scoped to?".
        """
        mc = ModuleConfig(prefix='fused-memory', lint_command=FM_LINT_COMMAND)
        plan = derive_verify_plan([_FM_TEST_FILE], [mc], None, fake_worktree_reader)
        run = _run_for(plan, 'fused-memory', 'lint:')
        assert run is not None
        assert run.scope_kind is ScopeKind.FILE_SCOPED

        # Execution shape UNCHANGED: raw-retained, structured targets empty (P3).
        assert run.cmd is not None
        assert run.cmd.raw is not None
        assert run.cmd.targets == ()
        # task-3061 regression guard: the sibling checker survives unscoped.
        assert 'check_bare_magicmock_config.py fused-memory/tests' in run.cmd.raw
        # ...and the narrowed list is still in the rendered string, as before.
        assert _FM_TEST_FILE in run.cmd.raw

        # The record that was previously lost.
        assert run.scoped_targets == (_FM_TEST_FILE,)
        assert run.to_dict()['scoped_targets'] == [_FM_TEST_FILE]

    def test_unchained_lint_records_the_same_scoped_targets(self):
        """UNIFORMITY: the record reads identically whether or not the config chained.

        Populating only the lossy path would leave consumers with a
        conditional contract ("read cmd.targets, unless cmd.raw is set") —
        the same implicit coupling that produced this bug. Here cmd.targets
        IS populated, and the two must agree: redundancy as a consistency
        check, not a second source of truth.
        """
        mc = ModuleConfig(prefix='fused-memory', lint_command=_UNCHAINED_LINT_COMMAND)
        plan = derive_verify_plan([_FM_TEST_FILE], [mc], None, fake_worktree_reader)
        run = _run_for(plan, 'fused-memory', 'lint:')
        assert run is not None
        assert run.scope_kind is ScopeKind.FILE_SCOPED
        assert run.cmd is not None
        assert run.cmd.raw is None  # structured path — the control
        assert run.cmd.targets  # non-empty
        assert run.scoped_targets == tuple(run.cmd.targets)

    def test_file_scoped_pyright_and_pytest_slots_record_scoped_targets(self):
        """The other two FILE_SCOPED sites in _derive_module_runs.

        pyright records the full touched-.py list (``scoped``); pytest
        records only the collectable tests — the two legitimately DIFFER, so
        one shared field per run, not one per plan, is the right shape.
        """
        source_file = 'orchestrator/src/orchestrator/some_module.py'
        test_file = 'orchestrator/tests/test_x.py'
        mc = ModuleConfig(
            prefix='orchestrator',
            type_check_command=MODULE_LINT_COMMANDS['orchestrator'].replace(
                'ruff check', 'npx pyright',
            ),
            test_command='uv run pytest tests/ --timeout=300',
        )
        plan = derive_verify_plan(
            [source_file, test_file], [mc], None, fake_worktree_reader,
        )

        pyright_run = _run_for(plan, 'orchestrator', 'pyright:')
        assert pyright_run is not None
        assert pyright_run.scope_kind is ScopeKind.FILE_SCOPED
        assert set(pyright_run.scoped_targets) == {source_file, test_file}

        pytest_run = _run_for(plan, 'orchestrator', 'pytest:')
        assert pytest_run is not None
        assert pytest_run.scope_kind is ScopeKind.FILE_SCOPED
        assert pytest_run.scoped_targets == (test_file,)

        # The two slots' records legitimately differ — the point of the assertion.
        assert set(pyright_run.scoped_targets) != set(pytest_run.scoped_targets)

    def test_full_suite_and_skipped_runs_record_no_scoped_targets(self):
        """The negative half of the invariant: empty is CORRECT, and meaningful.

        FULL_SUITE was deliberately not narrowed (D1: a conftest widens
        pytest to the whole suite); a SKIPPED slot never ran at all.
        """
        mc = ModuleConfig(prefix='orchestrator', test_command='uv run pytest tests/ --timeout=300')
        plan = derive_verify_plan(ROOT_CONFTEST_DIFF, [mc], None, fake_worktree_reader)

        pytest_run = _run_for(plan, 'orchestrator', 'pytest:')
        assert pytest_run is not None
        assert pytest_run.scope_kind is ScopeKind.FULL_SUITE
        assert pytest_run.scoped_targets == ()

        lint_run = _run_for(plan, 'orchestrator', 'lint:')
        assert lint_run is not None
        assert lint_run.scope_kind is ScopeKind.SKIPPED
        assert lint_run.scoped_targets == ()

    # -- the fallback branch (step-5) ---------------------------------------

    def test_fallback_path_records_scoped_targets(self):
        """The three FILE_SCOPED sites in _derive_fallback_runs.

        lint and pyright record the whole touched-.py list (``py_files``);
        pytest records only the collectable tests. The bare-``'pytest'``
        default is deliberate — a non-default configured suite is never
        file-scoped by this branch (it runs verbatim), so the bare default is
        the only shape that reaches the FILE_SCOPED pytest site at all.
        """
        files = [_SWEEP_TEST_FILE, SOURCE_ONLY_DIFF[0]]
        plan = derive_verify_plan(files, [], _BARE_FALLBACK_CONFIG, fake_worktree_reader)

        lint_run = _run_for(plan, '__fallback__', 'lint:')
        assert lint_run is not None
        assert lint_run.scope_kind is ScopeKind.FILE_SCOPED
        assert lint_run.scoped_targets == tuple(files)

        pyright_run = _run_for(plan, '__fallback__', 'pyright:')
        assert pyright_run is not None
        assert pyright_run.scope_kind is ScopeKind.FILE_SCOPED
        assert pyright_run.scoped_targets == tuple(files)

        pytest_run = _run_for(plan, '__fallback__', 'pytest:')
        assert pytest_run is not None
        assert pytest_run.scope_kind is ScopeKind.FILE_SCOPED
        assert pytest_run.scoped_targets == (_SWEEP_TEST_FILE,)

        # The pytest slot legitimately narrows further than lint/pyright.
        assert pytest_run.scoped_targets != lint_run.scoped_targets

    def test_scoped_targets_survive_executed_fallback_reconciliation(self):
        """The BROADER instance of the same bug — and why the field lives on PlannedRun.

        ``verify._executed_fallback_plan`` rebuilds every reconciled run's
        command as a fresh ``VerifyCmd(tool=..., raw=<executed string>)``, so
        ``cmd.targets`` is empty for EVERY fallback run — chained or not,
        FILE_SCOPED or not. A ``VerifyCmd``-hosted field would have been
        discarded by that rebuild; a ``PlannedRun``-hosted one rides through
        ``dataclasses.replace(run, module_prefix=..., cmd=...)`` untouched.
        This pins that propagation so a future rewrite of that function
        cannot silently drop it.
        """
        files = [_SWEEP_TEST_FILE, SOURCE_ONLY_DIFF[0]]
        plan = derive_verify_plan(files, [], _BARE_FALLBACK_CONFIG, fake_worktree_reader)
        before = {run.reason.split(':')[0]: run.scoped_targets for run in plan.runs}
        assert set(before) == {'lint', 'pyright', 'pytest'}
        assert all(before.values()), 'precondition: every decision run was FILE_SCOPED'

        # What _build_fallback_config actually produced — subproject-rescoped,
        # so module_prefix and every command differ from the flat decision.
        executed = ModuleConfig(
            prefix='orchestrator',
            lint_command='cd orchestrator && uv run ruff check src/orchestrator/some_module.py',
            type_check_command='cd orchestrator && npx pyright',
            test_command='cd orchestrator && uv run pytest tests/test_sweep.py',
        )
        reconciled = verify._executed_fallback_plan(plan, executed)

        for run in reconciled.runs:
            tool = run.reason.split(':')[0]
            # The reconciliation really ran — this is not a vacuous assertion.
            assert run.module_prefix == 'orchestrator'
            assert run.cmd is not None
            # Rebuilt raw-retained: the structured targets are gone for EVERY run.
            assert run.cmd.raw is not None
            assert run.cmd.targets == ()
            # ...but the plan-record answer survived the rebuild intact.
            assert run.scoped_targets == before[tool]

    # -- the cross-cutting invariant sweep (step-5) -------------------------

    @pytest.mark.parametrize(
        'branch_name, config, role',
        _SWEEP_MODULE_BRANCHES,
        ids=[case[0] for case in _SWEEP_MODULE_BRANCHES],
    )
    @pytest.mark.parametrize(
        'mc_name, mc', _SWEEP_MODULE_CONFIGS, ids=[case[0] for case in _SWEEP_MODULE_CONFIGS],
    )
    @pytest.mark.parametrize(
        'diff_name, files', _SWEEP_DIFFS, ids=[case[0] for case in _SWEEP_DIFFS],
    )
    def test_module_path_scoped_targets_nonempty_exactly_for_file_scoped_runs(
        self, diff_name, files, mc_name, mc, branch_name, config, role,
    ):
        """The regression pin over the MODULE path: command shape x file shape x branch.

        The command-shape axis is what matters here — it decides whether
        ``cmd`` ends up raw-retained, structurally scoped, or unscoped, and
        the invariant must hold across all three. See
        :func:`_assert_scoped_targets_invariant` for the properties asserted.
        """
        plan = derive_verify_plan(files, [mc], config, fake_worktree_reader, role=role)
        _assert_scoped_targets_invariant(plan, files, f'{branch_name}/{mc_name}/{diff_name}')

    @pytest.mark.parametrize(
        'branch_name, config, role',
        _SWEEP_FALLBACK_BRANCHES,
        ids=[case[0] for case in _SWEEP_FALLBACK_BRANCHES],
    )
    @pytest.mark.parametrize(
        'diff_name, files', _SWEEP_DIFFS, ids=[case[0] for case in _SWEEP_DIFFS],
    )
    def test_fallback_path_scoped_targets_nonempty_exactly_for_file_scoped_runs(
        self, diff_name, files, branch_name, config, role,
    ):
        """The same pin over the FALLBACK path — no module-config axis to vary.

        ``_derive_fallback_runs`` consults ``config``'s global commands only,
        never a ModuleConfig, so the module-config axis is deliberately
        absent rather than inert: crossing it in would re-run each case five
        times under ids naming a dimension nothing reads.
        """
        plan = derive_verify_plan(files, [], config, fake_worktree_reader, role=role)
        _assert_scoped_targets_invariant(plan, files, f'{branch_name}/{diff_name}')
