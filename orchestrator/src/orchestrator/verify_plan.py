"""Declarative decision layer for the merge/verify gate (verify-plan-prd.md task γ).

Unifies the twice-fixed scope decision between ``scope_module_config`` and
``_build_fallback_config`` (verify.py) behind a single pure
``derive_verify_plan``: file classification happens EXACTLY ONCE via
``FileKind``, so the class of bug independently fixed in both call sites
(task-1077 conftest, task-1852 data-module) closes by construction.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Literal

from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.verify_cmd import VerifyCmd, parse_config_command, scope_to, strip_cwd


class FileKind(Enum):
    """The six mutually-exclusive classifications ``classify_file`` assigns a path.

    Precedence (highest to lowest): CONFTEST > COLLECTABLE_TEST > TEST_DATA >
    STRUCTURAL > SOURCE > INERT. TEST_DATA outranks STRUCTURAL so a
    Protocol-defining data module under ``tests/`` still triggers the full
    suite (D1) rather than merely widening pyright — the structural widening
    (D2) only matters for real source files outside the test tree.
    """

    CONFTEST = 'conftest'
    COLLECTABLE_TEST = 'collectable_test'
    TEST_DATA = 'test_data'
    STRUCTURAL = 'structural'
    SOURCE = 'source'
    INERT = 'inert'


# Matches a class that inherits from Protocol or TypedDict (as a base class).
# Deliberately a cheap content grep rather than an AST parse — mirrors
# verify.py's _PROTOCOL_RE/_TYPEDDICT_RE. Duplicated (not imported) so this
# module stays a standalone, dependency-free decision layer during the
# incremental rollout; unified when verify.py's predicates are rewired to
# delegate to classify_file (task γ step-16).
_PROTOCOL_RE = re.compile(r'\bclass\s+\w+\s*\([^)]*\bProtocol\b')
_TYPEDDICT_RE = re.compile(r'\bclass\s+\w+\s*\([^)]*\bTypedDict\b')


def classify_file(path: str, content: str | None) -> FileKind:
    """Classify *path* into exactly one ``FileKind``, given its (optional) *content*.

    Runs the precedence ladder exactly once per file: a non-``.py`` path is
    INERT (cargo/.rs scoping is handled downstream, in
    ``run_scoped_verification``'s execute step, not here); a ``conftest.py``
    basename is CONFTEST at any depth; a ``test_*.py``/``*_test.py`` basename
    is COLLECTABLE_TEST (the files pytest will actually collect); any other
    path under a ``tests/`` directory is TEST_DATA (a test-tree member that
    is not pytest-collectable); a ``.py`` file whose *content* defines a
    ``Protocol``/``TypedDict`` subclass is STRUCTURAL; everything else is
    SOURCE.

    *content* may be ``None`` — e.g. the caller has no worktree to read from,
    or chose not to fetch content for a file where STRUCTURAL detection is
    moot (a CONFTEST/COLLECTABLE_TEST/TEST_DATA classification never consults
    it). STRUCTURAL is then simply never detected; this never raises.
    """
    if not path.endswith('.py'):
        return FileKind.INERT

    name = path.rsplit('/', 1)[-1]

    if name == 'conftest.py':
        return FileKind.CONFTEST

    if name.startswith('test_') or name.endswith('_test.py'):
        return FileKind.COLLECTABLE_TEST

    if '/tests/' in path or path.startswith('tests/'):
        return FileKind.TEST_DATA

    if content is not None and (_PROTOCOL_RE.search(content) or _TYPEDDICT_RE.search(content)):
        return FileKind.STRUCTURAL

    return FileKind.SOURCE


def _is_conftest(path: str) -> bool:
    """Return True when *path* is a ``conftest.py`` file."""
    return classify_file(path, None) is FileKind.CONFTEST


def _is_collectable_test_file(path: str) -> bool:
    """Return True for files pytest will actually collect when passed as a target.

    NARROW predicate: ``COLLECTABLE_TEST`` only. A data module under ``tests/``
    (e.g. ``shared/tests/silent_fallthrough_allowlist.py``) is test-tree
    membership (``_is_test_file``) but not this — passing it to pytest
    produces rc=5 ("no tests ran").
    """
    return classify_file(path, None) is FileKind.COLLECTABLE_TEST


def _is_test_file(path: str) -> bool:
    """Return True for test-tree members: collectable tests plus test data.

    BROAD predicate: ``COLLECTABLE_TEST ∪ TEST_DATA``, excluding conftest.
    Distinct from ``_is_collectable_test_file`` (narrow, collectable only).

    ``.py``-only contract: TEST_DATA (like every ``FileKind`` besides
    CONFTEST/COLLECTABLE_TEST) is only ever assigned to a ``.py`` path —
    ``classify_file`` maps every non-``.py`` path to INERT regardless of
    directory, so this returns False for a non-``.py`` file under ``tests/``
    (e.g. ``tests/fixture.json``). That is narrower than a hypothetical bare
    "is this path under tests/" predicate would be. Every current caller
    pre-filters to ``.py`` before calling this, so the narrowing is
    behaviorally invisible today — but a future caller passing an unfiltered
    path list should not assume broad test-tree membership from this alone.
    """
    return classify_file(path, None) in (FileKind.COLLECTABLE_TEST, FileKind.TEST_DATA)


class ScopeKind(StrEnum):
    """The four outcomes ``derive_verify_plan`` can assign a (module, tool) slot."""

    FULL_SUITE = 'full_suite'
    FILE_SCOPED = 'file_scoped'
    SKIPPED = 'skipped'
    TRIVIAL = 'trivial'


def _verify_cmd_to_dict(cmd: VerifyCmd) -> dict:
    """Render *cmd* as a plain JSON-native dict (D3).

    ``ToolKind`` (a StrEnum) renders as ``str``; ``base_flags``/``targets``/
    ``wrappers`` tuples render as lists; ``env`` renders as a plain dict —
    every value is a JSON-native primitive, dict, or list, never a nested
    dataclass or Enum member.
    """
    return {
        'tool': str(cmd.tool),
        'uv_project': cmd.uv_project,
        'cwd_rel': cmd.cwd_rel,
        'base_flags': list(cmd.base_flags),
        'targets': list(cmd.targets),
        'env': dict(cmd.env),
        'wrappers': list(cmd.wrappers),
        'raw': cmd.raw,
    }


@dataclass(frozen=True)
class PlannedRun:
    """One (module, tool) slot's planned outcome — the unit ``derive_verify_plan`` emits.

    Split per tool (never one PlannedRun bundling test/lint/type-check) so D1
    (CONFTEST/TEST_DATA -> FULL_SUITE pytest) and D2 (STRUCTURAL -> unscoped
    pyright) are independently expressible for the SAME module, and a skip is
    an explicit reasoned PlannedRun (``cmd=None``, non-empty ``reason``)
    rather than a silently dropped command (the task-1852 "not silent"
    requirement).
    """

    module_prefix: str
    cmd: VerifyCmd | None
    scope_kind: ScopeKind
    reason: str

    def to_dict(self) -> dict:
        """Render as a plain JSON-native dict (D3) — see ``_verify_cmd_to_dict``."""
        return {
            'module_prefix': self.module_prefix,
            'cmd': _verify_cmd_to_dict(self.cmd) if self.cmd is not None else None,
            'scope_kind': str(self.scope_kind),
            'reason': self.reason,
        }


@dataclass(frozen=True)
class VerifyPlan:
    """The full set of planned runs for one verify attempt, plus plan-level flags."""

    runs: tuple[PlannedRun, ...]
    needs_pipeline_guard_check: bool = False

    def to_dict(self) -> dict:
        """Render as a plain JSON-native dict (D3) — attached to ``VerifyResult.plan``."""
        return {
            'runs': [run.to_dict() for run in self.runs],
            'needs_pipeline_guard_check': self.needs_pipeline_guard_check,
        }


def _derive_module_runs(
    mc: ModuleConfig,
    existing_files: list[str],
    worktree_reader: Callable[[str], str | None],
) -> list[PlannedRun]:
    """Derive one ModuleConfig's PlannedRuns — one per (module, tool) slot.

    Filters *existing_files* to ``.py`` files under ``mc.prefix + '/'`` and
    classifies each EXACTLY ONCE via :func:`classify_file` (content is read
    through *worktree_reader* only when ``mc.type_check_command`` is
    configured — only STRUCTURAL detection needs content, and only
    ``.type_check_command``'s outcome can change because of it). Zero
    matching files yields a single explicit SKIPPED PlannedRun rather than
    silently contributing nothing (mirrors ``scope_module_config``'s
    ``return None`` "caller must skip this subproject" contract, upgraded to
    an explicit reasoned run).

    Each per-tool run's ``reason`` is prefixed with the tool name
    (``'lint:'``/``'pyright:'``/``'pytest:'``) so a caller can recover tool
    identity even for a SKIPPED slot, whose ``cmd`` is ``None``.
    """
    prefix = mc.prefix + '/'
    scoped = [f for f in existing_files if f.startswith(prefix) and f.endswith('.py')]
    if not scoped:
        return [PlannedRun(mc.prefix, None, ScopeKind.SKIPPED, 'no files under prefix')]

    # Guard: content is only consulted by classify_file's STRUCTURAL check, so
    # skip the I/O entirely when there is no type-check command to widen.
    need_structural = bool(mc.type_check_command)
    kinds: dict[str, FileKind] = {
        f: classify_file(f, worktree_reader(f) if need_structural else None)
        for f in scoped
    }

    conftest_trigger = next((f for f, k in kinds.items() if k is FileKind.CONFTEST), None)
    test_data_trigger = next((f for f, k in kinds.items() if k is FileKind.TEST_DATA), None)
    structural_trigger = next((f for f, k in kinds.items() if k is FileKind.STRUCTURAL), None)
    collectable_tests = [f for f, k in kinds.items() if k is FileKind.COLLECTABLE_TEST]

    runs: list[PlannedRun] = []

    # -- lint: always FILE_SCOPED to every matched file — ruff has no
    # cross-file invariant to protect, unlike pyright's Protocol/TypedDict
    # concern (D2), so there is no "widen to full suite" branch here. --
    if mc.lint_command:
        lint_cmd = strip_cwd(scope_to(parse_config_command(mc.lint_command), scoped))
        runs.append(PlannedRun(
            mc.prefix, lint_cmd, ScopeKind.FILE_SCOPED, 'lint: file-scoped to touched file(s)',
        ))
    else:
        runs.append(PlannedRun(
            mc.prefix, None, ScopeKind.SKIPPED, 'lint: no lint_command configured',
        ))

    # -- pyright: FULL_SUITE (unscoped) when a STRUCTURAL file is present (D2)
    # — file-scoped pyright cannot verify cross-file Protocol/TypedDict
    # conformance — else FILE_SCOPED. --
    if mc.type_check_command:
        if structural_trigger is not None:
            type_cmd = parse_config_command(mc.type_check_command)
            runs.append(PlannedRun(
                mc.prefix, type_cmd, ScopeKind.FULL_SUITE,
                f'pyright: structural file {structural_trigger} requires unscoped type check',
            ))
        else:
            type_cmd = strip_cwd(scope_to(parse_config_command(mc.type_check_command), scoped))
            runs.append(PlannedRun(
                mc.prefix, type_cmd, ScopeKind.FILE_SCOPED,
                'pyright: file-scoped to touched file(s)',
            ))
    else:
        runs.append(PlannedRun(
            mc.prefix, None, ScopeKind.SKIPPED, 'pyright: no type_check_command configured',
        ))

    # -- pytest: FULL_SUITE (unscoped) when CONFTEST or TEST_DATA is present
    # (D1) — a conftest's fixtures/hooks affect the whole subtree, and a data
    # module under tests/ is consumed by tests we can't enumerate from the
    # path alone — else FILE_SCOPED to collectable tests, else an explicit
    # reasoned SKIPPED (the task-1852 "not silent" requirement: never a
    # dropped command). --
    if mc.test_command:
        if conftest_trigger is not None:
            test_cmd = parse_config_command(mc.test_command)
            runs.append(PlannedRun(
                mc.prefix, test_cmd, ScopeKind.FULL_SUITE,
                f'pytest: conftest touched ({conftest_trigger}) — full suite required',
            ))
        elif test_data_trigger is not None:
            test_cmd = parse_config_command(mc.test_command)
            runs.append(PlannedRun(
                mc.prefix, test_cmd, ScopeKind.FULL_SUITE,
                f'pytest: test-data module touched ({test_data_trigger}) — full suite required',
            ))
        elif collectable_tests:
            test_cmd = strip_cwd(scope_to(parse_config_command(mc.test_command), collectable_tests))
            runs.append(PlannedRun(
                mc.prefix, test_cmd, ScopeKind.FILE_SCOPED,
                'pytest: file-scoped to touched test file(s)',
            ))
        else:
            runs.append(PlannedRun(
                mc.prefix, None, ScopeKind.SKIPPED,
                'pytest: no collectable test files touched — nothing to run',
            ))
    else:
        runs.append(PlannedRun(
            mc.prefix, None, ScopeKind.SKIPPED, 'pytest: no test_command configured',
        ))

    return runs


# Sentinel module_prefix for the fallback (no-module_configs) branch — mirrors
# _build_fallback_config's own '__fallback__' ModuleConfig.prefix literal.
_FALLBACK_PREFIX = '__fallback__'


def _fallback_pytest_targets(files: list[str]) -> list[str]:
    """Directory-mapped pytest targets for the fallback path's conftest case.

    Mirrors ``_build_fallback_config``'s ``has_conftest`` branch (and
    ``_select_subproject_pytest_targets``'s shared shape): each conftest's
    parent directory (a root-level conftest maps to ``'.'``) plus any
    collectable test living outside every such directory, so a test file is
    never silently dropped just because a conftest also touched. Returns the
    bare collectable-test list when there is no conftest at all.
    """
    conftest_files = [f for f in files if classify_file(f, None) is FileKind.CONFTEST]
    collectable_tests = [f for f in files if classify_file(f, None) is FileKind.COLLECTABLE_TEST]
    if not conftest_files:
        return collectable_tests

    conftest_dirs = sorted({
        f.rsplit('/', 1)[0] if '/' in f else '.'
        for f in conftest_files
    })
    if '.' in conftest_dirs:
        outside: list[str] = []
    else:
        outside = [
            t for t in collectable_tests
            if not any(t.startswith(d + '/') for d in conftest_dirs)
        ]
    return conftest_dirs + outside


def _derive_fallback_runs(
    existing_files: list[str],
    config: OrchestratorConfig | None,
    worktree_reader: Callable[[str], str | None],
) -> list[PlannedRun]:
    """Derive the fallback (no-module_configs) branch's PlannedRuns.

    Synthesises a single ``'__fallback__'`` module from *existing_files* and
    *config*'s global commands, applying the SAME D1/D2 rules as
    :func:`_derive_module_runs` with ONE reconciliation: CONFTEST/TEST_DATA
    only widen pytest to FULL_SUITE when a real suite is available — here,
    a non-default configured ``config.test_command`` (the module path's
    analogous "real suite" is ``mc.test_command``, always present by
    construction). When *config* carries only the bare ``'pytest'`` default
    (or no *config* at all), there is no real suite to fall back to: a
    TEST_DATA-only diff degrades to an explicit reasoned SKIPPED rather than
    a fabricated run that would rc=5 "no tests ran" (task-1852 golden,
    commit 7c9b316260). CONFTEST always full-suites regardless — a
    directory target is always safe to run, even bare (task-1077 golden,
    commit cb7277926d).

    Fidelity caveat: this derives an IDEALIZED D1/D2 record against the flat
    *existing_files* list and *config*'s global commands — it does NOT model
    the subproject-scoping (task 2344 ``_single_subproject_prefix``) or
    mixed-root+subproject (task 2368 ``_root_plus_single_subproject_prefix``)
    rescoping that ``_build_fallback_config`` itself actually performs. When
    a fallback diff lands entirely or partly inside a real subproject,
    execution runs ``cd <sub> && uv run pytest ...`` (or the mixed
    root+subproject chain via ``_ROOT_OWNING_TEST_COMMAND``), while this
    function still records a single ``'__fallback__'`` FILE_SCOPED/FULL_SUITE
    run against the flat file list — the recorded ``module_prefix`` and
    ``cmd`` targets/cwd will not match what actually ran in that case. The
    D1/D2 scope_kind decision itself (SKIPPED vs FULL_SUITE vs FILE_SCOPED)
    does not depend on subproject rescoping, so ``plan`` remains a reliable
    record of *why* a decision was made, but not always of *where*/*how* it
    ran for a subproject-shaped fallback diff.
    """
    py_files = [f for f in existing_files if f.endswith('.py')]
    if not py_files:
        return [PlannedRun(_FALLBACK_PREFIX, None, ScopeKind.SKIPPED, 'no .py files touched')]

    lint_command = config.lint_command if config is not None else 'ruff check'
    type_check_command = config.type_check_command if config is not None else 'pyright'
    test_command = config.test_command if config is not None else 'pytest'
    has_real_suite = bool(test_command) and test_command != 'pytest'

    # Guard: content is only consulted by classify_file's STRUCTURAL check, so
    # skip the I/O entirely when there is no type-check command to widen.
    need_structural = bool(type_check_command)
    kinds: dict[str, FileKind] = {
        f: classify_file(f, worktree_reader(f) if need_structural else None)
        for f in py_files
    }
    conftest_trigger = next((f for f, k in kinds.items() if k is FileKind.CONFTEST), None)
    test_data_trigger = next((f for f, k in kinds.items() if k is FileKind.TEST_DATA), None)
    structural_trigger = next((f for f, k in kinds.items() if k is FileKind.STRUCTURAL), None)
    collectable_tests = [f for f, k in kinds.items() if k is FileKind.COLLECTABLE_TEST]

    runs: list[PlannedRun] = []

    # -- lint: always FILE_SCOPED, mirrors the module path (no widening rule
    # for lint — ruff has no cross-file invariant to protect). --
    if lint_command:
        lint_cmd = strip_cwd(scope_to(parse_config_command(lint_command), py_files))
        runs.append(PlannedRun(
            _FALLBACK_PREFIX, lint_cmd, ScopeKind.FILE_SCOPED,
            'lint: file-scoped to touched file(s)',
        ))
    else:
        runs.append(PlannedRun(
            _FALLBACK_PREFIX, None, ScopeKind.SKIPPED, 'lint: no lint_command configured',
        ))

    # -- pyright: FULL_SUITE (unscoped) when a STRUCTURAL file is present
    # (D2) — the gap _build_fallback_config never closed — else FILE_SCOPED.
    if type_check_command:
        if structural_trigger is not None:
            type_cmd = parse_config_command(type_check_command)
            runs.append(PlannedRun(
                _FALLBACK_PREFIX, type_cmd, ScopeKind.FULL_SUITE,
                f'pyright: structural file {structural_trigger} requires unscoped type check',
            ))
        else:
            type_cmd = strip_cwd(scope_to(parse_config_command(type_check_command), py_files))
            runs.append(PlannedRun(
                _FALLBACK_PREFIX, type_cmd, ScopeKind.FILE_SCOPED,
                'pyright: file-scoped to touched file(s)',
            ))
    else:
        runs.append(PlannedRun(
            _FALLBACK_PREFIX, None, ScopeKind.SKIPPED, 'pyright: no type_check_command configured',
        ))

    # -- pytest: D1 (CONFTEST/TEST_DATA -> FULL_SUITE) reconciled against
    # whether a real suite exists to run full. --
    if conftest_trigger is not None or test_data_trigger is not None:
        if has_real_suite:
            trigger = conftest_trigger if conftest_trigger is not None else test_data_trigger
            kind_word = 'conftest touched' if conftest_trigger is not None else 'test-data module touched'
            test_cmd = parse_config_command(test_command)
            runs.append(PlannedRun(
                _FALLBACK_PREFIX, test_cmd, ScopeKind.FULL_SUITE,
                f'pytest: {kind_word} ({trigger}) — full suite required',
            ))
        elif conftest_trigger is not None:
            # No real suite, but a conftest's directory target is always
            # safe to run — never skip it (task-1077 golden cb7277926d).
            targets = _fallback_pytest_targets(py_files)
            test_cmd = scope_to(parse_config_command('pytest'), targets)
            runs.append(PlannedRun(
                _FALLBACK_PREFIX, test_cmd, ScopeKind.FULL_SUITE,
                f'pytest: conftest touched ({conftest_trigger}) — full suite required '
                '(directory-scoped, no configured suite)',
            ))
        else:
            # Bare-fallback data-module (task-1852 7c9b316260): no real suite
            # to run and no conftest to anchor a directory target — an
            # explicit reasoned SKIPPED, never a silent drop.
            runs.append(PlannedRun(
                _FALLBACK_PREFIX, None, ScopeKind.SKIPPED,
                f'pytest: test-data module touched ({test_data_trigger}) — no real suite '
                'configured (bare pytest default); skipping rather than risking rc=5 '
                '"no tests ran" (task 1852)',
            ))
    elif collectable_tests:
        test_cmd = scope_to(parse_config_command(test_command or 'pytest'), collectable_tests)
        runs.append(PlannedRun(
            _FALLBACK_PREFIX, test_cmd, ScopeKind.FILE_SCOPED,
            'pytest: file-scoped to touched test file(s)',
        ))
    else:
        runs.append(PlannedRun(
            _FALLBACK_PREFIX, None, ScopeKind.SKIPPED,
            'pytest: no collectable test files touched — nothing to run',
        ))

    return runs


# Extensions considered "real source" for the TRIVIAL / needs_pipeline_guard_check
# short-circuit below — mirrors verify._has_source_files exactly. NOT the same
# concept as FileKind.INERT: classify_file treats a .rs path as INERT too
# (cargo/.rs scoping is a separate concern left to run_scoped_verification's
# execute step — see FileKind's docstring), but .rs is still real source for
# the purposes of "is this diff trivially a no-op".
_SOURCE_EXTENSIONS = ('.py', '.rs')


def _has_source_files(files: list[str]) -> bool:
    """Return True when *files* contains at least one ``.py`` or ``.rs`` path.

    Duplicated from (not imported from) ``verify._has_source_files`` — this
    module stays a standalone decision layer during the incremental rollout;
    see :func:`classify_file`'s docstring for the same rationale re:
    ``_PROTOCOL_RE``/``_TYPEDDICT_RE``.
    """
    return any(f.endswith(_SOURCE_EXTENSIONS) for f in files)


_TRIVIAL_REASON = 'No source files changed — verify trivially passes'


def derive_verify_plan(
    existing_files: list[str],
    module_configs: list[ModuleConfig],
    config: OrchestratorConfig | None,
    worktree_reader: Callable[[str], str | None],
    role: Literal['merge', 'task'] = 'task',
) -> VerifyPlan:
    """Derive the declarative VerifyPlan for one verify attempt (PRD task γ).

    Unifies the twice-fixed scope decision (``scope_module_config`` +
    ``_build_fallback_config``) behind one pure decision layer: file
    classification happens EXACTLY ONCE per file via :func:`classify_file`,
    so D1 (CONFTEST/TEST_DATA -> full-suite pytest) and D2 (STRUCTURAL ->
    unscoped pyright) are each expressed a single time instead of being
    reimplemented per call site.

    TRIVIAL short-circuit: when *existing_files* has no ``.py``/``.rs`` file
    at all, every module-path/fallback branch below would no-op anyway —
    this unifies the two near-identical "no source files" checks
    ``run_scoped_verification`` currently duplicates once per branch
    (module-config and fallback) into the ONE check here, ahead of both.
    The plan then carries a single ``TRIVIAL`` :class:`PlannedRun` plus
    ``needs_pipeline_guard_check`` — set when *role* is ``'merge'`` —
    recording that the CALLER must still run the impure
    ``_verify_pipeline_guard_requires_full_gate`` subprocess check (e.g. a
    diff touching ``verify.sh`` itself shifts plan-line counts — the
    drift-ambush class) before trusting this trivial verdict.
    ``derive_verify_plan`` never executes that guard itself, staying pure.

    Module-config branch (*module_configs* non-empty): each ModuleConfig is
    scoped independently via :func:`_derive_module_runs`.

    Fallback branch (*module_configs* empty): a single synthetic
    ``'__fallback__'`` module is derived from *config*'s global commands via
    :func:`_derive_fallback_runs`.

    Fidelity: this is a decision record, not an execution trace — the two
    branches above are independently derived from *existing_files* /
    *module_configs* / *config*, not read back from whatever a caller
    actually executed, so two known gaps can make the returned
    :class:`VerifyPlan` diverge from what ran. (1) The fallback branch does
    NOT model the subproject / mixed-root+subproject rescoping that
    ``_build_fallback_config`` applies (see :func:`_derive_fallback_runs`'s
    docstring) — a diff landing in a real subproject executes
    ``cd <sub> && ...`` while the plan still records a flat
    ``'__fallback__'`` run. (2) The module-config branch recomputes each
    module's per-tool ``scope_kind`` from :func:`classify_file` independently
    of ``scope_module_config`` rather than reading back its actual output —
    the two are carefully kept in sync (both consume the same classify_file
    predicates) but are not the same code path, so a future change to one
    must be mirrored in the other to keep this record accurate. Callers that
    need a faithful diagnostic record of what ran, not just why, should treat
    the attached ``VerifyResult.plan`` accordingly.
    """
    if not _has_source_files(existing_files):
        return VerifyPlan(
            runs=(PlannedRun('', None, ScopeKind.TRIVIAL, _TRIVIAL_REASON),),
            needs_pipeline_guard_check=(role == 'merge'),
        )
    if module_configs:
        runs: list[PlannedRun] = []
        for mc in module_configs:
            runs.extend(_derive_module_runs(mc, existing_files, worktree_reader))
        return VerifyPlan(runs=tuple(runs))
    return VerifyPlan(runs=tuple(_derive_fallback_runs(existing_files, config, worktree_reader)))
