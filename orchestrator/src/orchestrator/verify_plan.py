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


def derive_verify_plan(
    existing_files: list[str],
    module_configs: list[ModuleConfig],
    config: OrchestratorConfig | None,
    worktree_reader: Callable[[str], str | None],
) -> VerifyPlan:
    """Derive the declarative VerifyPlan for one verify attempt (PRD task γ).

    Unifies the twice-fixed scope decision (``scope_module_config`` +
    ``_build_fallback_config``) behind one pure decision layer: file
    classification happens EXACTLY ONCE per file via :func:`classify_file`,
    so D1 (CONFTEST/TEST_DATA -> full-suite pytest) and D2 (STRUCTURAL ->
    unscoped pyright) are each expressed a single time instead of being
    reimplemented per call site.

    Module-config branch (*module_configs* non-empty): each ModuleConfig is
    scoped independently via :func:`_derive_module_runs`.

    The fallback branch (*module_configs* empty — a synthetic module is
    built from *config*'s defaults) is task γ step-10 and not yet
    implemented.
    """
    if module_configs:
        runs: list[PlannedRun] = []
        for mc in module_configs:
            runs.extend(_derive_module_runs(mc, existing_files, worktree_reader))
        return VerifyPlan(runs=tuple(runs))
    raise NotImplementedError(
        'derive_verify_plan fallback branch (module_configs=[]) lands in task 2126 step-10'
    )
