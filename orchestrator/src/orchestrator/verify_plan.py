"""Declarative decision layer for the merge/verify gate (verify-plan-prd.md task γ).

Unifies the twice-fixed scope decision between ``scope_module_config`` and
``_build_fallback_config`` (verify.py) behind a single pure
``derive_verify_plan``: file classification happens EXACTLY ONCE via
``FileKind``, so the class of bug independently fixed in both call sites
(task-1077 conftest, task-1852 data-module) closes by construction.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Literal

from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.pytest_markers import deselecting_expression_for_targets
from orchestrator.verify_cmd import (
    ToolKind,
    VerifyCmd,
    describe_dropped_clauses,
    has_unpreserved_chain_clauses,
    parse_config_command,
    render,
    scope_to,
    split_chain_tail,
    strip_cwd,
)

logger = logging.getLogger(__name__)


class FileKind(Enum):
    """The six mutually-exclusive classifications ``classify_file`` assigns a path.

    Precedence (highest to lowest): CONFTEST > COLLECTABLE_TEST > TEST_DATA >
    STRUCTURAL > SOURCE > INERT. TEST_DATA outranks STRUCTURAL so a
    Protocol-defining data module under ``tests/`` still triggers the full
    suite (D1) rather than merely widening pyright — the structural widening
    (D2) only matters for real source files outside the test tree.

    Signed-off coverage trade-off (task γ review, robustness finding #1):
    CONFTEST/COLLECTABLE_TEST/TEST_DATA outranking STRUCTURAL means a
    Protocol/TypedDict *defined inside* a conftest.py, test_*.py/*_test.py,
    or a tests/ data module is never STRUCTURAL, so touching it alone no
    longer widens pyright to a package-wide run. If that same Protocol is
    also implemented by non-test source elsewhere, a change to its shape
    could previously trigger a full pyright pass that catches a now-broken
    implementor; it is now file-scoped (or skipped) and can miss that
    breakage. Accepted: production Protocol/TypedDict contracts are not
    expected to live inside a test tree, and D1 already prioritizes the
    pytest-side signal over the pyright-side one for test-tree files. Pinned
    by ``TestVerifyPlanClassificationDelegation
    .test_is_structural_python_file_matches_classify_file`` (test_verify.py,
    the ``'a/tests/test_x.py'`` case).
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

    ``scoped_targets`` is the worktree-root-relative file list this (module,
    tool) slot was NARROWED to. **Invariant: non-empty iff ``scope_kind is
    ScopeKind.FILE_SCOPED``** — FULL_SUITE is deliberately unscoped and
    SKIPPED/TRIVIAL never ran, so empty is the correct and MEANINGFUL value
    there, not an absence of information. (Pinned by tests, deliberately not
    by a ``__post_init__`` assert: ``verify._safe_derive_verify_plan_dict``
    swallows exceptions and returns ``None``, so an assert here would
    degrade an invariant violation into losing the ENTIRE plan record —
    silently destroying the very observability this field exists to supply.)

    FIDELITY: it records the scoping INTENT the decision layer computed, NOT
    a guarantee about what the command actually ran on — the executed command
    may be BROADER, never narrower. Two reachable divergences: (i) when
    :func:`_scope_prefix_to_keyword` cannot narrow (the keyword is absent, or
    the prefix parses OPAQUE) it returns the command UNSCOPED while the slot
    is still labelled FILE_SCOPED, so a ``type_check_command='mypy src/'``
    slot records the touched files but type-checks the whole tree; (ii) after
    ``verify._executed_fallback_plan`` reconciliation this field still holds
    the DECISION-layer flat list while ``cmd`` is the executed, possibly
    subproject-rescoped or unscoped command — the same intent-vs-execution
    gap :func:`_derive_fallback_runs`' own "Fidelity caveat" paragraph
    documents for ``module_prefix``/``cmd``. Read it as "what the planner
    decided to narrow to", and ``cmd`` as "what ran".

    It exists (task 3219) because ``cmd.targets`` CANNOT answer "which files
    was this scoped to?" whenever ``cmd`` is raw-retained, and two paths
    routinely produce exactly that: the tail-preserving chained-lint accept
    path in :func:`_scope_prefix_to_keyword` (which every subproject's
    ``lint_command`` hits, since each chains a sibling checker), and every
    run rebuilt by ``verify._executed_fallback_plan``. In both cases the
    rendered string still carries the narrowed file list, but the
    machine-readable record was lost. It is populated uniformly for chained
    AND unchained FILE_SCOPED runs, so a consumer never has to branch on
    ``cmd.raw`` to read it; where ``cmd.targets`` is also populated the two
    agree, which is a consistency check rather than a second source of truth.

    It lives HERE and not on ``VerifyCmd`` because ``VerifyCmd`` is the
    execution model, whose P3 invariant (``verify_cmd.render``) exists to
    guarantee that no field it carries is silently dropped by ``render()`` —
    a deliberately-non-rendering field there would be precisely the class of
    field P3 forbids, and would make P3 self-contradictory. Relaxing P3 to
    let a raw-retained command carry these targets instead was considered
    and rejected: P3 is a general guard, not a lint on one call site, so
    admitting one caller's provenance need would also free ``cargo_scope``
    and ``serial_pytest`` — which mutate raw-retained commands by rewriting
    ``raw`` — to leave stale structured ``targets`` behind with nothing left
    to catch it. Scoping provenance is a PLAN fact, and ``PlannedRun`` is
    the plan record: sibling to ``scope_kind``/``reason``, which already
    record the WHY of a narrowing to this field's WHAT. It also rides
    untouched through ``verify._executed_fallback_plan``'s
    ``dataclasses.replace``-based reconciliation, which a ``VerifyCmd``-hosted
    field would not.
    """

    module_prefix: str
    cmd: VerifyCmd | None
    scope_kind: ScopeKind
    reason: str
    scoped_targets: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        """Render as a plain JSON-native dict (D3) — see ``_verify_cmd_to_dict``."""
        return {
            'module_prefix': self.module_prefix,
            'cmd': _verify_cmd_to_dict(self.cmd) if self.cmd is not None else None,
            'scope_kind': str(self.scope_kind),
            'reason': self.reason,
            'scoped_targets': list(self.scoped_targets),
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


def log_dropped_chain_clauses(
    log: logging.Logger, raw: str, keyword: str, retained: str,
) -> None:
    """Report onto *log* the trailing chain clauses a gate REJECT discarded from *raw*.

    Shared by BOTH scopers — ``_scope_prefix_to_keyword`` below and
    ``verify._scope_to_keyword``, which passes its own module logger — so the
    two records read alike structurally rather than by hand-mirroring. Same
    argument that put the tail-preservation policy in one shared
    ``split_chain_tail`` gate: lockstep the pair cannot drift out of. It lives
    here rather than in ``verify_cmd`` because ``verify`` already imports this
    module (not the reverse), and because keeping ``verify_cmd`` logging-free
    is what lets ``has_unpreserved_chain_clauses`` stay a pure predicate.

    Callers MUST gate this on ``has_unpreserved_chain_clauses(raw, tail)`` and
    MUST call it only on the path that actually rewrites the command: both
    scopers' bail-outs return the whole original, chain and all, so a record
    emitted before them would name clauses that in fact still run.

    *retained* is the caller's truncation point (``head[: idx + len(keyword)]``
    — each caller already computes it, as the argument to
    ``parse_config_command``) and must be a prefix of *raw*. The clause COUNT
    is the top-level `&&` SEGMENT DELTA across it, NOT
    ``len(split_top_level_and(raw)) - 1``: that earlier form counted every
    clause in the whole original, which is right only when the keyword sits in
    segment 0, and every ``cd X && <tool>`` config in this repo puts it in
    segment 1. Measured, it over-reported the root ``type_check_command`` as 5
    dropped clauses when 4 were dropped, and the root ``test_command`` as 15
    when 14 were.

    LEVEL and wording are decided by WHAT WAS DROPPED, via
    ``describe_dropped_clauses``, not by which slot is running. If any dropped
    clause re-invokes the tool at an argv-head position the truncation is an
    intended SAME-TOOL FAN-OUT -> DEBUG, not the WARNING the reverse-dependency
    widening's no-op uses: BOTH root configs are this case and hit it on every
    fallback verify, so a louder level would be steady noise that trains
    operators to ignore the record. If no dropped clause invokes the tool it is
    a genuine SIBLING CHECK that will now never run -> INFO, the
    possible-false-GREEN direction, matching the level
    ``verify._with_junitxml_str`` uses for the missing junit report, whose
    shape this record mirrors.

    That replaces an earlier rule keyed on ``keyword == 'pytest'``, which
    conflated which slot is running with what kind of chain got truncated.
    They come apart in both directions on real configs: this repo's root
    ``test_command`` is a pure pytest FAN-OUT with no sibling checker anywhere
    (so the keyword rule reported the highest-frequency pytest-slot record in
    the repo at INFO, claiming a dropped sibling check that does not exist),
    while a ``cd``-rejected pyright chain ending in ``python3
    scripts/check_pyright_config.py src`` is a genuine sibling check the
    keyword rule buried at DEBUG under fan-out prose.
    """
    dropped, fan_out = describe_dropped_clauses(raw, retained, keyword)
    log.log(
        logging.DEBUG if fan_out else logging.INFO,
        'scope-to-keyword %r dropped %d trailing chain clause(s) from %r '
        '(gate rejected tail preservation) — %s',
        keyword,
        len(dropped),
        raw,
        'an intended same-tool fan-out truncation'
        if fan_out
        # "this command", not "the test command": the sibling case is no
        # longer pytest-specific now that the classification comes from the
        # dropped clauses rather than from the keyword.
        else 'a sibling check chained onto this command will NOT run for this '
        'verify',
        # Attribute the record to the CALLING scoper, not to this shared
        # helper — otherwise every record, from either scoper, points at the
        # same line here and the file/line is useless for telling them apart.
        stacklevel=2,
    )


def _scope_prefix_to_keyword(raw: str, keyword: str, files: list[str]) -> VerifyCmd:
    """Scope *raw* to *files*, first-clause-scoping a raw-retained chain.

    The ``VerifyCmd``-layer counterpart of ``verify._scope_to_keyword`` (which
    operates on — and returns — a shell string): this returns a ``VerifyCmd``
    instead, since ``PlannedRun.cmd`` stores structured commands, not strings.
    Lockstep between the two is now STRUCTURAL rather than a convention kept
    by hand — both route through the single shared
    ``verify_cmd.split_chain_tail`` gate, so neither can drift from the
    other's tail-preservation rule.

    Within the matched segment, *raw* is ALWAYS truncated to everything up to
    and including the first occurrence of *keyword* before being (re-)parsed,
    regardless of whether the untruncated segment would itself have parsed as
    one structured command. Any flags trailing the target are therefore
    dropped: a value-taking flag after the target (e.g. ``'ruff check src/
    --select E'``) would otherwise have its value misread as an extra target
    by ``scope_to``, so truncating first — not scoping the whole parsed
    command — is what keeps this safe as well as byte-identical.

    A trailing ``&&``-chained clause is decided by the gate, not by that
    truncation. A SIBLING CHECKER (a different tool, no ``cd`` sequencing —
    every subproject's ``lint_command`` chains a ``python3 .../check_*.py
    <dir>`` gate after ``ruff check``) is PRESERVED unscoped and verbatim,
    because it asserts a whole-directory invariant that narrowing would
    break. A SAME-TOOL FAN-OUT (the root config's ``cd X && npx pyright``
    chain) is still dropped: preserving it would run two more subprojects
    unscoped AND leave a ``cd ../orchestrator`` that misresolves once
    ``strip_cwd`` has removed the leading ``cd``.

    The PYTEST slot is excluded from tail preservation outright (task 3218):
    ``'pytest'`` is off the gate's ``_TAIL_PRESERVING_KEYWORDS``, so a
    chained ``test_command`` always truncates and the returned ``VerifyCmd``
    stays STRUCTURED (``raw is None``) — which is what keeps
    ``with_junitxml`` and ``with_pytest_timeout`` live on it. A preserved
    tail there would have silently cost the junit report that drives
    ``_extract_failing_test_ids_from_junit``, flake confirmation and the
    per-test timeout floor. The dropped clauses are not silent: a
    multi-clause command whose tail the gate rejects is reported by
    :func:`log_dropped_chain_clauses` below, naming the keyword and the
    dropped-clause count — at DEBUG when a dropped clause re-invokes the tool
    (an intended same-tool fan-out truncation), at INFO when none does (a
    sibling check that will now never run), independent of which slot is
    running. ``verify._scope_to_keyword`` emits that same record through that
    same shared helper, since STRUCTURAL lockstep is this pair's documented
    property. It is emitted only on the rewriting path: both bail-outs below
    return *raw* with its chain intact, so nothing is dropped there and
    nothing is reported.

    If the *keyword*-prefix parses into a structured, non-OPAQUE command, it
    is scoped to *files*. When a tail was preserved the result is returned
    raw-retained — ``VerifyCmd(tool=<scoped tool>, raw=render(scoped) +
    tail)`` — the same shape both bail-outs below already produce, which
    ``render`` reproduces byte-for-byte. The real ``ToolKind`` is kept rather
    than OPAQUE so ``run.cmd.tool`` stays meaningful downstream.

    That raw-retained return drops the structured ``targets`` by design (P3),
    so the narrowed file list is not recoverable from the returned ``cmd``;
    scoping provenance is recorded by the caller on
    :class:`PlannedRun`'s ``scoped_targets`` — see that field's docstring.

    *keyword* absent from *raw*, or the prefix not parsing into one
    recognised structured invocation (P1), leaves *raw* untouched: the
    returned ``VerifyCmd`` is forced raw-retained (``raw=raw`` — the full
    original, never the gate's head) so rendering it reproduces *raw*
    byte-for-byte even when *raw* itself would otherwise have parsed into a
    structured command — a from-scratch render of which is only
    argv-equivalent, not guaranteed byte-identical, to the original string
    (e.g. a ``--directory`` flag renders back as a leading ``cd``).
    """
    parsed = parse_config_command(raw)
    unscoped = parsed if parsed.raw is not None else VerifyCmd(tool=parsed.tool, raw=raw)

    head, tail = split_chain_tail(raw, keyword)
    idx = head.find(keyword)
    if idx == -1:
        return unscoped
    retained = head[: idx + len(keyword)]
    prefix_parsed = parse_config_command(retained)
    if prefix_parsed.tool is ToolKind.OPAQUE or prefix_parsed.raw is not None:
        return unscoped
    # Sited AFTER both bail-outs on purpose — each returns *unscoped*, i.e.
    # *raw* in full with its chain intact, so nothing is dropped there and a
    # record would name clauses that in fact still run. Only the rewriting
    # path below actually discards them. Same record and same level policy as
    # verify._scope_to_keyword's, because it is literally the same call.
    #
    # `retained` is what makes the reported count right: it is the truncation
    # point, so the clauses past it are exactly the ones this call discards.
    if has_unpreserved_chain_clauses(raw, tail):
        log_dropped_chain_clauses(logger, raw, keyword, retained)
    scoped = strip_cwd(scope_to(prefix_parsed, files))
    if not tail:
        return scoped
    return VerifyCmd(tool=scoped.tool, raw=f'{render(scoped)} {tail}')


def _marker_deselecting_expression(
    mc: ModuleConfig,
    collectable_tests: list[str],
    worktree_reader: Callable[[str], str | None],
) -> str | None:
    """The module's effective ``-m`` expression when it provably deselects EVERY target.

    Returns None — "keep today's FILE_SCOPED behaviour" — for every other case,
    including any unreadable file or unparseable expression. See
    :mod:`orchestrator.pytest_markers` for the soundness and cost arguments;
    both reads go through the SAME injected *worktree_reader*, so this function
    introduces no filesystem access of its own.

    WHERE the ini file is looked for: pytest reads ``addopts`` from its
    ROOTDIR, which follows the command's effective cwd — NOT from
    ``mc.prefix``. The two come apart in this very repo: the ``scripts`` and
    ``tests/scripts`` modules both run ``uv run --project shared pytest
    tests/scripts/ ...`` from the REPO ROOT (no ``--directory``), so a
    ``scripts/pyproject.toml`` would never be the config pytest actually
    applies. The effective cwd is therefore taken from the parsed command's
    ``cwd_rel`` (a leading ``cd X &&``, or ``uv run --directory X``), falling
    back to the repo root when the command carries neither.

    Three guards REFUSE rather than guess. Each is fail-safe — no widening,
    i.e. exactly the pre-3494 FILE_SCOPED behaviour:

    1. a ``test_command`` that does not parse as PYTEST at all (``npm test``,
       a shell script): the module's ``addopts`` describe a suite this command
       never invokes, so consulting them would widen on a false premise;
    2. a raw-retained command (``cmd.raw is not None`` — OPAQUE, or an ``&&``
       chain): neither the effective cwd nor which invocation gets scoped is
       recoverable from it. ``_scope_prefix_to_keyword`` truncates a chained
       ``test_command`` to its FIRST ``pytest`` clause, so a probe that read a
       later clause's config would describe a different invocation than the
       one that runs;
    3. only ``pyproject.toml`` is consulted. ``pytest.ini`` / ``setup.cfg`` /
       ``tox.ini`` addopts are invisible here, so a module configured that way
       simply never widens. A deliberate UNDER-fire, never an over-fire.
    """
    if not mc.test_command:
        return None
    parsed = parse_config_command(mc.test_command)
    if parsed.tool is not ToolKind.PYTEST or parsed.raw is not None:
        return None
    cwd_rel = parsed.cwd_rel
    config_path = (
        'pyproject.toml' if not cwd_rel or cwd_rel == '.' else f'{cwd_rel}/pyproject.toml'
    )
    return deselecting_expression_for_targets(
        collectable_tests,
        worktree_reader(config_path),
        mc.test_command,
        worktree_reader,
    )


def _derive_module_runs(
    mc: ModuleConfig,
    existing_files: list[str],
    worktree_reader: Callable[[str], str | None],
    role: Literal['merge', 'task'] = 'task',
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

    *role* (λ, task 2589, R3; widened by task 3294) is the task-role pytest
    floor's policy fork. At ``role == 'task'``, ANY touched SOURCE/STRUCTURAL
    file under the prefix runs the owning module's full ``test_command`` —
    whether or not collectable test files were touched in the same diff.
    Only a test-tree-ONLY diff keeps FILE_SCOPED selection (R3's
    "touched-test-only diffs keep file-scoped selection"). ``role ==
    'merge'`` is unchanged in every cell (R4): it keeps the legacy
    FILE_SCOPED/SKIPPED shape, because the broad merge gate is a separate,
    knob-gated widening (see ``_derive_full_suite_runs``/
    ``merge_verify_breadth``), not this floor.

    WHY the floor sits ABOVE the collectable-test branch (task 3294):
    coverage must be MONOTONE in the diff. λ placed it below, so it only
    fired when the module's touched files contained no collectable test at
    all — a source-only diff paid the owning module's full suite, but that
    same diff plus a test file narrowed to just that test file. Adding a test
    REMOVED coverage. Task 3033 is the incident: its diff touched
    ``workflow.py`` plus tests, the plan file-scoped to 36 items instead of
    the module's ~13188, and a regression in
    ``test_workflow_resume_on_progress.py`` — a DIFFERENT consumer of
    ``workflow.py`` — was structurally invisible and reached a debugger.

    The predicate covers SOURCE ∪ STRUCTURAL, never SOURCE alone, because
    :func:`classify_file` returns STRUCTURAL for a Protocol/TypedDict-defining
    production file only when content is read — and content is read only when
    ``mc.type_check_command`` is set (see the ``need_structural`` guard
    above). ``workflow.py`` is exactly such a file. A SOURCE-only predicate
    would therefore make pytest breadth silently depend on whether a module
    happens to configure a type checker.

    MARKER DESELECTION (task 3494, escalation esc-3292-1): :func:`classify_file`
    is purely PATH-based, so a ``test_*.py`` basename is COLLECTABLE_TEST
    regardless of whether pytest would actually collect anything from it. When
    the module's own ``addopts`` deselect every item in every touched test file
    — as ``orchestrator``'s ``-m 'not warm_lane_bash'`` does for
    ``tests/test_warm_lane_bash_suite.py``, whose module-level ``pytestmark``
    carries that marker — the arm-4 FILE_SCOPED run collects ZERO items, pytest
    exits rc=5, and ``verify_classify._classify_opaque`` classifies rc=5 as RED.
    A false RED on a diff that touched a real, passing test file.

    This is the SECOND instance of the task-1852 class ("the path says
    collectable, pytest collects zero"), and it is fixed where the first one was
    — in THIS scoping layer, by widening to the owning module's full suite —
    rather than by softening rc=5 in the classification layer, which stays a
    genuine RED for a target that vanished for any other reason.

    The probe is FAIL-SAFE in exactly one direction: any unreadable file,
    unparseable expression, or merely-unknown marker yields no widening, i.e.
    precisely the pre-3494 behaviour. Its extra I/O is bounded at one
    ``worktree_reader`` read of the effective-rootdir ``pyproject.toml`` per
    ModuleConfig — through the same content-cached reader used above, so a
    target already read for STRUCTURAL detection costs nothing further — and
    ZERO target reads for a module that declares no ``-m`` expression at all.
    See :func:`_marker_deselecting_expression` for where that ini file is
    looked for and which commands are refused outright.

    The widened run applies the SAME addopts, so the touched file(s) remain
    deselected in it: the widening converts a false RED into a run of the
    module's OTHER tests, not into coverage of the changed lines. That is the
    right trade (a bucketed suite is re-selected by its own dedicated lane, and
    ``not integration``/``not smoke`` tests are excluded deliberately), but it
    is degradation, so the emitted ``reason`` names the still-unrun file(s)
    explicitly rather than reading as "the change was verified".

    The TWIN of this arm in :func:`_derive_fallback_runs` is deliberately NOT
    wired — see that function's own paragraph for why.
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
    # Task 3294: SOURCE ∪ STRUCTURAL, never SOURCE alone — see the *role*
    # paragraph in this function's docstring for why the union is load-bearing.
    production_trigger = next(
        (f for f, k in kinds.items() if k in (FileKind.SOURCE, FileKind.STRUCTURAL)), None,
    )

    runs: list[PlannedRun] = []

    # -- lint: always FILE_SCOPED to every matched file — ruff has no
    # cross-file invariant to protect, unlike pyright's Protocol/TypedDict
    # concern (D2), so there is no "widen to full suite" branch here. --
    if mc.lint_command:
        lint_cmd = _scope_prefix_to_keyword(mc.lint_command, 'ruff check', scoped)
        runs.append(PlannedRun(
            mc.prefix, lint_cmd, ScopeKind.FILE_SCOPED, 'lint: file-scoped to touched file(s)',
            scoped_targets=tuple(scoped),
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
            type_cmd = _scope_prefix_to_keyword(mc.type_check_command, 'pyright', scoped)
            runs.append(PlannedRun(
                mc.prefix, type_cmd, ScopeKind.FILE_SCOPED,
                'pyright: file-scoped to touched file(s)',
                scoped_targets=tuple(scoped),
            ))
    else:
        runs.append(PlannedRun(
            mc.prefix, None, ScopeKind.SKIPPED, 'pyright: no type_check_command configured',
        ))

    # -- pytest: a six-arm cascade, in this order --
    #   1. CONFTEST touched      -> FULL_SUITE (D1: fixtures/hooks affect the
    #      whole subtree)
    #   2. TEST_DATA touched     -> FULL_SUITE (D1: consumed by tests we can't
    #      enumerate from the path alone)
    #   3. role='task' AND a production (SOURCE ∪ STRUCTURAL) file touched
    #      -> FULL_SUITE, the task-3294 floor. It sits HERE, above arm 4, so
    #      that coverage is monotone in the diff: co-committing a test must
    #      not narrow a run the production file alone would have widened.
    #   4a. collectable tests, ALL provably deselected by the module's own `-m`
    #      -> FULL_SUITE, the task-3494 arm. It sits BELOW arms 1-3 so those
    #      stay unreachable-by-construction from the new probe, and it is a
    #      sub-branch of arm 4 rather than a peer because it fires only on the
    #      selection arm 4 would otherwise emit. See the MARKER DESELECTION
    #      paragraph in this function's docstring.
    #   4b. collectable tests     -> FILE_SCOPED to them
    #   5. otherwise             -> an explicit reasoned SKIPPED (the task-1852
    #      "not silent" requirement: never a dropped command)
    # Arm 5 is reachable only at role='merge'. At role='task' every
    # non-conftest, non-test-data .py file under the prefix is either a
    # COLLECTABLE_TEST or SOURCE/STRUCTURAL, and `scoped` is non-empty by the
    # guard above — so arm 3 or arm 4 always fires first. --
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
        elif role == 'task' and production_trigger is not None:
            test_cmd = parse_config_command(mc.test_command)
            # The reason is the operator-facing record of WHY this widened.
            # A mixed diff (production + co-committed tests) must not be
            # described as "source-only" — and must say that the touched
            # tests did NOT narrow it, or the widened run reads as a scoper
            # bug rather than the deliberate task-3294 policy.
            if collectable_tests:
                reason = (
                    f'pytest: production module touched ({production_trigger}) — '
                    'owning-module full suite (task role); co-committed test file(s) '
                    'do NOT narrow it; sibling modules NOT run'
                )
            else:
                reason = (
                    'pytest: source-only diff — owning-module full suite (task role); '
                    'sibling modules NOT run'
                )
            runs.append(PlannedRun(mc.prefix, test_cmd, ScopeKind.FULL_SUITE, reason))
        elif collectable_tests:
            deselecting = _marker_deselecting_expression(mc, collectable_tests, worktree_reader)
            if deselecting is not None:
                test_cmd = parse_config_command(mc.test_command)
                # The widened run applies the SAME addopts, so the very files
                # that triggered the widening stay deselected in it — they are
                # NOT executed here. FULL_SUITE forbids scoped_targets
                # (PlannedRun's invariant), so naming them in the reason is the
                # only channel left for recording which files went unrun; a
                # bare "ran the owning module's full suite" would read as "the
                # change was verified" and be exactly the silent degradation
                # the repo's design invariants forbid.
                unrun = ', '.join(collectable_tests)
                runs.append(PlannedRun(
                    mc.prefix, test_cmd, ScopeKind.FULL_SUITE,
                    f'pytest: touched test file(s) {unrun} are ALL deselected by the '
                    f"module's -m {deselecting!r} — owning-module full suite instead of a "
                    f'zero-collecting file-scoped run (rc=5); those file(s) stay '
                    f'deselected in this run too and are NOT executed by it',
                ))
            else:
                test_cmd = _scope_prefix_to_keyword(mc.test_command, 'pytest', collectable_tests)
                runs.append(PlannedRun(
                    mc.prefix, test_cmd, ScopeKind.FILE_SCOPED,
                    'pytest: file-scoped to touched test file(s)',
                    scoped_targets=tuple(collectable_tests),
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


def _merge_breadth_is_full(config: OrchestratorConfig | None) -> bool:
    """True iff *config* opts into the broad ``merge_verify_breadth='full'`` gate.

    A ``None`` *config* (every role='task' caller, and most existing tests)
    is treated as the shipped default ('scoped') — never accidentally widens
    to the broad merge gate.
    """
    return config is not None and config.merge_verify_breadth == 'full'


def _derive_full_suite_runs(
    mc: ModuleConfig,
    role: Literal['merge', 'task'] = 'merge',
) -> list[PlannedRun]:
    """One ModuleConfig's PlannedRuns under the broad ``merge_verify_breadth='full'``
    gate (λ, task 2589, R1).

    Unlike :func:`_derive_module_runs`, this never file-scopes and never
    consults which files the diff touched — every configured command
    (lint/pyright/pytest) runs FULL_SUITE unconditionally, so a module the
    diff never touched at all is still covered (the "only the touched
    modules are protected" gap the task-role floor deliberately leaves open —
    see :func:`_derive_module_runs`'s docstring; closing it broadly is this
    knob-gated gate's job, not the floor's). A tool with no command
    configured gets an explicit reasoned SKIPPED — the same "no X configured"
    shape :func:`_derive_module_runs` uses — never a fabricated run.
    """
    runs: list[PlannedRun] = []
    for tool_word, attr in (
        ('lint', 'lint_command'),
        ('pyright', 'type_check_command'),
        ('pytest', 'test_command'),
    ):
        cmd_str: str | None = getattr(mc, attr)
        if cmd_str:
            runs.append(PlannedRun(
                mc.prefix, parse_config_command(cmd_str), ScopeKind.FULL_SUITE,
                f'{tool_word}: {role} role, full breadth — registered-module full suite',
            ))
        else:
            runs.append(PlannedRun(
                mc.prefix, None, ScopeKind.SKIPPED, f'{tool_word}: no {attr} configured',
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

    MARKER DESELECTION (task 3494) is deliberately NOT wired here, so this
    branch knowingly retains the pre-3494 behaviour. The bare-default
    ``collectable_tests`` branch below has the identical "the path says
    collectable, pytest collects zero" failure mode whenever the repo root
    carries an ``addopts = "-m 'not X'"`` and the diff touches only
    X-marked test files — it will still plan a zero-collecting FILE_SCOPED run
    that rc=5-REDs. It is left open for two reasons. (i) SOUNDNESS: this branch
    fires only when there are NO registered module_configs, and the "Fidelity
    caveat" above is exactly why — the run this function records is not
    necessarily the run that executes. ``_build_fallback_config`` may rescope a
    fallback diff into a subproject (``cd <sub> && uv run pytest ...``), which
    moves pytest's rootdir and therefore which ``addopts`` apply, so a probe
    reading the ROOT ``pyproject.toml`` here could widen on a config the
    executed command never sees — an over-fire, the one direction task 3494
    forbids. (ii) REACH: the module path covers every subproject registered in
    this repo, so the residual exposure is a project with a marker-partitioned
    root suite and no ``orchestrator.yaml`` anywhere. Closing it properly needs
    the executed-config reconciliation :func:`_executed_fallback_plan` performs,
    which is a different layer than this decision function.

    This caveat describes THIS function's raw return value only. Its caller
    in :func:`run_scoped_verification` reconciles this gap before attaching
    anything to ``VerifyResult.plan``: it folds ``_build_fallback_config``'s
    already-executed ``ModuleConfig`` back onto this decision via
    :func:`_executed_fallback_plan`, so the record a consumer of
    ``VerifyResult.plan`` actually sees reflects *where*/*how* it ran too —
    see :func:`derive_verify_plan`'s own "Fidelity" docstring paragraph.
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
            scoped_targets=tuple(py_files),
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
                scoped_targets=tuple(py_files),
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
        if has_real_suite:
            # Fidelity fix (task γ review, correctness finding): mirrors
            # _build_fallback_config's `config.test_command != 'pytest'`
            # branch, which returns the configured suite VERBATIM (unscoped)
            # whenever ANY trigger is present — collectable tests alone
            # included, not just conftest/test-data. Recording FILE_SCOPED
            # here (as the else branch below does for the bare-default case)
            # would misrepresent what actually executes: a real configured
            # suite is never file-scoped by _build_fallback_config, only
            # ever run verbatim or skipped entirely.
            test_cmd = parse_config_command(test_command)
            runs.append(PlannedRun(
                _FALLBACK_PREFIX, test_cmd, ScopeKind.FULL_SUITE,
                'pytest: configured suite runs verbatim for touched test file(s) '
                '(non-default test_command is never file-scoped)',
            ))
        else:
            test_cmd = scope_to(parse_config_command(test_command or 'pytest'), collectable_tests)
            runs.append(PlannedRun(
                _FALLBACK_PREFIX, test_cmd, ScopeKind.FILE_SCOPED,
                'pytest: file-scoped to touched test file(s)',
                scoped_targets=tuple(collectable_tests),
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

    Fidelity: this plan is no longer just a decision record on either branch
    (task κ, verify-scope-inversion-prd.md) — by the time it reaches
    ``VerifyResult.plan``, both faithfully record what ran. For the
    module-config branch, the caller (:func:`run_scoped_verification`)
    derives this plan once and EXECUTES it directly (see
    :func:`_executed_module_configs_from_plan`). For the fallback branch
    (*module_configs* empty), THIS function's raw return value is still just
    an idealized D1/D2 decision record against the flat *existing_files*
    list — it does NOT model the subproject / mixed-root+subproject
    rescoping that ``_build_fallback_config`` applies (see
    :func:`_derive_fallback_runs`'s "Fidelity caveat" docstring paragraph for
    exactly what it omits) — but the caller folds
    ``_build_fallback_config``'s already-executed ``ModuleConfig`` back onto
    this plan via :func:`_executed_fallback_plan` before attaching it to
    ``VerifyResult.plan``, so the ATTACHED record reflects the actual
    subproject/rescoped command that ran, not the idealized flat
    ``'__fallback__'`` decision alone. A caller consuming THIS function's
    fallback-branch return value directly, bypassing that reconciliation,
    still gets only the decision, not the execution trace.
    """
    if not _has_source_files(existing_files):
        return VerifyPlan(
            runs=(PlannedRun('', None, ScopeKind.TRIVIAL, _TRIVIAL_REASON),),
            needs_pipeline_guard_check=(role == 'merge'),
        )
    if module_configs:
        runs: list[PlannedRun] = []
        merge_full = role == 'merge' and _merge_breadth_is_full(config)
        for mc in module_configs:
            if merge_full:
                runs.extend(_derive_full_suite_runs(mc, role))
            else:
                runs.extend(_derive_module_runs(mc, existing_files, worktree_reader, role=role))
        return VerifyPlan(runs=tuple(runs))
    return VerifyPlan(runs=tuple(_derive_fallback_runs(existing_files, config, worktree_reader)))


# ---------------------------------------------------------------------------
# Reverse-dependency test widening (task 2607)
# ---------------------------------------------------------------------------
#
# Closes a merge-verify blind spot distinct from D1/D2 above: scoped verify
# resolves module_configs from the TASK's own touched modules only, so an
# orchestrator-only diff never considers escalation's ModuleConfig at all —
# even though escalation/tests/test_server.py white-box-imports
# orchestrator.merge_queue and can be broken by an orchestrator-source
# change. This recurred 3x as a RED-main fix-forward (1736->1761,
# 2173->2038, 2435->2604) before being structurally closed here. Mirrors the
# hooks/project-checks reverse-dependency precedent (task 2551,
# DEP_PACKAGES=(shared escalation) -> pyright in consumers), adapted to
# pytest and import-scoped to the specific coupled test files rather than a
# consumer's whole suite (bounding merge-path flake/cost surface).

#: Minimal, hardcoded map of depended-upon package -> dependent packages whose
#: tests are known to import it. A module constant (not operator-configurable)
#: because the coupling is a structural fact of the codebase, mirroring
#: hooks/project-checks' hardcoded DEP_PACKAGES. Extensible by one entry if
#: another cross-package break class recurs.
_REVERSE_TEST_DEPENDENTS: dict[str, frozenset[str]] = {
    'orchestrator': frozenset({'escalation'}),
}


def _package_import_name(prefix: str) -> str:
    """Map a package directory prefix to its Python import name (``'-' -> '_'``)."""
    return prefix.replace('-', '_')


def _imports_any(content: str | None, import_names: list[str]) -> bool:
    """Return True when *content* has a top-level ``from``/``import`` of any of *import_names*.

    The regex is anchored per-line (``re.MULTILINE``) but tolerates leading
    whitespace, so a guarded/indented import — e.g. escalation/tests/
    test_server.py's ``    try:\\n        from orchestrator.merge_queue
    import ...`` — still matches. *content* of ``None`` (unreadable/missing
    file) never matches, same as ``classify_file``'s STRUCTURAL detection
    treats missing content as "simply not detected" rather than an error.
    """
    if content is None:
        return False
    return any(
        re.search(rf'(?m)^\s*(from|import)\s+{re.escape(name)}\b', content)
        for name in import_names
    )


def reverse_dependent_test_targets(
    existing_files: list[str],
    reverse_dep_map: dict[str, frozenset[str]],
    list_pkg_tests: Callable[[str], list[str]],
    read_content: Callable[[str], str | None],
) -> list[tuple[str, list[str]]]:
    """Which dependent packages' test files are coupled to *existing_files*' SOURCE changes.

    Pure decision layer (task 2607): detects which depended-upon packages in
    *reverse_dep_map* had a SOURCE file touched — a strict gate, a path
    starting with ``'<pkg>/src/'`` AND ending in ``.py``, so a non-.py file
    under ``<pkg>/src/`` (e.g. a stray data file) never triggers, and a path
    under ``<pkg>/tests/`` or a bare ``<pkg>/orchestrator.yaml`` never does
    either (only ``<pkg>/src/`` can defeat the ``/src/`` prefix check at
    all). Only a real SOURCE change can break a dependent's import/attribute
    contract; changes to the depended-upon package's own tests/config cannot.

    For each triggered package, collects its dependents into a per-dependent
    SET of triggering import names (``dependent_triggers``) — so a dependent
    triggered by more than one changed depended-upon package (not reachable
    with today's single-entry map, but preserved for extensibility) is
    checked against the union of all their import names and contributes
    exactly ONE ``(dependent, files)`` entry, never a duplicate — regardless
    of how many of ITS depended-upon packages fired or how many of THEIR
    files changed.

    Via the injected *list_pkg_tests* (a dependent's collectable test files)
    and *read_content* (that file's text) callables, keeps only the test
    files that actually import one of the triggering packages. A dependent
    with no importing test files is omitted entirely — never an empty-list
    entry. Both the outer list (by dependent name) and each inner file list
    are sorted for deterministic output.

    *list_pkg_tests*/*read_content* are injected so this stays pure and
    unit-testable — no filesystem, no subprocess — mirroring the
    ``worktree_reader`` injection pattern ``derive_verify_plan`` already uses.
    """
    triggered_pkgs = [
        pkg for pkg in reverse_dep_map
        if any(f.startswith(f'{pkg}/src/') and f.endswith('.py') for f in existing_files)
    ]

    dependent_triggers: dict[str, set[str]] = {}
    for pkg in triggered_pkgs:
        for dependent in reverse_dep_map[pkg]:
            dependent_triggers.setdefault(dependent, set()).add(pkg)

    result: list[tuple[str, list[str]]] = []
    for dependent in sorted(dependent_triggers):
        import_names = [_package_import_name(pkg) for pkg in dependent_triggers[dependent]]
        coupled = sorted(
            f for f in list_pkg_tests(dependent)
            if _imports_any(read_content(f), import_names)
        )
        if coupled:
            result.append((dependent, coupled))
    return result
