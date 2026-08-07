"""Test/lint/typecheck runner for verification stages."""

import asyncio
import concurrent.futures
import contextlib
import errno
import fnmatch
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import time
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    from orchestrator.event_store import EventStore

from shared.proc_group import terminate_process_group
from shared.verify_admission import acquire_task_slot, nice_prefix

from orchestrator import verify_plan
from orchestrator.cargo_scope import discover_workspace_crates, files_to_crates
from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.verify_categories import (
    ARCHIVE_DENY_LIST as _ARCHIVE_DENY_LIST,  # noqa: F401 — re-exported for external consumers
)
from orchestrator.verify_categories import (
    CATEGORY_PRIORITY as _CATEGORY_PRIORITY,  # already list[str] — see verify_categories
)
from orchestrator.verify_categories import (
    INFRA_TRANSIENT_CATEGORIES,
    PREEXISTING_BREAK_SKIP_CATEGORIES,  # noqa: F401 — re-exported for external consumers
    FailureCategory,
    should_archive,
)
from orchestrator.verify_classify import (
    classify_failure,
    is_external_kill_rc,
    is_interpreter_missing_workspace_packages,
    unresolved_top_level_modules,
)
from orchestrator.verify_cmd import (
    ChainSegment,
    ToolKind,
    VerifyCmd,
    apply_pytest_numprocesses,
    cargo_scope,
    govern_cpu,
    has_unpreserved_chain_clauses,
    parse_config_command,
    render,
    reproject,
    scope_to,
    serial_pytest,
    split_and_chain_segments,
    split_chain_tail,
    strip_cwd,
    with_junitxml,
    with_pytest_timeout,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Suppressed-flake audit registry (run_main_tip_sweep retry-on-flake)
# ---------------------------------------------------------------------------

#: In-process registry of flakes suppressed by run_main_tip_sweep's retry-on-
#: flake mechanism.  Each entry is a dict with keys ``sha`` (full hex),
#: ``first_pass_category``, and ``first_pass_cause_hint``.  Accumulates for
#: the lifetime of the process, making suppressed-but-possibly-real failures
#: observable from tests and from any operator inspecting the live object graph
#: — a durable complement to the WARNING log lines that otherwise live only in
#: the log stream.
_suppressed_flake_records: list[dict] = []

# ---------------------------------------------------------------------------
# Infra-class OSError classification (step-1/step-2)
# ---------------------------------------------------------------------------

#: Errno values that indicate a transient infrastructure failure (disk
#: pressure, quota, read-only fs, I/O error, fd exhaustion).  Mirrors the
#: set used in merge_queue._ENOSPC_MARKERS and git_ops.WarmLaneDiskPressure.
_INFRA_ERRNOS: frozenset[int] = frozenset({
    errno.ENOSPC,   # No space left on device
    errno.EDQUOT,   # Quota exceeded
    errno.EROFS,    # Read-only file system
    errno.EIO,      # I/O error
    errno.EMFILE,   # Too many open files (process)
    errno.ENFILE,   # File table overflow (system)
})


def _is_infra_oserror(exc: object) -> bool:
    """Return True iff *exc* is an OSError whose errno is in _INFRA_ERRNOS."""
    return isinstance(exc, OSError) and exc.errno in _INFRA_ERRNOS


class VerifyInfraError(Exception):
    """A transient infrastructure failure detected during the verify phase.

    Raised when an infra-class OSError (ENOSPC, EDQUOT, EROFS, EIO, EMFILE,
    ENFILE) is encountered at a well-known verify call site.  Unlike a bare
    OSError, this typed exception:

    * Is NOT an OSError subclass — caught distinctly before the broad
      ``except Exception`` handler in workflow.run().
    * Carries ``phase`` (which verify sub-step failed) and ``errno`` (the
      original errno value) for structured logging and metadata stamping.

    Modelled on WarmLaneDiskPressure (git_ops.py) but intentionally does NOT
    map to WorkflowOutcome.REQUEUED; it routes to the bounded in-process retry
    / infra_hold path instead.
    """

    def __init__(self, phase: str, errno: int | None, message: str = '') -> None:
        self.phase = phase
        self.errno = errno
        msg = message or f'infra OSError during verify phase={phase!r} errno={errno}'
        super().__init__(msg)


def _scope_to_keyword(cmd: str | None, keyword: str, files: list[str]) -> str | None:
    """Narrow *cmd* to *files* by parsing the prefix through *keyword*, scoping, and rendering.

    Replaces the historical ``_scope_command`` + ``_strip_directory_flag`` /
    ``_strip_leading_cd`` pairing: everything in *cmd* up to and including
    the first occurrence of *keyword* is parsed as one tool invocation
    (folding in any leading ``cd <dir> &&`` or uv ``--project``/
    ``--directory`` context), its targets are replaced by *files*, any cwd
    shift is cleared (the new targets are worktree-root-relative), and the
    result is rendered back to a shell string. This eliminates the old
    dash-token-harvesting mangling (verify.py's historical ``_scope_command``
    regression): ``scope_to`` only ever replaces targets, never re-derives
    flags from unparsed remainder text.

    Returns *cmd* unchanged (no scoping applied) when: *cmd* is ``None``;
    *keyword* is not present in *cmd* (nothing recognisable to scope, e.g. a
    no-op ``'true'`` or a ``mypy``-based command); or the *keyword*-prefix
    does not parse into a single recognised, structured tool invocation
    (P1 — an OPAQUE or raw-retained/unparseable prefix is left untouched
    rather than truncated into a possibly-broken argv).

    Content after the matched *keyword* occurrence is truncated WITHIN the
    matched segment: a value-taking flag positioned after the target (e.g.
    ``'ruff check src/ --select E'``) would otherwise have its value misread
    as an extra target by ``scope_to``, so truncating first is what keeps
    this safe.

    A trailing ``&&``-chained clause is a separate question, decided by
    ``verify_cmd.split_chain_tail``: a SIBLING CHECKER (a different tool, no
    ``cd`` sequencing — every subproject's ``lint_command`` chains a
    ``python3 .../check_*.py <dir>`` gate after ``ruff check``) is PRESERVED
    unscoped and verbatim, because it asserts a whole-directory invariant
    that narrowing would break; a SAME-TOOL FAN-OUT (the root config's ``cd X
    && npx pyright`` chain) is still dropped, since preserving it would run
    two more subprojects unscoped AND leave a ``cd ../orchestrator`` that
    misresolves once ``strip_cwd`` has removed the leading ``cd``.

    The PYTEST slot is excluded from tail preservation outright (task 3218):
    ``'pytest'`` is off the gate's ``_TAIL_PRESERVING_KEYWORDS``, so a
    chained ``test_command`` always truncates and the scoped result stays
    STRUCTURED — which is what keeps ``with_junitxml`` and
    ``with_pytest_timeout`` live on it. A preserved tail there would have
    silently cost the junit report that drives
    ``_extract_failing_test_ids_from_junit``, flake confirmation and the
    per-test timeout floor. The dropped clauses are not silent: a
    multi-clause command whose tail the gate rejects is reported by
    ``verify_plan.log_dropped_chain_clauses`` below, naming the keyword and
    the dropped-clause count — at DEBUG when a dropped clause re-invokes the
    tool (an intended same-tool fan-out truncation), at INFO when none does (a
    sibling check that will now never run), independent of which slot is
    running. The record is emitted only on the rewriting path: both bail-outs
    below return *cmd* with its chain intact, so nothing is dropped there and
    nothing is reported.

    Lockstep with ``verify_plan._scope_prefix_to_keyword`` (the ``VerifyCmd``-
    layer counterpart) is now STRUCTURAL rather than a convention: both route
    through that one shared ``split_chain_tail`` gate. When the gate rejects,
    ``head is cmd`` and ``tail == ''``, so the body below collapses to its
    pre-gate form byte-for-byte. Both bail-outs return *cmd* — the full
    original, never ``head`` — so a gate-accepted command can never be
    silently truncated on the unparseable path.
    """
    if cmd is None:
        return None
    head, tail = split_chain_tail(cmd, keyword)
    idx = head.find(keyword)
    if idx == -1:
        return cmd
    retained = head[: idx + len(keyword)]
    parsed = parse_config_command(retained)
    if parsed.tool is ToolKind.OPAQUE or parsed.raw is not None:
        return cmd
    # Sited AFTER both bail-outs on purpose: each of them returns *cmd* — the
    # whole original, chain and all — so a gate REJECT costs nothing there and
    # a record would name clauses that in fact still run. Only the rewriting
    # path below actually discards them. A record that does not correspond to
    # a real drop is the same failure mode this log exists to avoid.
    #
    # The LEVEL follows what was actually dropped, not which slot is running.
    # A dropped clause that re-invokes the tool at an argv head is an intended
    # same-tool fan-out -> DEBUG, not the WARNING the reverse-dependency
    # widening's no-op uses further down this module: BOTH root configs are
    # that case and hit it on every fallback verify, so a louder level would be
    # steady noise that trains operators to ignore the record. A dropped clause
    # that does NOT invoke the tool is a sibling check that will now never run
    # -> INFO, the possible-false-GREEN direction, the same level
    # `_with_junitxml_str` uses for the missing junit report.
    #
    # Keying that on `keyword == 'pytest'` instead — the first spelling of this
    # record — got it backwards on the live config: this repo's root
    # `test_command` is a pure pytest fan-out with no sibling checker anywhere,
    # and was reported at INFO as a dropped sibling check on every fallback
    # verify that scopes it.
    #
    # The record itself is `verify_plan`'s, emitted onto THIS module's logger.
    # Sharing the one implementation is what makes the two scopers' records
    # read alike structurally rather than by hand-mirroring — the same
    # argument that put the tail-preservation policy in one shared gate.
    # `retained` is the truncation point, so the clauses past it are exactly
    # the ones this call discards — which is what makes the count right.
    if has_unpreserved_chain_clauses(cmd, tail):
        verify_plan.log_dropped_chain_clauses(logger, cmd, keyword, retained)
    rendered = render(strip_cwd(scope_to(parsed, files)))
    return f'{rendered} {tail}' if tail else rendered


def _reproject_str(cmd: str | None, project: str) -> str | None:
    """Reproject a bare ``uv run <tool>`` command string into *project*'s uv context.

    Thin string-level wrapper around ``parse_config_command`` ->
    ``reproject`` -> ``render`` (replaces ``_reproject_bare_uv_run``): a
    no-op when *cmd* is ``None`` or does not parse into a structured,
    non-OPAQUE VerifyCmd (covers ``'true'``/``mypy``-based commands, which
    ``reproject`` would never touch anyway).

    A gated trailing ``&&``-chained clause is carried through VERBATIM and is
    NOT itself reprojected — it is a sibling checker, not a uv invocation
    (see ``split_chain_tail``). Without this, a chain reaching here would
    re-parse as OPAQUE and the ``--project`` injection would be SILENTLY
    dropped; per the fallback path's own comment (and task 2036) the depless
    workspace-root project cannot spawn ruff/pyright, so that is an exit-127
    breakage rather than a cosmetic diff.

    The gate is driven with the keyword ``'uv run'`` because that is exactly
    the head phrase ``reproject`` rewrites: "the thing I am about to rewrite
    lives in segment 0, no later segment invokes it, and there is no ``cd``
    sequencing" is precisely the right admission test here too. ``'uv run'``
    is therefore one of the three keywords on the gate's
    ``_TAIL_PRESERVING_KEYWORDS`` allowlist, and it is there because
    preservation here is LOAD-BEARING rather than merely desirable — see the
    exit-127 paragraph above. (``split_chain_tail`` tests index 0 as an
    argv-head position BEFORE peeling any wrapper, which is what keeps this
    two-token keyword matchable at the front of a ``uv run ... ruff check``
    segment 0.)

    This helper is only ever called on a ``lint_command`` /
    ``type_check_command`` (the fallback path's three call sites), never on a
    ``test_command``. That is a convention, though, not something the gate can
    check — and ``'uv run'`` is a WRAPPER phrase, so a caller who did point it
    at ``'uv run pytest tests/ && python3 check.py tests'`` would clear the
    keyword allowlist and hand the pytest slot a preserved tail, resurrecting
    the exact no-op task 3218 closed. The gate therefore does not rely on the
    convention: its condition 0b independently refuses a tail to any first
    clause that INVOKES pytest, whatever keyword it was called with, so this
    entry is closed against that misuse structurally. When the gate
    rejects, ``head is cmd`` and ``tail == ''``, so the body below collapses
    to its pre-gate form byte-for-byte. All three bail-outs return *cmd* —
    the full original, never ``head`` — so a rejected command can never be
    silently truncated.

    A tail is carried ONLY when the parsed head has no ``cwd_rel``, because
    ``render()`` re-emits ``cwd_rel`` as a leading ``cd X &&`` — a cwd shift
    the preserved tail was never written for. ``split_chain_tail``'s
    ``cd``-TOKEN rejection cannot see it: that gate inspects the INPUT
    string, where the shift is still spelled ``--directory X``. All seven
    module ``lint_command``s are exactly that shape, and the introduced
    ``cd fused-memory &&`` would make their tail's
    ``fused-memory/scripts/check_*.py`` resolve as
    ``fused-memory/fused-memory/scripts/...`` (exit 2) — a spurious RED
    verify on a clean tree.

    The bail forfeits nothing: ``reproject`` is a documented no-op when
    ``cwd_rel is not None`` ("an explicit ``--directory`` is already set"),
    so in exactly the bailed case the discarded expression was a pure
    re-render with no ``--project`` injection to lose. Note this asymmetry
    against the two scopers: they apply ``strip_cwd``, so their ``cwd_rel``
    is always already ``None`` at render time and neither needs this guard.
    ``strip_cwd`` is deliberately NOT used here — unlike the scopers, which
    re-target to worktree-root-relative files, this helper acts on an
    unscoped/bail-through command whose ``--directory`` is load-bearing (it
    selects the directory ruff/pyright run in and find their config from).
    Bail, do not rewrite. Guarding on *tail* first keeps the pre-existing
    no-tail ``--directory`` -> ``cd`` renormalisation byte-identical.
    """
    if cmd is None:
        return None
    head, tail = split_chain_tail(cmd, 'uv run')
    parsed = parse_config_command(head)
    if parsed.tool is ToolKind.OPAQUE or parsed.raw is not None:
        return cmd
    if tail and parsed.cwd_rel is not None:
        return cmd
    rendered = render(reproject(parsed, project))
    return f'{rendered} {tail}' if tail else rendered


def _cargo_scope_str(cmd: str | None, crates: list[str]) -> str | None:
    """Rewrite a cargo ``--workspace`` command to per-crate ``-p`` scoping via VerifyCmd.

    Thin string-level wrapper around ``parse_config_command`` ->
    ``cargo_scope`` -> ``render`` (replaces ``_scope_cargo_workspace``).
    Mirrors ``_scope_to_keyword``/``_reproject_str``'s guard style: a cheap
    ``'--workspace'`` substring pre-check (identical to the old helper's)
    skips parsing entirely for a command with plainly nothing for
    ``cargo_scope`` to act on, and an identity check after mutating skips
    rendering when ``cargo_scope`` legitimately no-ops (a non-cargo
    ToolKind, including OPAQUE — P1) — so such commands come back
    byte-identical rather than reformatted (``render``'s canonical
    flags-then-targets ordering is not guaranteed byte-identical to
    arbitrary input token ordering). This is also what keeps an
    unparseable-but-``--workspace``-bearing command from being mangled: it
    parses OPAQUE, ``cargo_scope`` no-ops on it, and the identity check
    short-circuits before ``render`` is ever called.

    Returns *cmd* unchanged when: *cmd* is ``None``; *crates* is empty;
    ``'--workspace'`` is not a substring of *cmd*; or ``cargo_scope``
    no-ops (non-cargo ToolKind/OPAQUE, or a raw-retained chain with no
    ``--workspace`` in it).
    """
    if cmd is None or not crates or '--workspace' not in cmd:
        return cmd
    parsed = parse_config_command(cmd)
    scoped = cargo_scope(parsed, crates)
    if scoped is parsed:
        return cmd
    return render(scoped)


# Task γ (2126): file classification lives in exactly ONE place —
# verify_plan.classify_file's precedence ladder (CONFTEST > COLLECTABLE_TEST
# > TEST_DATA > STRUCTURAL > SOURCE > INERT). This closes the "same
# file-classification bug independently fixed in both scope_module_config AND
# _build_fallback_config" class by construction (task-1077 conftest,
# task-1852 data-module): both functions call these same four names, so a
# fix (or a future D-invariant) here is automatically visible in both.
#
# _is_conftest / _is_collectable_test_file / _is_test_file are direct
# identity re-exports — step-3/4 (TestDerivedPredicates) already proved them
# behaviorally equivalent to the predicates they replace, across a path
# table covering conftest at various depths, test_*/*_test files, data
# modules under tests/, and plain source — all of that table is ``.py``
# paths, which is what every current caller in this module pre-filters to
# before calling any of these three. The equivalence does NOT extend to
# non-``.py`` paths: classify_file maps any non-``.py`` path to INERT
# regardless of directory, so e.g. _is_test_file('tests/fixture.json') is
# False here, whereas a hypothetical bare "is this under tests/" predicate
# would say True. See verify_plan._is_test_file's docstring for the full
# narrowed contract.
_is_conftest = verify_plan._is_conftest
_is_collectable_test_file = verify_plan._is_collectable_test_file
_is_test_file = verify_plan._is_test_file


def _is_structural_python_file(path: str, content: str) -> bool:
    """Return True when *path* defines a Protocol or TypedDict subclass.

    Delegates to :func:`verify_plan.classify_file`'s precedence ladder: a
    CONFTEST / COLLECTABLE_TEST / TEST_DATA file is never STRUCTURAL even
    when its content defines a Protocol/TypedDict — those classifications
    outrank STRUCTURAL (see ``FileKind``'s docstring; TEST_DATA outranking
    STRUCTURAL is what keeps a Protocol-defining data module under ``tests/``
    triggering the full pytest suite rather than merely widening pyright).
    This only differs from the pre-task-γ content-only check for that
    conftest/test-tree edge case — every existing non-test-tree caller
    (``scope_module_config``'s ``has_structural`` loop, and now
    ``_build_fallback_config``'s) sees identical results, including the
    documented ``.pyi``-exclusion and type-argument false-positive behavior
    (``TestIsStructuralPythonFile``).

    Coverage trade-off, signed off (task γ review, robustness finding #1): a
    test-tree file that defines a Protocol/TypedDict implemented by non-test
    source elsewhere no longer widens pyright to a package-wide run on its
    own change — narrower than the pre-task-γ content-only check for that
    case. See :class:`verify_plan.FileKind`'s docstring for the full
    rationale and the pinning test (``TestVerifyPlanClassificationDelegation
    .test_is_structural_python_file_matches_classify_file``).
    """
    return verify_plan.classify_file(path, content) is verify_plan.FileKind.STRUCTURAL


# Workspace member whose venv declares ``ruff`` (shared/pyproject.toml: ``ruff>=0.4``).
# Used by :func:`_reproject_str` to reproject a bare ``uv run <tool>`` fallback
# command into a uv context that can actually spawn the tool.  Mirrors the proven
# ``uv run --project shared pytest tests/scripts/`` pattern in scripts/orchestrator.yaml
# for this exact repo-root directory.
_FALLBACK_UV_PROJECT = 'shared'

# The repo-root-owning test suite: validates repo-root files (workspace
# registration, config, service scripts) that are not owned by any
# subproject.  Mirrors the same ``uv run --project shared pytest
# tests/scripts/`` pattern as _FALLBACK_UV_PROJECT above and config.yaml's
# own tests/scripts fanout segment.  Used by :func:`_build_fallback_config`'s
# mixed root+single-subproject branch to cover the root-owning portion of a
# diff that also touches exactly one real subproject.
_ROOT_OWNING_TEST_COMMAND = 'uv run --project shared pytest tests/scripts/'


# Splits a `&&`-chained command on each `&&` separator while preserving the
# separator (with its surrounding whitespace) verbatim as its own token: the
# capturing group makes `re.split` yield ``[clause, sep, clause, sep, ...,
# clause]``, so untouched clauses/separators round-trip byte-for-byte when
# rejoined via ``''.join`` (task 3022).
_AND_CLAUSE_SPLIT_RE = re.compile(r'(\s*&&\s*)')


def _cd_clause_target(clause: str) -> str | None:
    """Return *clause*'s target directory when it is a bare ``cd <dir>`` clause.

    Used by :func:`_scope_fallback_tool_to_subproject` (task 3022 amendment)
    to track the net effect of the ``cd`` clauses interleaved through a
    chained ``type_check_command`` (e.g. ``cd fused-memory && npx pyright &&
    cd ../orchestrator && npx pyright && ...``), so a uv ``--project``
    inserted into a LATER clause can be computed relative to where the
    chain actually is at that point — uv resolves a relative ``--project``
    against the shell's current directory, not the worktree root, so
    inserting the bare touched-subproject name unconditionally would
    silently resolve inside the wrong fleet member's directory.

    Returns ``None`` (leaving cwd-tracking unchanged) for anything that
    isn't exactly a two-token ``cd <dir>`` — an unparseable clause
    (unbalanced quotes), a no-op ``cd`` with no argument, or a clause that
    contains more than a lone ``cd``. None of these shapes occur in any
    current config; this mirrors the module's pre-existing `&&`-inside-a-
    quoted-argument boundary (not a new gap introduced here).
    """
    try:
        tokens = shlex.split(clause)
    except ValueError:
        return None
    if len(tokens) == 2 and tokens[0] == 'cd':
        return tokens[1]
    return None


def _rescope_clause_to_subproject(clause: str, project: str) -> str:
    """Reproject a single ``&&``-clause into uv context *project*.

    Extracted from :func:`_scope_fallback_tool_to_subproject` (task 3022) so
    the same parse -> reproject -> render pipeline can be applied to every
    keyword-bearing clause of a chained command, not just one. *project* is
    whatever uv ``--project`` value the caller has already computed for
    THIS clause's position in the chain (task 3022 amendment) — the touched
    subproject's bare name, or that name's path relative to any ``cd`` the
    chain has already executed by this clause (see :func:`_cd_clause_target`)
    — not necessarily the bare subproject name itself.

    Returns *clause* verbatim when it does not parse into a single
    structured tool invocation (P1 — an OPAQUE/unparseable clause, or one
    with ``raw`` retained, is left untouched), or when reprojecting it is a
    no-op (it already carries an explicit ``--project``/``--directory`` —
    don't second-guess an already-set uv context). Otherwise returns the
    clause rendered with its uv context set to *project*.
    """
    parsed = parse_config_command(clause)
    if parsed.tool is ToolKind.OPAQUE or parsed.raw is not None:
        return clause
    reprojected = reproject(parsed, project)
    if reprojected == parsed and parsed.uv_project is None:
        # Not uv-wrapped at all — reproject() deliberately no-ops on this
        # (it only reprojects an ALREADY-bare `uv run <tool>`), but this
        # helper's own job additionally covers "no uv context whatsoever"
        # by prepending one, closing the cold-verify dev-dep race described
        # on :func:`_scope_fallback_tool_to_subproject`.
        reprojected = replace(parsed, uv_project=project)
    if reprojected == parsed:
        return clause
    return render(reprojected)


def _scope_fallback_tool_to_subproject(cmd: str | None, tool_keyword: str, sub: str) -> str | None:
    """Rescope a fallback TYPE/LINT command into subpackage *sub*'s own uv context.

    Cold-verify dev-dep race (task 2355): on a cold throwaway merge-verify
    worktree (:func:`create_throwaway_verify_worktree`) the shared ``.venv``
    starts empty. When a diff lives entirely under a single real subpackage
    (see :func:`_single_subproject_prefix`), the TEST command is already
    scoped to run *inside* that subpackage via ``uv run`` (task 2344), which
    syncs the subpackage's deps — including its dev group (e.g.
    ``hypothesis``) — as a side effect. TYPE and LINT, however, were left
    running in the worktree-root uv context (``_reproject_str``'s hardcoded
    ``_FALLBACK_UV_PROJECT``, or no uv context at all for a non-uv-run
    runner like ``npx pyright``). Because verify runs test/lint/type
    concurrently via one ``asyncio.gather``, TYPE/LINT would race the TEST
    command's sync and could deterministically fail to resolve a dev-only
    import (esc-2293-20). Rescoping TYPE/LINT to ``uv run --project <sub>``
    makes each self-sync *sub*'s deps before the tool runs, closing that race
    regardless of a warm or cold venv.

    *cmd* is expected to already be scoped to the touched files and stripped
    of any leading ``cd`` (i.e. :func:`_scope_to_keyword` has already run) —
    this helper only adds/adjusts the uv context. A multi-subproject
    ``type_check_command`` (``cd X && npx pyright && cd Y && npx pyright &&
    ...``) DOES reach this helper via the has_structural widening path (task
    3022), so EVERY ``&&``-clause carrying *tool_keyword* is rescoped via
    :func:`_rescope_clause_to_subproject` — not just the first — while every
    other clause and ``&&`` separator is left byte-identical.

    uv resolves a relative ``--project`` against the shell's CURRENT
    directory at the point that clause runs, not the worktree root — so for
    a keyword clause that runs after one or more ``cd`` clauses earlier in
    the SAME chain (e.g. the second/third ``npx pyright`` in ``cd
    fused-memory && npx pyright && cd ../orchestrator && npx pyright &&
    ...``), inserting the bare *sub* would resolve inside the wrong fleet
    member's directory instead of *sub* (task 3022 amendment). This helper
    therefore tracks the net effect of every ``cd`` clause seen so far via
    :func:`_cd_clause_target` and passes :func:`_rescope_clause_to_subproject`
    *sub*'s path relative to that tracked position (e.g. ``--project
    ../cockpit`` from inside ``fused-memory``) rather than always the bare
    *sub* — a no-op when no ``cd`` clause precedes the keyword clause (the
    tracked position is still the worktree root, so *sub* relative to it is
    *sub* itself).

    Returns:
        ``None`` when *cmd* is ``None``.
        *cmd* unchanged when *tool_keyword* is not present anywhere (no-op
        ``true``, an unrelated tool like ``mypy``), or when rescoping every
        keyword-bearing clause was a no-op for each of them (each such
        clause is OPAQUE/unparseable, or already carries an explicit
        ``--project``/``--directory`` — don't second-guess it; this
        deliberately also covers a clause explicitly pre-scoped to a
        *different* member than *sub*, which is left alone rather than
        re-targeted).
        *cmd* with every keyword-bearing clause reprojected (bare ``uv run
        <tool_keyword>`` gains ``--project <sub>`` — or *sub*'s path
        relative to any preceding ``cd`` clause in the same chain — or, when
        a clause carries no ``uv run`` wrapper at all, e.g. a bare ``npx
        pyright`` or bare ``pyright`` invocation — with that same ``uv run
        --project <...>`` prepended to it) otherwise.
    """
    if cmd is None:
        return None
    if tool_keyword not in cmd:
        return cmd
    parts = _AND_CLAUSE_SPLIT_RE.split(cmd)
    changed = False
    # Tracks, as a path relative to the worktree root, the net effect of
    # every `cd` clause encountered so far — '.' (the root itself) until the
    # first one. See the uv-resolves-against-cwd paragraph above.
    cwd = '.'
    for i in range(0, len(parts), 2):
        clause = parts[i]
        cd_target = _cd_clause_target(clause)
        if cd_target is not None:
            cwd = os.path.normpath(os.path.join(cwd, cd_target))
            continue
        if tool_keyword not in clause:
            continue
        project = os.path.relpath(sub, cwd)
        rescoped = _rescope_clause_to_subproject(clause, project)
        if rescoped != clause:
            parts[i] = rescoped
            changed = True
    return ''.join(parts) if changed else cmd


# Non-.rs file extensions that are safe to ignore when deciding whether to scope
# cargo commands to individual crates.  These are pure config/data files that
# do not contain executable source code and therefore don't require running a
# non-Rust toolchain alongside cargo.  Any extension NOT in this set (including
# the empty string for files like Dockerfile or LICENSE) triggers a fallthrough
# to ``--workspace`` — the conservative default protects chained commands such
# as ``cargo test --workspace && uv run pytest``.
_CARGO_SCOPE_SAFE_NON_RS_EXTS = frozenset({'.toml', '.yaml', '.yml', '.json', '.md'})

# Rust-specific filenames whose extensions can't be globally whitelisted.
# ``Cargo.lock`` has the ``.lock`` extension, which is also used by non-Rust
# ecosystem lockfiles (``yarn.lock``, ``poetry.lock``, ``uv.lock``), so adding
# ``.lock`` to ``_CARGO_SCOPE_SAFE_NON_RS_EXTS`` would silently admit those
# files and break the polyglot guard for mixed-ecosystem diffs.
# ``rust-toolchain`` is a rustup toolchain pin file with no extension (unlike
# ``rust-toolchain.toml``, which is already handled by the ``.toml`` whitelist).
_CARGO_SCOPE_SAFE_NON_RS_NAMES = frozenset({'Cargo.lock', 'rust-toolchain'})


# Pytest-aware cause-hint patterns. Anchored to whole lines so they don't
# false-match prose. ``_PYTEST_PROGRESS_*`` patterns are used to filter the
# fallback (last-non-blank-line) path so a pytest run killed mid-progress
# doesn't surface "...." dots as the cause hint.
_PYTEST_FAILED_LINE_RE = re.compile(r'^FAILED .+$', re.MULTILINE)
_PYTEST_INTERNALERROR_RE = re.compile(r'^INTERNALERROR>.+$', re.MULTILINE)
_PYTEST_FAILURE_SUMMARY_RE = re.compile(r'^=+ \d+ failed.*=+$', re.MULTILINE)
_PYTEST_TRACEBACK_E_RE = re.compile(r'^E   .+$', re.MULTILINE)
_PYTEST_PROGRESS_BARE_RE = re.compile(r'^[\.FsxXEPp]+(\s+\[\s*\d+%\])?$')
_PYTEST_PROGRESS_FILE_RE = re.compile(r'^\S+\.py [\.FsxXEPp]+(\s+\[\s*\d+%\])?$')


# Bare pytest-xdist worker-crash signature (task 2365). Grounded in
# config.yaml's task-2361 comment: under host CPU oversubscription a starved
# xdist worker crosses the per-test wall-clock ceiling, gets os._exit()'d by
# pytest-timeout's thread method, and --max-worker-restart=0 (kept
# intentionally at 0 — task 1907) turns that into a false-failing per-test
# "node down" attributed to whatever test happens to be running, not a real
# per-test defect. Not anchored to line-start/end (unlike the _PYTEST_* line
# patterns above) since xdist's crash notices can be prefixed by pytest's own
# progress/worker-id decoration.
_XDIST_WORKER_CRASH_RE = re.compile(
    r"node down: Not properly terminated|worker '?gw\d+'? crashed|\[gw\d+\] node down",
    re.MULTILINE,
)


# Small, ENUMERATED allow-list of known load-induced test flakes (esc-2496-3),
# grounded in the same config.yaml task-2361 worker-kill-catalog reasoning as
# _XDIST_WORKER_CRASH_RE above: under host CPU oversubscription, a bare
# second-worker hard-crash ([gwN] node down) can co-occur with an unrelated,
# already-known load-induced flake in a DIFFERENT test — one whose ``FAILED``
# line would otherwise defeat _is_bare_xdist_worker_crash's veto below and
# misroute a code-complete task to the debugger instead of the bounded infra
# retry (task 2496). Kept to a single entry today — the PGID-liveness race in
# test_verify_merge_cancel_end_to_end — to minimize the accepted fail-safe
# tradeoff documented on _is_bare_xdist_worker_crash below.
#
# Patterns are anchored on the full repo-relative node-id path (not just the
# bare filename) since pytest is invoked with cwd=config.project_root and the
# orchestrator verifies multiple projects — a bare ``test_cli.py::...`` match
# would also discount a same-named test living anywhere else, including in an
# unrelated project's own test suite. Future entries should follow the same
# repo-path-anchored convention.
_KNOWN_LOAD_FLAKE_NODEID_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r'(?:^|/)orchestrator/tests/test_cli\.py::test_verify_merge_cancel_end_to_end\b'),
)


def _is_known_load_flake_nodeid(nodeid: str) -> bool:
    """Return True iff *nodeid* matches an enumerated known load-flake test."""
    return any(rx.search(nodeid) for rx in _KNOWN_LOAD_FLAKE_NODEID_RES)


def _is_bare_xdist_worker_crash(output: str) -> bool:
    """Return True when *output* is a bare xdist worker crash with no real failure.

    A hard ``os._exit()`` worker kill (task 2361) produces no assertion
    traceback, so the presence of a genuine pytest failure marker normally
    indicates a real failure occurred alongside the crash. However, under
    host CPU oversubscription a bare crash can co-occur with an unrelated,
    already-known load-induced test flake (esc-2496-3) whose own ``^FAILED
    `` line would otherwise defeat this discriminator and misroute a
    code-complete task to the debugger (task 2496).

    To stay strict while accommodating that case: once the crash signature
    is present, every ``^FAILED `` line is inspected individually. If ANY
    names a test that is not on the narrow, enumerated
    ``_KNOWN_LOAD_FLAKE_NODEID_RES`` allow-list (or has no extractable
    node-id), this returns ``False`` — never mask a real failure. A
    co-occurring ``INTERNALERROR>`` line or ``ERROR`` short-summary line
    (a fixture/setup error or a whole-module collection failure) is
    likewise never attributable to a known FAILED-line flake, so either one
    also forces ``False`` even when every FAILED line is allow-listed —
    those failure surfaces produce no FAILED line of their own, so the
    per-FAILED-line check alone would never see them. Only when there is at
    least one ``FAILED`` line, every one of them is an allow-listed known
    flake, AND no such ERROR/INTERNALERROR surface is present, are the
    accompanying ``^E   `` traceback lines and ``=== N failed ===`` summary
    treated as attributable to those flakes and this returns ``True``. When
    there are NO ``FAILED`` lines at all, this falls back to the original
    strict guard: any ``^E   ``/failure-summary marker suppresses
    reclassification.

    Accepted fail-safe tradeoff: a genuine regression IN an allow-listed
    known-flake test, co-occurring with a crash, is discounted here and
    goes to the bounded infra-retry; if it recurs (a real regression
    doesn't self-heal, unlike a load flake) the retry window is exhausted
    and it lands in infra_hold + escalate_to_human instead of the debugger
    — a human sees it, nothing is silently greened.

    The opposite-direction case — an UNLISTED co-occurring load flake that
    defeats this veto (esc-3514-2 / task 3514) — is deliberately NOT fixed
    by broadening the allow-list; see ``_main_probe_failure_is_isolated_flake``
    (task 3597) for the downstream confirm gate that catches it instead.

    Returns ``False`` for falsy *output* or when the crash signature itself
    is absent.
    """
    if not output:
        return False
    if not _XDIST_WORKER_CRASH_RE.search(output):
        return False
    failed_lines = _PYTEST_FAILED_LINE_RE.findall(output)
    if failed_lines:
        if (
            _PYTEST_INTERNALERROR_RE.search(output)
            or _ERROR_LINE_NODEID_RE.search(output)
            or _ERROR_LINE_FILE_RE.search(output)
        ):
            # A collection/fixture/internal error produces no FAILED line
            # of its own, so the per-line allow-list check below would
            # never see it — veto here instead of silently masking it.
            return False
        for line in failed_lines:
            match = _FAILED_LINE_NODEID_RE.match(line)
            if match is None or not _is_known_load_flake_nodeid(match.group(1)):
                return False
        return True
    return not (
        _PYTEST_TRACEBACK_E_RE.search(output)
        or _PYTEST_FAILURE_SUMMARY_RE.search(output)
    )


# Pytest node-id extraction for the main-tip-sweep isolated-rerun confirm gate
# (task 2370). Three failure surfaces produce a recoverable node-id:
#   1. A genuine assertion/collection-level test failure: the ``FAILED
#      <nodeid>`` summary line pytest prints per failing test (optionally
#      followed by a trailing `` - <reason>``, e.g. `` - AssertionError:
#      ...``).
#   2. A fixture/teardown/collection ERROR: the ``ERROR <nodeid>`` short
#      summary line pytest prints for a test whose setup/teardown raised
#      (test-level, ``::``-qualified), or the bare ``ERROR <file.py>`` form
#      pytest prints when an entire module fails to collect (no single test
#      to name, so the whole file becomes the isolation target). Without
#      this surface, a failing_result mixing a genuine ERROR with one or
#      more load-induced FAILED flakes would extract only the FAILED
#      node-ids, re-run just those, see them pass, and suppress — masking
#      the ERROR, which is never re-run.
#   3. An xdist worker crash (task 1907's --max-worker-restart=0): either an
#      explicit ``crashed while running '<nodeid>'`` notice, or — when that
#      phrasing is absent — the in-progress ``<nodeid>`` line pytest-xdist
#      prints immediately before reporting ``[gwN] node down: Not properly
#      terminated`` for the worker that was running it.
# The FAILED/ERROR summary-line patterns and the node-down-preceding pattern
# are all line-anchored (``^``), like the _PYTEST_* patterns above, so they
# don't false-match indented traceback prose. _XDIST_CRASH_NODEID_RE is the
# one exception: the crash notice is not line-anchored (it can appear
# mid-line), so it instead relies on its distinctive literal "crashed while
# running" prefix to avoid false matches.
_FAILED_LINE_NODEID_RE = re.compile(r'^FAILED\s+(\S+\.py::\S+)', re.MULTILINE)
_ERROR_LINE_NODEID_RE = re.compile(r'^ERROR\s+(\S+\.py::\S+)', re.MULTILINE)
_ERROR_LINE_FILE_RE = re.compile(r'^ERROR\s+(\S+\.py)(?:\s|$)', re.MULTILINE)
_XDIST_CRASH_NODEID_RE = re.compile(
    r"crashed while running '?([^'\s]+\.py::[^'\s]+)'?", re.MULTILINE,
)
_XDIST_NODE_DOWN_PRECEDING_NODEID_RE = re.compile(
    r'^(\S+\.py::\S+?)\s*\n\[gw\d+\] node down: Not properly terminated',
    re.MULTILINE,
)


def _extract_failing_test_ids(test_output: str) -> list[str]:
    """Extract pytest node-ids of failing/errored/crashed tests from *test_output*.

    Scans for the three failure surfaces documented above the module-level
    patterns: ``FAILED <nodeid>`` summary lines, ``ERROR <nodeid>`` /
    ``ERROR <file.py>`` summary lines (fixture/teardown/collection errors),
    and xdist worker-crash notices (both the explicit ``crashed while
    running '<nodeid>'`` phrasing and the in-progress ``<nodeid>`` line
    immediately preceding a ``node down: Not properly terminated`` marker).

    Returns node-ids in first-seen (leftmost-match) order, de-duplicated.
    Returns ``[]`` for falsy *test_output* or output with no recoverable
    node-id (a non-test failure such as a lint/type error block, or a
    worker-crash notice with no adjacent node-id) — the caller
    (``confirm_main_tip_failure_is_real``) treats an empty list as
    "unconfirmable" and fails safe to alarm rather than guessing.
    """
    if not test_output:
        return []
    matches: list[tuple[int, str]] = []
    for pattern in (
        _FAILED_LINE_NODEID_RE,
        _ERROR_LINE_NODEID_RE,
        _ERROR_LINE_FILE_RE,
        _XDIST_CRASH_NODEID_RE,
        _XDIST_NODE_DOWN_PRECEDING_NODEID_RE,
    ):
        for m in pattern.finditer(test_output):
            matches.append((m.start(1), m.group(1)))
    matches.sort(key=lambda item: item[0])
    seen: set[str] = set()
    ordered: list[str] = []
    for _, node_id in matches:
        if node_id not in seen:
            seen.add(node_id)
            ordered.append(node_id)
    return ordered


def _extract_failing_test_ids_from_junit(path: Path) -> list[str] | None:
    """Parse a pytest junitxml report at *path* into failing/errored test ids.

    Task μ (verify-scope-inversion-prd.md): the STRUCTURED counterpart to
    :func:`_extract_failing_test_ids` above (which regexes pytest stdout) —
    this is the baseline-attribution signal, parsed via stdlib
    ``xml.etree.ElementTree`` rather than a regex. The SAME parser feeds both
    the per-main-SHA baseline probe and a branch's merge-gate result, so diff
    consistency — not exact pytest-node-id fidelity — is what matters (see
    ``diff_new_failures``/``is_wholly_preexisting``).

    A ``<testcase>`` counts as failing iff it has a ``<failure>`` or
    ``<error>`` child (a ``<skipped>`` child does not count). Its id is
    ``f'{classname}::{name}'``; when ``classname`` is absent, falls back to
    ``f'{file}::{name}'`` (the ``file`` attribute some junit writers emit),
    then to the bare ``name`` when neither is present. ``<testcase>``
    elements are found via ``root.iter('testcase')`` so both the modern
    ``<testsuites><testsuite>...`` wrapping and a bare ``<testsuite>`` root
    are handled uniformly, and multiple ``<testsuite>`` blocks are all
    covered.

    Returns:
        - ``None`` when *path* does not exist, or the file is empty/malformed
          (``ET.ParseError``) — "no junit collected", the B3 degrade signal
          callers fall back on.
        - ``[]`` when the report parses but no testcase is failing/errored
          ("junit collected, zero failing" — main/branch genuinely clean).
        - Otherwise a sorted, de-duplicated list of failing/errored ids.

    Never raises: any unexpected parse-time exception is treated the same as
    ``ET.ParseError`` (fail-soft to ``None``).
    """
    try:
        tree = ET.parse(path)
    except (OSError, ET.ParseError):
        return None
    root = tree.getroot()
    if root is None:
        return None

    ids: set[str] = set()
    for testcase in root.iter('testcase'):
        if testcase.find('failure') is None and testcase.find('error') is None:
            continue
        name = testcase.get('name', '')
        classname = testcase.get('classname')
        if classname:
            test_id = f'{classname}::{name}'
        else:
            file_attr = testcase.get('file')
            test_id = f'{file_attr}::{name}' if file_attr else name
        ids.add(test_id)
    return sorted(ids)


def _extract_cause_hint(output: str) -> str:
    """Extract a one-line failure hint from command output.

    Uses a pattern ladder (first match wins):
    1. ``FAILED test::name`` — pytest failure lines (start of line)
    2. ``INTERNALERROR>`` — pytest collection / plugin errors
    3. ``===== N failed in Xs =====`` — pytest summary line
    4. ``error: …``         — cargo/clippy surface errors
    5. ``… FAILED``         — Rust test runner failure lines
    6. ``Command timed out after Ns: …`` — our own timeout wrapper
    7. ``ERROR: …``         — flock/script wrapper errors
    8. ``… npm (ERR!|error) …`` — npm errors
    9. last ``E   …`` line  — pytest traceback (last match wins; most specific)
    10. fallback: last non-blank line of output, with pytest progress lines
        filtered. If only progress lines remain, returns an opaque-exit message.

    Returns ``''`` for None, empty, or whitespace-only input.
    Result is stripped to a single line and capped at 200 chars.
    """
    if not output or not output.strip():
        return ''

    _HINT_PATTERNS = [
        _PYTEST_FAILED_LINE_RE,
        _PYTEST_INTERNALERROR_RE,
        _PYTEST_FAILURE_SUMMARY_RE,
        re.compile(r'^error: .+$', re.MULTILINE),
        re.compile(r'^.+\s+FAILED$', re.MULTILINE),
        re.compile(
            r'^Command (?:timed out after \d+s|clock-stop timed out[^:]+):.+$',
            re.MULTILINE,
        ),
        re.compile(r'^ERROR: .+$', re.MULTILINE),
        re.compile(r'^.*npm (ERR!|error).*$', re.MULTILINE),
    ]

    for pattern in _HINT_PATTERNS:
        m = pattern.search(output)
        if m:
            return m.group(0).strip()[:200]

    # Pytest traceback E-line — capture LAST one (most specific).
    e_matches = _PYTEST_TRACEBACK_E_RE.findall(output)
    if e_matches:
        return e_matches[-1].strip()[:200]

    # Filtered fallback: drop pytest progress lines before the last-non-blank.
    # Without this, a killed-mid-run pytest surfaces lines like
    # "orchestrator/tests/test_scheduler.py .." as the cause hint.
    meaningful = [
        line for line in reversed(output.splitlines())
        if line.strip()
        and not _PYTEST_PROGRESS_BARE_RE.match(line)
        and not _PYTEST_PROGRESS_FILE_RE.match(line)
    ]
    if not meaningful:
        return 'opaque test exit (no failure markers in output)'
    return meaningful[0].strip()[:200]


# ---------------------------------------------------------------------------
# Failure-anchored excerpting (PART 1, task 2549) — VerifyResult.failure_report
# used to slice a fixed test_output[-3000:] tail for its "## Test Failures"
# section. For a long suite that tail is a PASS-wall that elides the actual
# failing test, so downstream block reports/investigators chase the wrong
# thing. _failure_anchored_excerpt locates failure markers instead and
# excerpts bounded context windows around them, falling back to the tail
# only when no marker is found. A separate, deliberately independent concern
# from _extract_cause_hint (the one-line human hint ladder above) and from
# verify_classify.classify_failure (the machine FailureCategory) — this only
# decides what raw text goes into the report's code block.
# ---------------------------------------------------------------------------

# "Strong" markers: structurally specific to a real failure line, effectively
# never emitted by unrelated prose or fixture/decoy data (e.g. a test whose
# body happens to print the literal string "FAIL:" to demonstrate parsing
# behavior). Reuses the existing _PYTEST_FAILED_LINE_RE / _PYTEST_INTERNALERROR_RE
# constants above so the excerpt anchors on the same grounded pytest-line
# shapes _extract_cause_hint already relies on.
_TRAILING_FAILED_RE = re.compile(r'^.+\s+FAILED\s*$', re.MULTILINE)
_TRACEBACK_HEADER_RE = re.compile(r'Traceback \(most recent call last\)')
_RUST_PANIC_RE = re.compile(r'\bpanicked\b')
_RUSTC_ERROR_CODE_RE = re.compile(r'error\[E\d+\]:')

_STRONG_FAILURE_MARKER_RE = re.compile(
    '|'.join(p.pattern for p in (
        _PYTEST_FAILED_LINE_RE,
        _TRAILING_FAILED_RE,
        _PYTEST_INTERNALERROR_RE,
        _TRACEBACK_HEADER_RE,
        _RUST_PANIC_RE,
        _RUSTC_ERROR_CODE_RE,
    )),
    re.MULTILINE,
)

# "Weak" markers: real failure signal, but generic enough that fixture/decoy
# output can emit lookalikes (a test asserting on parser behavior might print
# a literal "FAIL:" line; a benign log line can start with "error:"). Only
# consulted for anchoring windows — never preferred over a strong marker's
# window when the excerpt must be capped (see _failure_anchored_excerpt).
_BARE_FAIL_COLON_RE = re.compile(r'^FAIL:\s', re.MULTILINE)
_GENERIC_ERROR_LINE_RE = re.compile(r'^error:', re.MULTILINE)

_WEAK_FAILURE_MARKER_RE = re.compile(
    '|'.join(p.pattern for p in (
        _BARE_FAIL_COLON_RE,
        _GENERIC_ERROR_LINE_RE,
        _PYTEST_TRACEBACK_E_RE,
    )),
    re.MULTILINE,
)

# Union of strong+weak, plus a trailing bare "FAILED" substring alternative —
# used by failure_report() as the emit-gate for the ## Test Failures section
# (broadened from the legacy `'FAILED' in test_output` substring check to
# "any failure marker present, OR that legacy substring itself"). The
# trailing `FAILED` alternative is what makes this a *genuine* superset: a
# structurally-anchored marker alone is NOT one, since e.g. a cargo summary
# line like "test result: FAILED. 3 passed; 1 failed" has no line-start/
# line-end FAILED (see _PYTEST_FAILED_LINE_RE / _TRAILING_FAILED_RE) and no
# strong/weak marker at all, yet still contains the old substring — without
# this alternative such output would silently stop emitting the section.
# Note this union is the emit *gate* only; _failure_anchored_excerpt anchors
# windows using _STRONG_FAILURE_MARKER_RE/_WEAK_FAILURE_MARKER_RE directly
# (not this pattern), so a gate-only match like the cargo line above falls
# back to _failure_anchored_excerpt's tail slice — identical to pre-anchoring
# (legacy) behavior for that case, i.e. no regression there either.
_FAILURE_MARKER_RE = re.compile(
    _STRONG_FAILURE_MARKER_RE.pattern + '|' + _WEAK_FAILURE_MARKER_RE.pattern + '|FAILED',
    re.MULTILINE,
)


def _failure_anchored_excerpt(output: str, *, cap: int = 3000, window: int = 10) -> str:
    """Excerpt *output* around failure markers instead of a fixed tail slice.

    Locates every failure-marker line (FAILED / trailing "... FAILED" /
    INTERNALERROR> / Traceback / panicked / error[E..]: / bare FAIL: /
    generic error: / pytest "E   " lines), builds a *window*-line context
    block around each, and merges overlapping/adjacent blocks into one
    contiguous excerpt (joining any remaining disjoint blocks with a short
    "..." elision separator). Falls back to ``output[-cap:]`` (today's
    behavior) when no marker is found at all. Pure string function, no I/O.

    Decoy suppression (best-effort): when the merged excerpt exceeds *cap*,
    windows anchored ONLY on a "weak" marker (bare ``FAIL:``, generic
    ``error:``, or a lone pytest ``E   `` line — patterns fixture/decoy data
    can emit incidentally) are dropped before falling back to hard
    tail-truncation, so a "strong" structured marker (FAILED <path>::<test>,
    trailing "... FAILED", Traceback, panicked, error[E..]:, INTERNALERROR>)
    is preferred to survive the cap.

    Short-circuit: when *output* already fits within *cap*, it is returned
    unchanged (byte-identical to today's tail slice, which is a no-op for
    anything shorter than the cap) — windowing only matters once *output*
    is long enough to need trimming in the first place.
    """
    if len(output) <= cap:
        return output

    lines = output.split('\n')
    strong_idxs = {i for i, ln in enumerate(lines) if _STRONG_FAILURE_MARKER_RE.search(ln)}
    weak_idxs = {
        i for i, ln in enumerate(lines)
        if i not in strong_idxs and _WEAK_FAILURE_MARKER_RE.search(ln)
    }
    marker_idxs = strong_idxs | weak_idxs
    if not marker_idxs:
        return output[-cap:]

    # One window per marker line, sorted by start so overlap-merging is a
    # single left-to-right sweep.
    raw_windows = sorted(
        (max(0, i - window), min(len(lines), i + window + 1), i in strong_idxs)
        for i in marker_idxs
    )
    merged: list[list] = []
    for start, end, is_strong in raw_windows:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2] = merged[-1][2] or is_strong
        else:
            merged.append([start, end, is_strong])

    def _render(windows: list[list]) -> str:
        blocks = []
        prev_end = None
        for start, end, _is_strong in windows:
            if prev_end is not None and start > prev_end:
                blocks.append('...')
            blocks.append('\n'.join(lines[start:end]))
            prev_end = end
        return '\n'.join(blocks)

    excerpt = _render(merged)
    if len(excerpt) <= cap:
        return excerpt

    # Over budget: drop windows with no strong marker (decoy de-prioritization).
    strong_only = [w for w in merged if w[2]]
    if strong_only:
        excerpt = _render(strong_only)
        if len(excerpt) <= cap:
            return excerpt

    # Still over cap (or no strong windows survived) — hard tail-truncate.
    return excerpt[-cap:]


# _ARCHIVE_DENY_LIST, _CATEGORY_PRIORITY, and PREEXISTING_BREAK_SKIP_CATEGORIES
# are re-exported (imported above) from orchestrator.verify_categories, the
# single source of truth for the per-category policy (archive / priority /
# preexisting-probe / infra-transient) — see CATEGORY_POLICY there for the
# full table and the rationale behind each category's flags. Previously these
# were three hand-written literals that had to be kept in sync by hand (see
# bug_history: task 2048, one category change required 4 registry edits + 2
# inline sets); deriving them from one table eliminates that hazard.

# Process-wide cache for main-probe results: avoids redundant worktree-add +
# full-build/test re-runs when the same task retries (helper returned False, debugger
# ran, verify failed again) or when sibling tasks probe the same unchanged main.
# Key: (main_sha, category, normalised_cause_hint); Value: (probe_time, is_preexisting).
_PROBE_CACHE: dict[tuple[str, str, str], tuple[float, bool]] = {}
_PROBE_CACHE_TTL: float = 300.0  # 5 minutes; main_sha changes on every hotfix merge


# ---------------------------------------------------------------------------
# Task μ (verify-scope-inversion-prd.md) — per-main-SHA baseline attribution:
# pure failing-test-id diff helpers.  Both operate on plain iterables of test
# ids (typically VerifyResult.failing_test_ids) and are entirely side-effect
# free — no cache, no I/O.  They are the decision core of B1 ("a broad gate
# blocks a branch only for failing test ids NOT already red on main"); the
# surrounding cache/probe machinery (_BASELINE_FAILING_IDS_CACHE and friends,
# added alongside verify_failure_is_preexisting_on_main) feeds them a
# *baseline* set collected from a real main-tip probe.
# ---------------------------------------------------------------------------


def diff_new_failures(branch: Iterable[str], baseline: Iterable[str]) -> frozenset[str]:
    """Return the ids present in *branch* but absent from *baseline*.

    Plain set difference (``frozenset(branch) - frozenset(baseline)``), just
    named/typed for the baseline-attribution call sites.  Pure; no I/O, no
    caching, no ordering guarantee beyond frozenset's own (callers that need
    a stable order should ``sorted()`` the result).
    """
    return frozenset(branch) - frozenset(baseline)


def is_wholly_preexisting(branch: Iterable[str], baseline: Iterable[str]) -> bool:
    """True iff *branch* is non-empty and every id in it already appears in *baseline*.

    An empty *branch* (no failing ids at all) returns False — "wholly
    preexisting" is meaningless with nothing to attribute; a passing verify
    is the caller's concern, not this classifier's.
    """
    branch_set = frozenset(branch)
    if not branch_set:
        return False
    return not diff_new_failures(branch_set, baseline)


# Process-wide cache for the per-main-SHA failing-test-id BASELINE (task μ,
# verify-scope-inversion-prd.md, B2): distinct from _PROBE_CACHE above (that
# one caches a bool — "is THIS specific failure preexisting"; this one caches
# the FULL SET of ids already failing on a given main tip). Seeded for free
# on every successful merge+full gate run (merge_queue.py's
# _run_post_merge_verify pass path — see seed_main_baseline's docstring) so
# steady-state lookups never pay for a probe; a probe only runs on a genuine
# cold-start miss. Same TTL discipline as _PROBE_CACHE (mirrors its
# docstring/shape) so a long-idle orchestrator doesn't pin a stale baseline
# forever.
# Key: main_sha; Value: (seeded_or_probed_at, failing_test_ids frozenset).
_BASELINE_FAILING_IDS_CACHE: dict[str, tuple[float, frozenset[str]]] = {}


def seed_main_baseline(main_sha: str, ids: Iterable[str]) -> None:
    """Seed (or refresh) the per-main-SHA failing-id baseline cache for free.

    Called from the PASS path of a merge+full gate run (merge_sha IS the
    merged tree that is about to CAS-advance to become the next main tip —
    see merge_queue.py's ``_run_post_merge_verify``), so in steady state
    ``main_baseline_failing_ids`` below is always a cache hit and never pays
    for a probe (B2).
    """
    _BASELINE_FAILING_IDS_CACHE[main_sha] = (time.monotonic(), frozenset(ids))


async def main_baseline_failing_ids(
    config: 'OrchestratorConfig',
    module_configs: 'list[ModuleConfig]',
    git_ops: object,
    main_sha: str,
) -> 'frozenset[str] | None':
    """Return the set of test ids already failing on *main_sha*, cache-first.

    Cache hit (seeded by a prior gate pass, or a prior probe of this same sha
    within the TTL window): returned immediately — no probe, no worktree.

    Cache miss: runs exactly ONE full-suite, merge-role probe of bare main,
    reusing the same ``ephemeral_worktree(WorktreeKind.MAIN_PROBE, ...,
    warm_seed=True)`` + ``run_scoped_verification`` lifecycle
    :func:`verify_failure_is_preexisting_on_main` uses for its own probe —
    a leaseless, local-only probe that NEVER routes through
    :class:`~orchestrator.verify_runner.HostAllocator` or
    :class:`~orchestrator.verify_runner.RemoteRunner` (see that function's
    docstring for the full LEASE-SAFETY & HOST-AFFINITY rationale, which
    applies identically here). The probe passes no ``task_files`` — full
    suite, no scoping — so its id-set is apples-to-apples with a merge+full
    branch verify's id-set.

    A probe that doesn't yield a junit-derived id set
    (``failing_test_ids is None`` — OPAQUE/non-pytest command, or the probe
    itself errored) returns ``None`` (B3 degrade) and is deliberately **not**
    cached, so the next caller retries rather than being stuck with a
    falsely-empty baseline for the whole TTL window.

    Does not alter deferred-probe scheduling/transport (G4, task 2564) — the
    probe body reused here is exactly the one that function already owns.
    """
    from orchestrator.git_ops import EphemeralWorktreeError, WorktreeKind

    if not main_sha:
        return None

    _now = time.monotonic()
    _cached = _BASELINE_FAILING_IDS_CACHE.get(main_sha)
    if _cached is not None:
        _cached_at, _cached_ids = _cached
        if _now - _cached_at < _PROBE_CACHE_TTL:
            logger.debug(
                'main_baseline_failing_ids: cache hit (main_sha=%.8s, %d id(s))',
                main_sha, len(_cached_ids),
            )
            return _cached_ids

    try:
        async with git_ops.ephemeral_worktree(  # type: ignore[union-attr]
            WorktreeKind.MAIN_PROBE, main_sha, warm_seed=True,
        ) as tmp_path:
            try:
                probe_result = await run_scoped_verification(
                    tmp_path, config, module_configs,
                    task_files=None,
                    max_retries=0,
                    role='merge',
                )
            except Exception:
                logger.warning(
                    'main_baseline_failing_ids: probe verify raised', exc_info=True,
                )
                return None

            if probe_result.failing_test_ids is None:
                # OPAQUE / non-pytest / probe-side failure to collect a junit
                # report — degrade (B3). Deliberately not cached: a transient
                # probe hiccup shouldn't pin "no baseline" for the TTL window.
                logger.debug(
                    'main_baseline_failing_ids: probe collected no junit ids '
                    '(main_sha=%.8s) — degrading to None (B3)', main_sha,
                )
                return None

            ids = frozenset(probe_result.failing_test_ids)
            seed_main_baseline(main_sha, ids)
            return ids

    except EphemeralWorktreeError as e:
        logger.warning(
            'main_baseline_failing_ids: %s — baseline probe disabled for this attempt', e,
        )
        return None
    except Exception:
        logger.warning('main_baseline_failing_ids: unexpected error', exc_info=True)
        return None


def cached_main_baseline_failing_ids(main_sha: str) -> 'frozenset[str] | None':
    """Cache-ONLY peek at the per-main-SHA failing-id baseline — never probes.

    Pure, synchronous, side-effect-free: returns the cached id set for
    *main_sha* when present and within :data:`_PROBE_CACHE_TTL`, else
    ``None``.  Used by the synchronous branch-block reason enrichment in
    ``merge_queue._run_post_merge_verify`` (task μ, verify-scope-inversion-
    prd.md), which must NEVER trigger a probe on the critical path (G4, task
    2564) — unlike :func:`main_baseline_failing_ids` (cache-first, THEN
    probes on a miss), this helper only ever reads.
    """
    _cached = _BASELINE_FAILING_IDS_CACHE.get(main_sha)
    if _cached is None:
        return None
    _cached_at, _cached_ids = _cached
    if time.monotonic() - _cached_at >= _PROBE_CACHE_TTL:
        return None
    return _cached_ids


def _worst_category(categories: list[str]) -> str:
    """Return the highest-severity category from *categories*.

    Priority is defined by ``_CATEGORY_PRIORITY``; a category not in the list
    sorts below all listed entries.  Returns ``''`` when *categories* is empty.
    """
    def _rank(cat: str) -> int:
        try:
            return _CATEGORY_PRIORITY.index(cat)
        except ValueError:
            return len(_CATEGORY_PRIORITY)  # unknown → lowest priority

    return min(categories, key=_rank, default='')


def _should_archive_category(category: str) -> bool:
    """Return True when the failure category warrants durable archival.

    Archival means the log is copied to ``data/verify-logs/<task_id>/`` for
    human triage — categories where the debugger can self-correct (compile
    errors, known test failures, timeouts) are excluded.

    Pure delegation to ``verify_categories.should_archive`` — a CATEGORY_POLICY
    table lookup, no ``endswith('_error')`` heuristic.  An unrecognized
    category (e.g. a verify_runner UNSCOPED_TYPECHECK_* sentinel) defaults to
    False rather than auto-archiving on a bare '_error' suffix match.
    """
    return should_archive(category)


def _serial_pytest_str(cmd: str | None) -> str | None:
    """Rewrite every ``pytest`` invocation in *cmd* to run serially, via VerifyCmd.

    Thin string-level wrapper around ``parse_config_command`` ->
    ``serial_pytest`` -> ``render`` (replaces ``_force_serial_pytest``):
    appends `` -p no:xdist -o addopts=`` to a structured PYTEST command's
    flags, or — for a raw-retained ``&&``-chain — to every ``pytest``
    invocation's arguments via ``serial_pytest``'s localised regex rewrite.
    ``-o addopts=`` clears any pyproject-level ``addopts`` (e.g. ``-n auto``)
    — this is the exact ``-o addopts=""`` workaround that task 2045 proved
    recovers a shared-venv-mutation transient, applied structurally rather
    than by gambling on the concurrent ``uv sync`` window having closed.
    ``-p no:xdist`` is belt-and-suspenders: it disables the xdist plugin
    outright and is safe even when xdist is already absent from the venv.

    Returns *cmd* unchanged when it is ``None`` or does not parse/chain into
    a PYTEST ToolKind (e.g. a ``cargo test --workspace`` command — covers
    OPAQUE too, P1).

    Tradeoff: clearing ``addopts`` also drops any per-subproject marker
    filters baked into pyproject (e.g. ``-m 'not integration'``).  Accepted
    for a single bounded recovery run whose only purpose is a
    non-misattributed pass/fail signal — see run_verification's env-recovery
    retry — and unavoidable at the CLI layer since the subproject's addopts
    contents aren't visible to this string rewrite.
    """
    if cmd is None:
        return None
    parsed = parse_config_command(cmd)
    rewritten = serial_pytest(parsed)
    if rewritten is parsed:
        return cmd
    return render(rewritten)


def _with_pytest_timeout_str(cmd: str | None, secs: int) -> str | None:
    """Inject a ``--timeout <secs>`` per-test timeout into every structured
    ``pytest`` invocation in *cmd*, via VerifyCmd.

    Thin string-level wrapper around ``parse_config_command`` ->
    ``with_pytest_timeout`` -> ``render`` (mirrors ``_serial_pytest_str``):
    appends ``--timeout <secs>`` to a structured PYTEST command's ``base_flags``.
    Returns *cmd* unchanged when it is ``None`` or does not parse into a
    structured PYTEST command (an OPAQUE / raw-retained chain / non-pytest
    command — covers the same no-op surface as ``with_pytest_timeout``, P1).

    The α confirm gate composes this OUTSIDE ``_serial_pytest_str`` — the
    generous explicit ``--timeout`` is required because the serial recovery's
    ``-o addopts=`` clears pyproject ``addopts`` but NOT the
    ``[tool.pytest.ini_options] timeout=60`` default, which would otherwise
    starve the isolated confirm run into a false non-suppression.
    """
    if cmd is None:
        return None
    parsed = parse_config_command(cmd)
    rewritten = with_pytest_timeout(parsed, secs)
    if rewritten is parsed:
        return cmd
    return render(rewritten)


def _with_pytest_numprocesses_str(cmd: str | None, n: str) -> str | None:
    """Inject a ``-n <n>`` pytest-xdist worker cap into *cmd*, via VerifyCmd.

    Thin string-level wrapper around ``parse_config_command`` ->
    ``apply_pytest_numprocesses`` -> ``render`` (mirrors
    ``_with_pytest_timeout_str`` / ``_with_junitxml_str``). Returns *cmd*
    unchanged — BYTE-identically, via the ``is`` identity check, since a
    from-scratch render of an untouched parse is only argv-equivalent — when
    it is ``None`` or when the mutation is a no-op.

    All three of ``apply_pytest_numprocesses``' fail-safe guards ride along
    unchanged: a non-PYTEST tool (which covers OPAQUE), *n* in ``{'',
    'auto'}`` (the shipped default, so this is inert until an operator
    configures a numeric cap), and an already-serial-forced command (``-p
    no:xdist`` — injecting ``-n`` there would fail the run with
    ``unrecognized arguments: -n`` and defeat the serial-recovery safety
    net).

    Unlike ``_with_junitxml_str`` this does NOT log its no-op. A suppressed
    junit injection means an expected report will not be written, degrading
    named downstream capabilities; a suppressed ``-n`` just leaves the
    command at its configured worker count, which is the pre-cap status quo
    and not a lost capability.

    Task 3478 extracted this so the cap has ONE rewrite site with two
    callers — ``_run_or_skip_timed``'s non-segmented branch and the
    per-segment application inside ``_run_one_segment``. Two copies of the
    parse/mutate/identity/render dance would be free to drift (INV-5
    no-lockstep-duplication).
    """
    if cmd is None:
        return None
    parsed = parse_config_command(cmd)
    rewritten = apply_pytest_numprocesses(parsed, n)
    if rewritten is parsed:
        return cmd
    return render(rewritten)


def _with_junitxml_str(cmd: str | None, junit_path: str) -> str | None:
    """Inject ``--junitxml <junit_path>`` into a structured ``pytest`` command, via VerifyCmd.

    Thin string-level wrapper around ``parse_config_command`` ->
    ``with_junitxml`` -> ``render`` (mirrors ``_with_pytest_timeout_str``).
    Returns *cmd* unchanged — BYTE-identically, via the ``is`` identity check,
    since a from-scratch render is only argv-equivalent — when it is ``None``
    or does not parse into a structured PYTEST command.

    **Why this logs (task 3218).** The no-op is not always benign. The caller
    only injects when ``role=='merge'`` and ``breadth=='full'``, i.e. exactly
    when a junit report WAS expected: it drives
    ``_extract_failing_test_ids_from_junit``, the α flake-confirmation gate
    and the per-test timeout floor. Silently skipping it there degrades those
    downstream capabilities with no record anywhere — the failure mode the
    reverse-dependency widening already avoids by logging its own no-op
    (see the ``logger.warning`` in ``reverse_dependent_test_targets``'s
    caller). So a suppressed injection on a command that IS pytest is
    reported at INFO, naming the path that will not be written.

    The INFO is gated on ``parsed.tool is ToolKind.PYTEST`` deliberately: a
    lint/type/OPAQUE command was never eligible to produce junit, and logging
    those would fire on legs that were never going to collect a report,
    training operators to ignore the record.

    Siting the log HERE rather than in ``split_chain_tail`` covers every
    cause of the no-op — a raw-retained ``&&``-chain, an OPAQUE command, a
    non-pytest tool — rather than only the tail-preservation one. As of task
    3218 step-2 the tail-preservation cause is no longer reachable through
    the two scopers at all (``'pytest'`` is off ``_TAIL_PRESERVING_KEYWORDS``,
    so a chained pytest slot now comes back structured); this remains as
    defence in depth for a hand-written multi-clause ``test_command`` that
    reaches the injection site directly.

    *junit_path* should be absolute: the rendered command may run with a
    shifted cwd (a structured command's own ``cd <cwd_rel> &&``), so a
    relative value would land in the wrong directory.
    """
    if cmd is None:
        return None
    parsed = parse_config_command(cmd)
    rewritten = with_junitxml(parsed, junit_path)
    if rewritten is parsed:
        if parsed.tool is ToolKind.PYTEST:
            logger.info(
                'junitxml injection suppressed: %s is pytest but not structured '
                '(raw-retained &&-chain or unparseable), so no junit report will be '
                'collected at %s for this run — junit-driven failing-test extraction, '
                'flake confirmation and the per-test timeout floor are unavailable',
                cmd,
                junit_path,
            )
        return cmd
    return render(rewritten)


def _tool_for_cmd(cmd: str | None) -> ToolKind:
    """Resolve *cmd*'s ``ToolKind`` for ``classify_failure`` dispatch (task δ).

    ``None`` (the module doesn't define this check) resolves to
    ``ToolKind.OPAQUE``. In practice this default is never actually consulted
    by ``classify_failure``: every caller checks ``rc == 0`` before
    classifying, and a ``None`` command's check is always skipped (rc stays
    0) — so a failing check always has a real, non-``None`` command string.

    NOTE: this resolution is also what gates env_transient auto-recovery
    (below, ``category == FailureCategory.ENV_TRANSIENT``) — that category is
    only ever produced when a command resolves here to ``ToolKind.PYTEST``
    (see verify_classify.py's "BEHAVIORAL NARROWING" note above
    ``_ENV_TRANSIENT_PATTERNS``). A test command wrapped such that
    ``parse_config_command`` can't see a literal ``pytest`` token (e.g. a
    ``make test`` / shell-script / bare tox-nox indirection) resolves to
    ``ToolKind.OPAQUE`` here and so cannot trigger env_transient recovery,
    even if the underlying tool is in fact pytest. Not a concern for any
    test_cmd shape in production today (all contain a literal ``pytest``
    token).

    The same gating applies unconditionally to the lint/type checks: their
    commands never resolve here to ``ToolKind.PYTEST``, so the pytest-scoped
    shared-venv-mutation signatures can never fire for them — unlike the pre-δ
    tool-blind ladder, which consulted these signatures for every check's
    output (see the env-recovery retry's comment in ``run_verification`` for
    the fuller note on this narrowing).

    That is a statement about THIS pattern source only, NOT about the category
    (task 3367 correction — this NOTE previously over-claimed that a lint/type
    output "can never classify as env_transient either"). ``classify_failure``
    guard 3 (``_classify_environmental``) is tool-blind and has three
    ToolKind-independent ``ENV_TRANSIENT`` producers: task 2756's broken
    ``_merge-verify`` worktree, task 2831's restart collateral, and task 3367's
    mis-resolved pyright interpreter. A TYPE check therefore CAN classify
    ``ENV_TRANSIENT`` today, which is exactly why the env-recovery retry gate
    below now checks ``attempt.test.rc != 0`` explicitly rather than inferring
    the failing leg from the category.
    """
    if not cmd:
        return ToolKind.OPAQUE
    return parse_config_command(cmd).tool


# task 3173: the substring that marks a summary fragment as a signal-kill
# note.  Shared by `_killed_leg_note` (producer) and `_aggregate_results`
# (consumer), so multi-module aggregation — which rebuilds the summary by
# substring-scanning child summaries for 'tests failed'/'lint issues'/
# 'type errors' — cannot silently drop the one fact that says the run means
# nothing.  Any new consumer should match on this constant, never a literal.
#
# CONSTRAINT ON PRODUCERS: a marker-bearing fragment must not contain ', '.
# `_summarize_checks` JOINS fragments with ', ' and `_aggregate_results`
# recovers them with `.split(', ')`, so a comma inside the note splits it in
# two and only the marker-bearing half survives — silently truncating the
# sentence, which is the exact degradation the carry-through exists to
# prevent.  Separate clauses with '; '.  Pinned by
# test_verify.py::TestKillNoteIsOneAggregationFragment, which fails at the
# producer rather than letting the consumer degrade quietly.
SIGNAL_KILL_SUMMARY_MARKER = 'killed by signal'


def _killed_leg_note(label: str, rc: int, duration: float | None) -> str:
    """Describe a leg that was terminated by an external signal.

    Every clause is a MEASURED fact: which leg, which signal delivered it,
    how long the process survived, and — the point of the whole exercise —
    that no verdict exists.  ``duration`` is ``None`` (not ``0.0``) when the
    caller has no timing in scope, in which case the "after N.NNs" clause is
    omitted rather than fabricating "after 0.00s".

    ``rc`` is asyncio's returncode, i.e. the NEGATIVE signal number, so the
    signal is ``-rc``.

    Clauses are separated by ``'; '`` and the sentence must stay free of
    ``', '`` — see ``SIGNAL_KILL_SUMMARY_MARKER``'s producer constraint: the
    ``', '``-joined summary is the wire format between this function and
    ``_aggregate_results``, and a comma here would silently truncate the note
    on the way through aggregation.
    """
    after = f' after {duration:.2f}s' if duration is not None else ''
    return (
        f'{label} leg {SIGNAL_KILL_SUMMARY_MARKER} {-rc}{after}; '
        f'no diagnostics produced; verdict indeterminate'
    )


def _summarize_checks(
    test_rc: int, test_out: str, test_timed_out: bool, test_cmd: str | None,
    lint_rc: int, lint_out: str, lint_timed_out: bool, lint_cmd: str | None,
    type_rc: int, type_out: str, type_timed_out: bool, type_cmd: str | None,
    *,
    test_duration: float | None = None,
    lint_duration: float | None = None,
    type_duration: float | None = None,
) -> tuple[bool, str, str, str, list[str]]:
    """Classify the three checks into (passed, category, cause_hint, summary,
    failing_leg_categories).

    Shared by ``run_verification``'s primary post-loop classification and its
    bounded env-recovery retry (task 2048) so the failure-reclassification
    logic — worst-category selection via ``_worst_category`` plus cause-hint
    and summary-parts assembly — lives in exactly one place instead of being
    duplicated per call site.

    Each check's config command (``test_cmd``/``lint_cmd``/``type_cmd`` — the
    ungoverned, un-scoped command string, already in scope at both
    ``run_verification`` call sites) is resolved to a ``ToolKind`` via
    ``_tool_for_cmd`` and threaded into ``classify_failure`` (PRD task δ,
    Invariant C1): a tool-T pattern can only ever match tool-T output, so a
    cargo token embedded in pytest's own captured output (or vice versa) can
    no longer swallow the wrong check's failure line.

    Does NOT compute the ``timed_out`` bookkeeping flag (pure-timeout-retry
    eligibility / consistency) — that stays with the caller, which alone
    knows whether this is the first pass (loop-bounded by ``max_retries``) or
    the env-recovery retry, and overrides the returned ``summary`` with
    timeout-specific text when its own ``timed_out`` is True.

    CONTRACT (task 3173): **the summary may never assert a property the gate
    did not measure.**  ``parts`` used to be derived purely from ``rc != 0``,
    so a leg SIGKILLed before it could emit a single diagnostic was reported
    as "lint issues" — and ``merge_queue`` surfaced that verbatim as
    ``Post-merge verification failed: Failures: lint issues``, blaming the
    branch for a process it never got to influence (the measured incident;
    same defect shape as task 3110).  A leg whose rc is an EXTERNAL
    termination signal now contributes ``_killed_leg_note`` instead.
    Non-killed legs keep byte-identical wording, and the
    ``f'Failures: {...}'`` envelope is preserved so every existing consumer
    that prefix- or substring-matches on it stays green.

    CONTRACT (task 3173 review amendment): the returned ``category`` and
    ``failing_leg_categories`` answer DIFFERENT questions and neither
    substitutes for the other.  ``category`` is the severity-ranked worst leg
    (``_worst_category``) — "how bad was this run", which the retry loop, the
    archive and the transient-infra hold consume.  ``failing_leg_categories``
    is what EACH failing leg actually decided, in test/lint/type order.
    ``merge_queue``'s per-land veto gate needs the second and must never infer
    it from the first: a rank-1 ``INFRA_KILL`` DOMINATES a co-occurring
    rank-11 ``TEST_FAILURE``, so a run whose test leg completed and blamed the
    branch reports ``category == 'infra_kill'``, and reading that single value
    as "this run produced no verdict" would discard the completed evidence.
    The list is empty on the passing short-circuit — no failing legs, which is
    not the same claim as "every failing leg was verdict-less".

    ``test_duration``/``lint_duration``/``type_duration`` are keyword-only and
    default to ``None``, so the twelve positional parameters are untouched and
    every pre-existing caller is byte-identical (see the design decision on
    not refactoring to take ``CheckRun`` objects).  Both production call sites
    pass ``attempt.<leg>.duration_secs``, which they already hold.
    """
    passed = test_rc == 0 and lint_rc == 0 and type_rc == 0
    if passed:
        return True, 'passed', '', 'All checks passed', []

    hint_parts = []
    per_check_categories: list[str] = []
    for rc, out, to, cmd in (
        (test_rc, test_out, test_timed_out, test_cmd),
        (lint_rc, lint_out, lint_timed_out, lint_cmd),
        (type_rc, type_out, type_timed_out, type_cmd),
    ):
        if rc != 0:
            h = _extract_cause_hint(out)
            if h:
                hint_parts.append(h)
            per_check_categories.append(classify_failure(_tool_for_cmd(cmd), rc, out, to))
    cause_hint = ' | '.join(hint_parts)
    category = _worst_category(per_check_categories) if per_check_categories else 'unknown_test_failure'

    parts = []
    for rc, label, tool_verdict, duration in (
        (test_rc, 'test', 'tests failed', test_duration),
        (lint_rc, 'lint', 'lint issues', lint_duration),
        (type_rc, 'type', 'type errors', type_duration),
    ):
        if rc == 0:
            continue
        # An externally killed leg produced no verdict, so it gets a note
        # saying exactly that instead of a fabricated tool verdict. Crash
        # signals (SIGSEGV/SIGABRT/...) are NOT external kills — they are
        # genuine faults of the code under test and keep today's wording.
        if is_external_kill_rc(rc):
            parts.append(_killed_leg_note(label, rc, duration))
        else:
            parts.append(tool_verdict)
    summary = f'Failures: {", ".join(parts)}'
    return passed, category, cause_hint, summary, per_check_categories


def _build_summary_payload(runs: list[dict], category: str, cause_hint: str) -> dict:
    """Build the summary.json payload dict from a list of run dicts.

    Extracted from ``_persist_attempt_logs`` so both the task-path summary
    (written into ``<worktree>/.task/verify/``) and the merge-path summary
    (written directly to the durable archive) share an identical shape.

    Top-level rc/cmd/timed_out/started_at/duration_secs fields come from the
    run with the highest numeric rc (timed_out used as a tiebreaker).  This
    intentionally differs from the 'category' field, which uses _worst_category
    priority semantics.  The rationale: rc is the most unambiguous exit-code
    signal for the outermost process, while category conveys semantic severity
    across tools that may use different rc scales.  Downstream readers should
    treat top-level metadata as "the loudest raw exit code" and 'category' as
    "the highest-severity classification".

    Each ``commands`` entry also carries ``segments`` (task 3338): the
    per-segment execution facts when that check ran as a SEGMENTED `&&` chain,
    ``None`` when it did not.  ``.get`` rather than ``[]`` because this helper
    also takes hand-built run dicts (the remote merge-verify path in
    ``verify_runner``), which predate the key.  Without it the entry rebuild
    below — an explicit key whitelist, not a passthrough — silently DROPPED
    the one structured record of which segments never ran, leaving those facts
    only as free text inside the aggregated ``output`` blob.

    A NEGATIVE rc is not a quiet outcome — it is asyncio reporting that the
    process was terminated by signal ``-rc`` and never got to exit at all, so
    it is the LOUDEST possible outcome and sorts above every non-negative rc
    (task 3173).  Under the plain numeric ordering a -9 sorted below even a
    passing 0, so a killed leg co-occurring with a passing leg made this
    payload name the PASSING run — hiding the kill in the one artifact that
    survives for triage.  Ordering among non-negative rcs is unchanged.
    """
    active_runs = [r for r in runs if r.get('cmd') is not None]
    if active_runs:
        worst = max(active_runs, key=lambda r: (r['rc'] < 0, r['rc'], r['timed_out']))
    else:
        worst = {'rc': 0, 'timed_out': False, 'cmd': None,
                 'started_at': '', 'duration_secs': 0.0}

    return {
        'category': category,
        'cause_hint': cause_hint,
        'rc': worst['rc'],
        'timed_out': worst['timed_out'],
        'cmd': worst['cmd'],
        'started_at': worst['started_at'],
        'duration_secs': worst['duration_secs'],
        'commands': [
            {
                'label': r['label'],
                'cmd': r['cmd'],
                'rc': r['rc'],
                'timed_out': r['timed_out'],
                'started_at': r['started_at'],
                'duration_secs': r['duration_secs'],
                'segments': r.get('segments'),
            }
            for r in active_runs
        ],
    }


def _make_infix(module_prefix: 'str | None') -> str:
    """Return the dot-prefixed filename infix for *module_prefix*, or '' if None.

    Sanitizes by replacing ``/`` and spaces with ``_``, mirroring
    :func:`_warm_marker_name`.

    Examples::

        _make_infix('src/my module') -> '.src_my_module'
        _make_infix(None)            -> ''
    """
    if module_prefix is None:
        return ''
    safe = module_prefix.replace('/', '_').replace(' ', '_')
    return f'.{safe}'


def _prepare_junit_report_path(
    worktree: Path, module_prefix: 'str | None',
) -> 'Path | None':
    """Build the merge-verify junit report path under *worktree*, or None.

    Returns the absolute ``<worktree>/.df-verify-junit/report{infix}.xml`` path
    (creating ONLY the ``.df-verify-junit`` child dir) for a live worktree, or
    None WITHOUT creating anything when the worktree is gone or unwritable.

    Shape-2 husk guard (task 2922): a late merge-role verify writer can fire
    AFTER its worktree was torn down. The previous inline
    ``mkdir(parents=True, exist_ok=True)`` would re-create the entire
    torn-down ``<worktree>`` path as an empty husk that the merge-worktree
    ledger audit then flags as an unregistered ``_merge-*`` directory. Two
    guards prevent that: an explicit :meth:`~pathlib.Path.is_dir` check
    returns None without touching the filesystem, and ``mkdir(exist_ok=True)``
    (NOTE: no ``parents=True``) so a missing worktree ancestor raises
    ``FileNotFoundError`` — a subclass of ``OSError`` — routed to the same
    None return (the already-tolerated B3 "no junit collected" late-write
    degrade).

    Reuses :func:`_make_infix` for the per-module sanitized filename infix
    (``pkg/sub`` -> ``.pkg_sub``) so a per-module fan-out never collides.
    """
    if not worktree.is_dir():
        return None
    junit_dir = worktree / '.df-verify-junit'
    try:
        junit_dir.mkdir(exist_ok=True)
    except OSError:
        return None
    return (junit_dir / f'report{_make_infix(module_prefix)}.xml').resolve()


def _write_run_log(
    target_dir: Path,
    attempt_id: int,
    infix: str,
    run: dict,
    *,
    ts_suffix: str = '',
    skip_if_exists: bool = False,
    caller: str = '_write_run_log',
) -> 'Path | None':
    """Write a single run's output to a log file; return the path or None on error.

    Filename: ``attempt-{attempt_id}{infix}.{run['label']}{ts_suffix}.log``

    Parameters
    ----------
    target_dir:
        Directory to write into (must already exist).
    attempt_id:
        Attempt counter embedded in the filename stem.
    infix:
        Module-prefix infix (including leading ``.``), or ``''``.  Build via
        :func:`_make_infix`.
    run:
        Run dict with at least ``label`` (str) and optionally ``output`` (str).
    ts_suffix:
        Timestamp suffix to append before ``.log`` (e.g. ``'-20260615T120000_000000Z'``).
        Empty string on the task path (no timestamp); non-empty on the merge path.
    skip_if_exists:
        When True and the target path already exists, return the path without
        rewriting.  Used on the task path where ``_run_cmd`` may have already
        streamed output there.
    caller:
        Name embedded in warning log messages for attribution.

    Returns ``None`` on OSError (logged at warning level).
    """
    log_path = target_dir / f'attempt-{attempt_id}{infix}.{run["label"]}{ts_suffix}.log'
    if skip_if_exists and log_path.exists():
        return log_path
    try:
        log_path.write_text(run.get('output', ''), encoding='utf-8')
        return log_path
    except OSError as exc:
        logger.warning('%s: could not write %s: %s', caller, log_path, exc)
        return None


def _persist_attempt_logs(
    worktree: Path,
    attempt_id: int,
    runs: list[dict],
    category: str,
    cause_hint: str,
    *,
    module_prefix: 'str | None' = None,
) -> list[Path]:
    """Write per-command outputs and a summary JSON to ``<worktree>/.task/verify/``.

    Each *run* dict must have:
        ``label``        — "test", "lint", or "type"
        ``cmd``          — shell command string or ``None`` (skipped check)
        ``rc``           — return code (int)
        ``output``       — combined stdout+stderr (str)
        ``timed_out``    — bool
        ``started_at``   — ISO timestamp string
        ``duration_secs``— elapsed seconds (float)

    No-op (returns ``[]``) when ``(worktree / '.task')`` is absent — review-
    checkpoint and merge-queue paths lack ``.task/`` and must not be created.

    When *module_prefix* is provided it is sanitized (``/`` and spaces →
    ``_``, mirroring :func:`_warm_marker_name`) and inserted as a middle infix:
    ``attempt-{N}.{safe_prefix}.{label}.log`` and
    ``attempt-{N}.{safe_prefix}.summary.json``.  This prevents last-writer-wins
    clobber when ``run_scoped_verification`` gathers multiple concurrent
    :func:`run_verification` calls for different sub-projects into the same
    worktree + attempt_id.

    When *module_prefix* is ``None`` the filenames remain
    ``attempt-{N}.{label}.log`` / ``attempt-{N}.summary.json`` so the single-
    module path is byte-identical to the pre-prefix behaviour.

    Writes:
    - ``attempt-{N}[.{safe_prefix}].{label}.log`` for every run where ``cmd is not None``
    - ``attempt-{N}[.{safe_prefix}].summary.json`` with the summary shape described in the
      task description: top-level keys are from the worst-failing run plus a
      ``commands`` list containing all per-run sub-dicts.

    Returns the list of log paths actually written (summary.json excluded
    so callers can pass the list straight to ``_archive_attempt_log``).
    """
    task_dir = worktree / '.task'
    if not task_dir.is_dir():
        return []

    verify_dir = task_dir / 'verify'
    try:
        verify_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning('_persist_attempt_logs: could not create %s: %s', verify_dir, exc)
        return []

    infix = _make_infix(module_prefix)
    written: list[Path] = []

    # Write per-command log files.  When the file already exists on disk it
    # was streamed there by ``_run_cmd`` (Change 2 — streaming variant): skip
    # rewriting to avoid clobbering streamed-but-truncated output on a kill,
    # but still record the path so summary.json and downstream archival see it.
    for run in runs:
        if run.get('cmd') is None:
            continue
        # No ts_suffix on the task path; skip_if_exists handles streamed files.
        path = _write_run_log(
            verify_dir, attempt_id, infix, run,
            skip_if_exists=True, caller='_persist_attempt_logs',
        )
        if path is not None:
            written.append(path)

    # Build summary.json via the shared helper (same shape as merge-path summary).
    summary_payload = _build_summary_payload(runs, category, cause_hint)

    summary_path = verify_dir / f'attempt-{attempt_id}{infix}.summary.json'
    try:
        summary_path.write_text(
            json.dumps(summary_payload, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )
    except OSError as exc:
        logger.warning('_persist_attempt_logs: could not write %s: %s', summary_path, exc)

    return written


def _archive_attempt_log(
    worktree_log_paths: list[Path],
    archive_root: 'Path | None',
    task_id: str,
    attempt_id: int,
    category: str,
) -> list[Path]:
    """Copy worktree logs to the durable archive when ``category`` warrants it.

    Archive target: ``<archive_root>/<task_id>/attempt-{N}-<utc_ts>.log``.

    Early-returns ``[]`` when:
    - ``archive_root`` is ``None``
    - ``_should_archive_category(category)`` is ``False``

    All filesystem errors are caught, logged, and swallowed (best-effort).

    .. note::
        ``_prune_archive`` is intentionally NOT called here.  The caller
        (``run_scoped_verification``) calls it exactly once after all modules
        have been gathered, preventing concurrent per-module prune walks from
        racing on the same ``archive_root`` directory tree.
    """
    if archive_root is None:
        return []
    if not _should_archive_category(category):
        return []

    target_dir = archive_root / task_id
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning('_archive_attempt_log: could not create %s: %s', target_dir, exc)
        return []

    utc_ts = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
    archived: list[Path] = []
    for src in worktree_log_paths:
        src = Path(src)
        # Preserve the source stem to avoid collisions when multiple log files
        # share the same suffix (e.g. attempt-1.test.log and attempt-1.lint.log
        # would both resolve to attempt-1-TS.log without the stem).
        dest = target_dir / f'{src.stem}-{utc_ts}{src.suffix}'
        try:
            shutil.copy2(src, dest)
            archived.append(dest)
        except OSError as exc:
            logger.warning('_archive_attempt_log: could not copy %s → %s: %s', src, dest, exc)

    return archived


def _archive_merge_verify_logs(
    runs: list[dict],
    archive_root: 'Path | None',
    task_id: str,
    attempt_id: int,
    category: str,
    cause_hint: str,
    *,
    module_prefix: 'str | None' = None,
) -> list[Path]:
    """Write merge-verify run outputs + summary DIRECTLY to the durable archive.

    Unlike ``_archive_attempt_log`` (which copies files from the worktree's
    ``.task/verify/``), this function writes directly to
    ``<archive_root>/<task_id>/`` because merge worktrees have ``.task/``
    scrubbed by design (git_ops.py) and there are no intermediate worktree
    log files to copy from.

    Differences from the task-path helpers:
    - **No deny-list check**: on the merge path there is no debugger loop;
      every failure reaches a human.  ``infra_timeout`` and ``test_failure``
      (the exact categories that distinguish timeout-vs-real-failure) are
      archived unconditionally.
    - **Direct-to-archive**: bypasses ``.task/verify/``; the archive is the
      only persistence target that survives merge_wt cleanup.

    Filename convention mirrors ``_archive_attempt_log``:
        ``attempt-{N}[.{safe_prefix}].{label}-{utc_ts}.log``
        ``attempt-{N}[.{safe_prefix}].summary-{utc_ts}.json``

    Returns the list of paths actually written (both .log and .json).
    Returns ``[]`` when ``archive_root`` is ``None``.
    All filesystem errors are caught, logged, and swallowed (best-effort).
    """
    if archive_root is None:
        return []

    target_dir = archive_root / task_id
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning('_archive_merge_verify_logs: could not create %s: %s', target_dir, exc)
        return []

    infix = _make_infix(module_prefix)
    # Use microsecond precision so rapid back-to-back merge-verify retries for
    # the same task (same attempt_id=1 default, same second) never overwrite each
    # other.  The format is still lexicographically sortable.
    utc_ts = datetime.now(UTC).strftime('%Y%m%dT%H%M%S_%fZ')
    ts_suffix = f'-{utc_ts}'
    archived: list[Path] = []

    # Write per-command log files directly to the archive.
    for run in runs:
        if run.get('cmd') is None:
            continue
        path = _write_run_log(
            target_dir, attempt_id, infix, run,
            ts_suffix=ts_suffix, caller='_archive_merge_verify_logs',
        )
        if path is not None:
            archived.append(path)

    # Write summary.json using the shared payload builder.
    summary_path = target_dir / f'attempt-{attempt_id}{infix}.summary-{utc_ts}.json'
    try:
        summary_payload = _build_summary_payload(runs, category, cause_hint)
        summary_path.write_text(
            json.dumps(summary_payload, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )
        archived.append(summary_path)
    except OSError as exc:
        logger.warning(
            '_archive_merge_verify_logs: could not write %s: %s', summary_path, exc,
        )

    return archived


_DEFAULT_ARCHIVE_MAX_AGE_DAYS = 30
_DEFAULT_ARCHIVE_MAX_BYTES = 500 * 1024 * 1024
# Process-local throttle: at most one rglob walk per process per 30 min.
# Cross-process redundancy is accepted as a cost-only trade-off; see task 1102.
# Thread-safety: _maybe_prune_archive has no awaits, so concurrent async coroutines
# cannot interleave the check + update. This is NOT safe for multi-threaded callers
# (e.g. asyncio.to_thread) without a threading.Lock — add one if that ever changes.
_PRUNE_THROTTLE_SECS: float = 1800  # 30 minutes
_LAST_PRUNE_AT: float | None = None
# Module-level reference capture for test injection: patch `verify._monotonic`
# instead of `verify.time.monotonic` (which would mutate the stdlib globally).
_monotonic = time.monotonic


def _prune_archive(
    archive_root: Path,
    max_age_days: int = _DEFAULT_ARCHIVE_MAX_AGE_DAYS,
    max_total_bytes: int = _DEFAULT_ARCHIVE_MAX_BYTES,
) -> None:
    """Enforce age + size retention on ``archive_root``.

    Two-pass strategy:
    1. Delete files older than ``max_age_days`` (by mtime).
    2. If aggregate size still exceeds ``max_total_bytes``, delete oldest-first
       until under cap.

    Best-effort: per-file errors are logged and swallowed. Outer FS errors (e.g.
    archive_root.exists() or rglob walk) may raise OSError — callers wishing to
    ignore those should wrap the call (see _maybe_prune_archive).
    """
    if not archive_root.exists():
        return

    # module-level `import time` is sufficient; no local import needed.
    now = time.time()
    cutoff = now - max_age_days * 86_400

    # Single rglob walk — collect all archivable files once, avoiding a second
    # directory scan for the size-cap pass.  Both *.log and *.json are counted
    # because _archive_merge_verify_logs emits summary.json files into the same
    # tree and they would otherwise accumulate unbounded (never counted toward the
    # size budget, never pruned).
    _PRUNE_SUFFIXES = frozenset(('.log', '.json'))
    all_entries: list[tuple[Path, float, int]] = []
    for path in archive_root.rglob('*'):
        if path.suffix not in _PRUNE_SUFFIXES:
            continue
        try:
            st = path.stat()
            if not path.is_file():
                continue
            all_entries.append((path, st.st_mtime, st.st_size))
        except OSError:
            continue

    # Pass 1: age-based deletion; collect survivors for the size-cap pass.
    survivors: list[tuple[Path, float, int]] = []
    for path, mtime, size in all_entries:
        if mtime < cutoff:
            try:
                path.unlink()
            except OSError as exc:
                logger.warning('_prune_archive: could not delete %s: %s', path, exc)
        else:
            survivors.append((path, mtime, size))

    # Pass 2: size cap on survivors (no second rglob needed).
    total = sum(sz for _, _, sz in survivors)
    if total > max_total_bytes:
        # Sort oldest-first.
        survivors.sort(key=lambda t: t[1])
        for path, _mtime, size in survivors:
            if total <= max_total_bytes:
                break
            try:
                path.unlink()
                total -= size
            except OSError as exc:
                logger.warning('_prune_archive: could not delete %s: %s', path, exc)


def _maybe_prune_archive(archive_root: Path | None) -> bool:
    """Thin wrapper around ``_prune_archive`` with None-guard and time throttle.

    Returns True if ``_prune_archive`` was invoked, False otherwise.

    - ``archive_root=None`` short-circuits immediately without updating the
      throttle timestamp (preserves semantics: None means no archival/pruning).
    - First call in a process always fires (``_LAST_PRUNE_AT is None``).
    - Subsequent calls within ``_PRUNE_THROTTLE_SECS`` are skipped.
    - After the window elapses, the next call fires and slides the window forward.
    - If ``_prune_archive`` raises OSError (e.g. ``archive_root.exists()`` or
      ``rglob`` fails on a permission-broken FS), the error is logged at warning
      level and ``_LAST_PRUNE_AT`` still advances — preventing the same exception
      from being raised on every subsequent verification call within the throttle
      window.  Non-OSError exceptions still propagate.
    """
    global _LAST_PRUNE_AT
    if archive_root is None:
        return False
    now = _monotonic()
    if _LAST_PRUNE_AT is not None and now - _LAST_PRUNE_AT < _PRUNE_THROTTLE_SECS:
        logger.debug(
            'skipping prune: %.0fs since last (throttle %ds)',
            now - _LAST_PRUNE_AT,
            _PRUNE_THROTTLE_SECS,
        )
        return False
    try:
        _prune_archive(archive_root)
    except OSError as exc:
        # Deliberate OSError-only scope: _prune_archive uses path.unlink() (not
        # shutil.rmtree), so shutil.Error cannot arise.  PermissionError /
        # FileNotFoundError are OSError subclasses on Python ≥ 3.3.  Programming
        # bugs (RuntimeError, AttributeError, etc.) still propagate uncaught.
        logger.warning(
            '_maybe_prune_archive: prune raised %s; advancing throttle to suppress retry storm',
            exc,
        )
    _LAST_PRUNE_AT = now
    return True


async def _derive_task_files_from_git(
    worktree: Path, config: OrchestratorConfig,
) -> list[str] | None:
    """Derive task file list from ``git diff main...HEAD`` in the worktree.

    Returns ``None`` when:
    - the worktree is on main (no diff to derive)
    - ``git diff`` fails for any reason
    - no files changed
    """
    from orchestrator.git_ops import _run
    try:
        rc, main_sha, _ = await _run(
            ['git', 'rev-parse', '--verify', config.git.main_branch],
            cwd=worktree,
        )
        if rc != 0:
            return None
        rc, head_sha, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=worktree,
        )
        if rc != 0 or head_sha == main_sha:
            return None
        rc, output, _ = await _run(
            ['git', 'diff', '--name-only',
             f'{config.git.main_branch}...HEAD'],
            cwd=worktree,
        )
        if rc != 0:
            return None
        files = [f for f in output.strip().splitlines() if f.strip()]
        if files:
            logger.info('Derived %d task files from git diff', len(files))
            return files
    except Exception:
        logger.debug(
            'Failed to derive task files from git diff', exc_info=True,
        )
    return None


def _apply_cargo_scope(
    mc: ModuleConfig,
    task_files: list[str],
    project_root: Path,
    scope_cargo_enabled: bool,
) -> ModuleConfig:
    """Return *mc* with cargo ``--workspace`` rewritten to touched crates.

    Guard conditions — returns *mc* unchanged when any fail:
    - ``scope_cargo_enabled`` is False, or ``mc.scope_cargo`` is explicitly False
    - *task_files* is empty
    - *task_files* contains no ``.rs`` files (no Rust source touched)
    - any non-``.rs`` file has an extension outside the safe config/data whitelist
      (``.toml``, ``.yaml``, ``.yml``, ``.json``, ``.md``) AND its basename is not
      in the filename allowlist (``Cargo.lock``, ``rust-toolchain``); this prevents under-protecting
      the non-Rust side of polyglot tasks with chained commands such as
      ``cargo test --workspace && uv run pytest``
    - the workspace has no discoverable crates
    - ``files_to_crates`` returns ``None`` (a file lives outside all crates)
    - the rewritten commands are byte-identical to the originals
    """
    if not scope_cargo_enabled:
        return mc
    if mc.scope_cargo is False:
        return mc
    if not task_files:
        return mc
    # Filter to .rs files for crate mapping — only Rust files determine which
    # crates need testing.
    rs_files = [f for f in task_files if f.endswith('.rs')]
    if not rs_files:
        return mc
    # Polyglot guard: if any non-.rs file has an extension outside the safe
    # config/data whitelist, bail to --workspace.  This protects chained
    # non-Rust commands (e.g. ``cargo test --workspace && uv run pytest``)
    # from being silently skipped when only some crates are scoped.
    non_rs = [f for f in task_files if not f.endswith('.rs')]
    for f in non_rs:
        p = Path(f)
        if (
            p.suffix.lower() not in _CARGO_SCOPE_SAFE_NON_RS_EXTS
            and p.name not in _CARGO_SCOPE_SAFE_NON_RS_NAMES
        ):
            return mc

    crates_map = discover_workspace_crates(project_root)
    if not crates_map:
        return mc
    matched = files_to_crates(rs_files, crates_map)
    if not matched:
        return mc

    new_test = _cargo_scope_str(mc.test_command, matched)
    new_lint = _cargo_scope_str(mc.lint_command, matched)
    new_type = _cargo_scope_str(mc.type_check_command, matched)

    if (new_test, new_lint, new_type) == (
        mc.test_command, mc.lint_command, mc.type_check_command,
    ):
        return mc  # nothing to rewrite — original didn't contain --workspace

    for label, old, new in (
        ('test', mc.test_command, new_test),
        ('lint', mc.lint_command, new_lint),
        ('type', mc.type_check_command, new_type),
    ):
        if old != new:
            logger.info('cargo scope (%s): %r -> %r', label, old, new)

    return ModuleConfig(
        prefix=mc.prefix,
        test_command=new_test,
        lint_command=new_lint,
        type_check_command=new_type,
        lock_depth=mc.lock_depth,
        max_per_module=mc.max_per_module,
        module_overrides=mc.module_overrides,
        verify_command_timeout_secs=mc.verify_command_timeout_secs,
        verify_cold_command_timeout_secs=mc.verify_cold_command_timeout_secs,
        concurrent_verify=mc.concurrent_verify,
        verify_env=mc.verify_env,
        scope_cargo=mc.scope_cargo,
    )


# Source-file extensions that warrant running verify at all. Markdown / YAML /
# TOML / JSON diffs are inert: every existing scoping branch
# (scope_module_config filters .py, _build_fallback_config filters .py,
# _apply_cargo_scope filters .rs) would no-op on them anyway, so the
# previous global-pytest fall-through was the only path that actually fired
# — and it was unsafe (see plans/fix-all-root-causes-humble-dream.md R1/R2).
_SOURCE_EXTENSIONS: frozenset[str] = frozenset({'.py', '.rs'})


def _has_source_files(task_files: list[str]) -> bool:
    """Return True when *task_files* contains at least one .py or .rs path."""
    exts = tuple(_SOURCE_EXTENSIONS)
    return any(f.endswith(exts) for f in task_files)


def _trivial_pass(reason: str) -> 'VerifyResult':
    """Build a VerifyResult that represents 'verify trivially passed'.

    Forward-reference string annotation: ``VerifyResult`` is defined further
    down in this module (no ``from __future__ import annotations`` here).
    """
    return VerifyResult(
        passed=True,
        summary=reason,
        test_output='',
        lint_output='',
        type_output='',
        timed_out=False,
        cause_hint='',
        trivial=True,
    )


#: INV-1 (task 2883) loud-FAIL category — greppable and, crucially, carrying NO
#: parseable pytest node-ids, so the merge-flake suppression gate (which keys on
#: node-ids) can never suppress a no-evidence merge FAIL.
_MERGE_NO_EVIDENCE_CATEGORY = 'merge_no_evidence'


def _trivial_pass_reason(existing_files: list[str]) -> str:
    """INV-1 (task 2883) escalation reason for a no-source merge-gate resolution.

    Single source of truth so Site 1 (module_configs branch) and Site 2
    (no-module_configs branch) label the escalation IDENTICALLY:
    ``'empty_existing_files'`` when the changed set resolved to nothing on disk
    (e.g. an ENOENT-clobbered worktree, incident 83336a32) — distinguishing
    evidence-absence from a genuine ``'no_source_files'`` docs-only diff.
    (The global-tail backstop uses its own fixed ``'empty_command_set'`` reason.)
    """
    return 'empty_existing_files' if not existing_files else 'no_source_files'


def _merge_no_evidence_fail(reason: str) -> 'VerifyResult':
    """Build the INV-1 loud-FAIL VerifyResult for a merge gate that resolved to
    'nothing to run' with no full-gate command to escalate to (task 2883).

    Mirrors :func:`_trivial_pass`'s construction with the passed/trivial flags
    INVERTED: ``passed=False`` so the merge worker treats it as blocked,
    ``trivial=False`` so the task-2823 trivial-pass main-red gate never mistakes
    it for a config-only short-circuit, and a distinct
    ``category='merge_no_evidence'`` that carries no pytest node-ids so the
    merge-flake suppression gate cannot suppress it.  *reason* ∈
    {no_source_files, empty_existing_files, empty_command_set}.
    """
    summary = (
        f'Merge gate produced no evidence ({reason}): the adoptable merge '
        f'verdict resolved to "nothing to run" with no full-gate command — '
        f'failing loud per INV-1 rather than trivially passing.'
    )
    return VerifyResult(
        passed=False,
        summary=summary,
        test_output='',
        lint_output='',
        type_output=summary,
        timed_out=False,
        cause_hint=f'{_MERGE_NO_EVIDENCE_CATEGORY}:{reason}',
        category=_MERGE_NO_EVIDENCE_CATEGORY,
        trivial=False,
    )


async def _verify_pipeline_guard_requires_full_gate(
    worktree: Path,
    changed_files: list[str],
) -> bool:
    """Return True iff reify's verify-pipeline-guard.sh says the full gate is required.

    Shells out to ``<worktree>/scripts/verify-pipeline-guard.sh requires-full-gate
    <changed_files...>`` and returns ``True`` when the script exits 0 (conventional
    Unix predicate: exit 0 ⟹ condition is true ⟹ full gate is required).

    Fail-open contract — returns False for ANY of:
    - ``scripts/verify-pipeline-guard.sh`` absent in the worktree (backward-compat:
      dark-factory's own merges, pre-4626 reify, non-reify projects).
    - ``changed_files`` is empty.
    - Script exits non-zero (guard says fast-path is safe).
    - Script non-executable, spawn fails, WorktreeMissing, or any other exception
      (guard hiccup must never wedge the merge pipeline → log WARNING, return False).

    Mirrors GitOps._provision_reify_debug_port (the canonical cross-repo seam).
    """
    try:
        script = worktree / 'scripts' / 'verify-pipeline-guard.sh'
        if not script.exists() or not changed_files:
            return False
        from orchestrator.git_ops import _run  # noqa: PLC0415, I001 — lazy, mirrors verify_failure_is_preexisting_on_main
        rc, _out, _err = await _run(
            [str(script), 'requires-full-gate', *changed_files],
            cwd=worktree,
        )
        return rc == 0
    except Exception:
        logger.warning(
            '_verify_pipeline_guard_requires_full_gate: unexpected error for %s',
            worktree, exc_info=True,
        )
        return False


def _merge_config_only_diff_forces_full_gate(
    config: OrchestratorConfig,
    changed_files: list[str],
) -> bool:
    """Return True iff a config-only diff touches a manifest-relevant glob.

    The deterministic dark-factory-side manifest-drift backstop (task 2838),
    the local complement to the async remote
    ``_verify_pipeline_guard_requires_full_gate`` consult above.  A merge-role
    config-only (no .py/.rs) diff whose files match any configured glob forces
    the full per-subproject verify gate EVEN WHEN the reify
    verify-pipeline-guard.sh consult falls open — closing the fail-open
    residual that let a config-only diff CAS-advance a new manifest-drift RED
    onto main (incident deb-reify-964887, tasks 5247/5249).

    Pure and synchronous (no subprocess, no git), so it cannot wedge the merge
    pipeline and is trivially unit-testable.  Uses ``fnmatch.fnmatchcase`` —
    case-sensitive and OS-independent (unlike ``fnmatch.fnmatch``, which
    normalizes case per-OS); shell-glob ``*`` crosses ``/``, an intentional,
    safe over-approximation (over-forcing the full gate is the safe direction,
    under-forcing is the bug being closed).  The call sites pass the FULL
    changed set (``task_files``) — added, modified, AND deleted paths — a safe
    over-approximation of "adds files that shift the manifest": a modification
    to a classification/manifest file shifts it too, and so does a DELETION of
    a file the manifest enumerates (deleted paths are absent from the on-disk
    ``existing_files`` the reify consult receives, so matching ``task_files``
    here is what closes the deletion residual — reviewer amendment, task 2838).

    Empty globs (the default,
    ``config.git.merge_config_only_full_gate_globs``) short-circuit to False in
    O(1), leaving the config-only fast-path byte-identical for dark-factory's
    own merges and non-reify projects.
    """
    globs = config.git.merge_config_only_full_gate_globs
    if not globs:
        return False
    return any(
        fnmatch.fnmatchcase(f, g) for f in changed_files for g in globs
    )


def _single_subproject_prefix(files: list[str], worktree: Path | None) -> str | None:
    """Return the sole top-level subproject directory shared by *files*, else ``None``.

    A "subproject" is a top-level directory of *worktree* that carries its own
    ``pyproject.toml`` — the same ``(worktree / prefix / 'pyproject.toml').exists()``
    check ``workflow._sync_worktree_venvs`` already uses to decide which task
    modules need their own ``uv sync``.  This distinguishes a real subproject
    (e.g. ``cockpit/``) from a bare repo-root directory like ``tests/`` or
    ``src/`` that has no ``pyproject.toml`` of its own.

    Returns ``None`` when *worktree* is ``None``, *files* is empty, any file
    lives at the repo root (no top-level directory to attribute it to — a
    mixed root+subproject diff must not collapse to the subproject alone and
    silently drop the root-level file from test scoping), the files span more
    than one top-level directory, or the sole top-level directory lacks its
    own ``pyproject.toml``.
    """
    if worktree is None or not files:
        return None
    if any('/' not in f for f in files):
        return None
    components = {f.split('/', 1)[0] for f in files}
    if len(components) != 1:
        return None
    (prefix,) = components
    return prefix if (worktree / prefix / 'pyproject.toml').is_file() else None


def _root_plus_single_subproject_prefix(files: list[str], worktree: Path | None) -> str | None:
    """Return the sole subproject prefix when *files* mix root-owning file(s) with it, else ``None``.

    Complements :func:`_single_subproject_prefix`, whose "any mixed
    root+subproject diff → None" contract is deliberately pinned (used by the
    pure-subproject fallback branch, which must not collapse a mixed diff to
    the subproject alone). This helper instead detects the distinct "root
    file(s) + exactly one real subproject" shape.

    Each file is classified as either root-owning — a bare repo-root file (no
    ``/``), or a file under a top-level directory that has no ``pyproject.toml``
    of its own (e.g. ``tests/``) — or as belonging to a real subproject: a
    top-level directory that DOES carry its own ``pyproject.toml`` (same
    ``(worktree / prefix / 'pyproject.toml').is_file()`` discriminator as
    :func:`_single_subproject_prefix`).

    Returns the subproject's prefix only when at least one root-owning file
    AND exactly one distinct subproject are both present in *files*.  Returns
    ``None`` when *worktree* is ``None``, *files* is empty, no root-owning
    file is present (pure subproject diff — :func:`_single_subproject_prefix`
    already handles that case), no subproject is touched (pure root diff), or
    more than one distinct subproject is touched (ambiguous — which
    subproject would TEST be scoped to?).
    """
    if worktree is None or not files:
        return None
    subprojects: set[str] = set()
    has_root = False
    for f in files:
        if '/' not in f:
            has_root = True
            continue
        top = f.split('/', 1)[0]
        if (worktree / top / 'pyproject.toml').is_file():
            subprojects.add(top)
        else:
            has_root = True
    if has_root and len(subprojects) == 1:
        (prefix,) = subprojects
        return prefix
    return None


def _select_subproject_pytest_targets(files: list[str], prefix: str) -> list[str]:
    """Return pytest targets, relative to *prefix*, for a subproject-scoped fallback TEST command.

    Shared by :func:`_build_fallback_config`'s pure-subproject and mixed
    root+subproject branches (task 2368 amendment — previously each branch
    duplicated this selection logic inline): *files* is the list of touched
    ``.py`` files that all live under top-level subproject directory
    *prefix* (the pure-sub branch passes all of ``py_files``, guaranteed
    single-prefix by :func:`_single_subproject_prefix`'s contract; the mixed
    branch passes the subset already filtered to ``mixed_sub``).

    Selects conftest parent-dirs plus collectable tests outside them when a
    conftest is touched, else the collectable tests themselves, else no
    targets at all — logging a task-1852 warning when an orphaned test-data
    module has no anchor to derive a directory target from. Each selected
    target is then mapped to a path relative to *prefix*: a target that sits
    directly at the subproject root (e.g. ``'cockpit/conftest.py'``) maps to
    ``'.'`` rather than being left as ``prefix`` itself, so the resulting
    command is ``cd cockpit && uv run pytest .``, not the invalid ``pytest
    cockpit`` (which pytest would resolve against cwd ``cockpit`` as the
    nonexistent ``cockpit/cockpit``).

    Returns an empty list when there is nothing to target (e.g. *files* is
    source-only) — callers treat that as "no subproject test segment".
    """
    has_conftest = any(_is_conftest(f) for f in files)
    collectable_tests = [f for f in files if _is_collectable_test_file(f)]
    has_test_data = any(
        _is_test_file(f) and not _is_collectable_test_file(f) for f in files
    )
    if has_conftest:
        conftest_dirs = sorted({
            f.rsplit('/', 1)[0] if '/' in f else '.'
            for f in files
            if _is_conftest(f)
        })
        if '.' not in conftest_dirs:
            outside = [
                t for t in collectable_tests
                if not any(t.startswith(d + '/') for d in conftest_dirs)
            ]
        else:
            outside = []
        targets = conftest_dirs + outside
    elif collectable_tests:
        targets = collectable_tests
    else:
        targets = []
        if has_test_data:
            # No collectable tests and no conftest to anchor a directory
            # target, so this data-module change ships unvalidated.  Warn
            # rather than silently skipping (amendment to task 2344).
            _data_files = [
                f for f in files
                if _is_test_file(f) and not _is_collectable_test_file(f)
            ]
            logger.warning(
                '_build_fallback_config: test-tree data module(s) %s skipped '
                '— no tests will run for this change; configure a non-default '
                'test_command to validate data-module changes (task 1852)',
                _data_files,
            )
    return [
        '.' if t == prefix else (t[len(prefix) + 1:] if t.startswith(prefix + '/') else t)
        for t in targets
    ]


def _config_test_extras(cmd: str | None) -> list[str]:
    """Extract every ``--extra <name>`` / ``--extra=<name>`` flag from *cmd*, in order.

    Carries a project's canonical ``config.test_command`` extras into the
    fallback-synthesized ``uv run pytest`` command (task 2641; the TEST-path
    twin of the task-2355 TYPE/LINT cold-verify dev-dep fix) so a cold merge
    worktree syncs the project's dev-group deps before pytest is spawned.

    Scope (review follow-up, task 2641): only ``--extra``/``--extra=`` is
    recognized. Other uv dependency-selection flags — ``--group``/
    ``--group=``, ``--all-extras``, ``--all-groups``, ``--dev``/``--no-dev``
    — are intentionally NOT extracted; a project that selects its test deps
    via one of those instead of ``--extra`` will still hit the cold-verify
    "Failed to spawn" race this helper fixes for ``--extra``-based projects.
    Broadening coverage to those flags is a follow-up, not handled here.

    Returns ``[]`` when *cmd* is ``None``, carries no ``--extra`` flags, or
    fails to tokenize (unbalanced quotes — mirrors :func:`parse_config_command`'s
    ``shlex.split`` + ``except ValueError`` guard).
    """
    if cmd is None:
        return []
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return []
    extras: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == '--extra' and i + 1 < len(tokens):
            extras.extend(['--extra', tokens[i + 1]])
            i += 2
            continue
        if token.startswith('--extra='):
            extras.extend(['--extra', token.split('=', 1)[1]])
        i += 1
    return extras


def _build_fallback_config(
    task_files: list[str],
    config: OrchestratorConfig | None = None,
    worktree: Path | None = None,
    content_cache: dict[str, str | None] | None = None,
) -> ModuleConfig | None:
    """Build a synthetic ModuleConfig from *task_files* when no module configs match.

    Filters to ``.py`` files, classifies into source vs test, and builds
    targeted commands.  When *config* provides non-default commands (e.g.
    ``uv run --extra dev --extra web pytest``), those are used directly so
    that tools not installed globally can still be reached.

    For ``lint_command`` and ``type_check_command``, :func:`_scope_to_keyword`
    narrows the configured command to the touched files when the standard tool
    keyword appears (e.g. ``ruff check`` in ``uv run ruff check``), and
    returns the command unchanged when it doesn't (e.g. ``true`` or
    ``mypy``-based type checking).  For ``test_command``: when *worktree* is
    given and every touched file lives under a single top-level directory
    that is itself a real subproject (its own ``pyproject.toml`` — see
    :func:`_single_subproject_prefix`), the test command is scoped to run
    from *inside* that subproject alone.  This prevents a fleet-wide
    configured ``test_command`` (e.g. ``cd shared && uv run pytest tests/ &&
    cd ../escalation && ...``) from dragging every OTHER subproject's suite —
    and its unrelated red-main state — into this task's verify (regression
    guard for esc-2293-13 / task 2293).  Otherwise, the configured command is
    used as-is when it differs from the bare ``pytest`` default so that
    complex flag sequences like ``-m 'not slow' --ignore=tests/e2e`` are not
    mangled by :func:`_scope_to_keyword`'s prefix-then-parse approach.

    *content_cache*, when given, is threaded into the structural-file content
    read (see :func:`_worktree_reader`) so a file already read through the
    SAME dict is not read from disk again (task γ amendment). Mirrors
    ``scope_module_config``'s identical parameter for interface symmetry;
    ``run_scoped_verification``'s fallback branch currently leaves this
    unwired (does not pass a cache shared with ``derive_verify_plan``'s
    reader) — see the "NOTE (task γ amendment)" comment at that call site for
    why. A direct caller can still pass its own dict to dedupe repeat calls.

    Returns ``None`` when no ``.py`` files are found.
    """
    py_files = [f for f in task_files if f.endswith('.py')]
    if not py_files:
        return None

    # conftest.py cannot be passed directly to pytest (pytest >= 9 exits 1 with
    # "no tests ran").  The fallback path has no mc.test_command to reuse, so
    # we target the *parent directory* of each conftest instead — that directory
    # contains every test the conftest can affect.  A root-level conftest (no
    # parent) maps to '.' so we never produce 'pytest conftest.py'.  Sorted
    # deduped set gives deterministic output.
    has_conftest = any(_is_conftest(f) for f in py_files)
    # Narrow: only test_*.py / *_test.py that pytest will actually collect.
    # A data module under tests/ (e.g. tests/some_data.py) satisfies
    # _is_test_file but NOT _is_collectable_test_file — passing it to pytest
    # produces rc=5 ("no tests ran") → RED.  Task 1852 fixes at scoping layer.
    collectable_tests = [f for f in py_files if _is_collectable_test_file(f)]
    # has_test_data: in-tree but not collectable (and not conftest).
    # In _build_fallback_config there is no owning module suite to fall back to,
    # so we only propagate has_test_data to the non-default configured command
    # branch (where a real suite exists).  In the bare-pytest branch a lone data
    # module yields test_cmd = None — targeting its parent dir risks rc=5 if
    # that dir holds only fixtures/data with zero tests.
    has_test_data = any(
        _is_test_file(f) and not _is_collectable_test_file(f) for f in py_files
    )

    # Lint and type commands: use configured commands when *config* is provided.
    # _scope_to_keyword narrows to the touched files when the standard tool
    # keyword appears in the command (folding in any leading `cd <subproject>
    # &&`, since the fallback runs from the worktree root and a module-cd
    # would misresolve the root-relative file path just scoped in), and
    # returns the command unchanged when it doesn't. _reproject_str then
    # reprojects a bare ``uv run <tool>`` into a tool-bearing member uv
    # context (task 2036): the depless workspace-root project cannot spawn
    # ruff/pyright.
    if config is not None:
        lint_scoped = _scope_to_keyword(config.lint_command, 'ruff check', py_files)
        lint_cmd = _reproject_str(lint_scoped, _FALLBACK_UV_PROJECT)
        type_scoped = _scope_to_keyword(config.type_check_command, 'pyright', py_files)
        type_cmd = _reproject_str(type_scoped, _FALLBACK_UV_PROJECT)
    else:
        lint_scoped = lint_cmd = 'ruff check ' + ' '.join(py_files)
        type_scoped = type_cmd = 'pyright ' + ' '.join(py_files)

    # Structural widening (D2, task γ): a Protocol/TypedDict-defining .py file
    # needs the FULL unscoped type-check command — file-scoped pyright cannot
    # verify cross-file Protocol/TypedDict conformance.  Mirrors
    # scope_module_config's has_structural block; this is the gap
    # _build_fallback_config used to lack entirely (it never inspected file
    # content at all) prior to task γ.  Overrides type_scoped/type_cmd
    # in-place so every branch below (pure-subproject, mixed-subproject,
    # non-default-configured, bare-default) inherits the widening uniformly,
    # matching how has_structural applies module-wide in scope_module_config.
    has_structural = False
    structural_trigger: str | None = None
    _type_check_command_raw = config.type_check_command if config is not None else 'pyright'
    if worktree is not None and _type_check_command_raw:
        _read = _worktree_reader(worktree, cache=content_cache)
        for f in py_files:
            content = _read(f)
            if content is not None and _is_structural_python_file(f, content):
                has_structural = True
                structural_trigger = f
                break
    if has_structural:
        logger.info(
            'pyright unscoped (fallback): structural file %s in diff', structural_trigger,
        )
        type_scoped = _type_check_command_raw
        type_cmd = (
            _reproject_str(type_scoped, _FALLBACK_UV_PROJECT) if config is not None else type_scoped
        )

    # Subproject-scoped TEST command (task 2344): when every touched file
    # lives under a single top-level directory that is itself a real
    # subproject (its own pyproject.toml in *worktree*), scope TEST to that
    # subproject alone rather than falling through to config.test_command's
    # fleet-wide chain verbatim below.  lint_cmd/type_cmd above are already
    # scoped to the touched files, so only test_command needs this treatment.
    sub = _single_subproject_prefix(py_files, worktree)
    if sub is not None:
        # Target selection (conftest parent-dirs + outside collectable tests,
        # else collectable tests, else none) is shared with the mixed-sub
        # branch below via _select_subproject_pytest_targets (task 2368
        # amendment: previously duplicated inline in both branches, plus a
        # third near-copy in the bare-pytest branch further below, which is
        # left untouched since it also covers the worktree=None case that
        # has no `prefix` to make relative).
        rel_targets = _select_subproject_pytest_targets(py_files, sub)
        # Cold-verify dev-dep sync (task 2641): carry any --extra flags from
        # the project's canonical test_command into the synthesized command
        # so a cold merge worktree's `uv run` syncs the project's dev-group
        # extra before spawning pytest (TEST-path twin of the task-2355
        # TYPE/LINT fix below). extras is [] for a no-extra config, so the
        # output is byte-identical to before this change.
        #
        # Assumption (review follow-up, task 2641): extras are spliced into
        # *sub*'s own uv context without checking that *sub*'s own
        # pyproject.toml actually declares them. If config.test_command's
        # extra is declared elsewhere (e.g. the repo root or another
        # subproject) but not in *sub*, `uv run --extra <name>` hard-fails
        # with "Extra `<name>` is not defined" — turning a passing change
        # RED instead of silently dropping the flag. This mirrors the
        # identical, pre-existing assumption in the task-2355 LINT/TYPE fix
        # just below (`_scope_fallback_tool_to_subproject` leaves a verbatim
        # `--extra` clause untouched with no subproject-declaration check
        # either), so it is not a new gap introduced here — see
        # TestBuildFallbackConfigSubprojectScoped
        # .test_extras_carried_verbatim_even_when_undefined_in_touched_subproject.
        extras = _config_test_extras(config.test_command if config is not None else None)
        test_cmd = (
            'cd ' + sub + ' && uv run ' + ' '.join([*extras, 'pytest', *rel_targets])
            if rel_targets else None
        )
        # Cold-verify dev-dep race (task 2355): rescope TYPE/LINT into *sub*'s
        # own uv context so `uv run --project <sub>` syncs its dev-group deps
        # (e.g. hypothesis) before the tool runs, rather than racing the
        # concurrently-run TEST command's `uv run` sync on a cold shared
        # .venv (esc-2293-20).
        type_cmd = _scope_fallback_tool_to_subproject(type_scoped, 'pyright', sub)
        lint_cmd = _scope_fallback_tool_to_subproject(lint_scoped, 'ruff check', sub)
        return ModuleConfig(
            prefix=sub,
            lint_command=lint_cmd,
            type_check_command=type_cmd,
            test_command=test_cmd,
        )

    # Mixed root+single-subproject TEST scoping (task 2368): `sub` above is
    # None whenever the diff mixes root-owning file(s) (a bare repo-root
    # file, or a file under a top-level dir with no pyproject.toml of its
    # own, e.g. tests/) with a subproject — `_single_subproject_prefix`
    # deliberately disqualifies ANY mixed diff.  Without this branch, such a
    # diff falls through to the config-verbatim branch below and runs the
    # entire fleet-wide chain (~7400 unrelated tests; esc-2293-13/-26/-27
    # misattributed flakes).  `_root_plus_single_subproject_prefix` detects
    # the specific "root file(s) + exactly one real subproject" shape;
    # when it fires, TEST is scoped to that subproject's own tests plus the
    # root-owning `tests/scripts/` suite — the two suites that actually
    # cover the diff — rather than the fleet fanout.  lint_cmd/type_cmd are
    # NOT rescoped into the subproject's narrow uv context here (unlike the
    # pure-sub branch above): a mixed diff includes root files that don't
    # belong to that narrow env, so lint/type keep the broad file-scoped +
    # reprojected commands already computed above.
    mixed_sub = _root_plus_single_subproject_prefix(py_files, worktree)
    if mixed_sub is not None:
        # Target selection restricted to sub_files (files under `mixed_sub`)
        # so a root-level conftest/test file doesn't leak into the
        # subproject-scoped target selection — it's covered by
        # _ROOT_OWNING_TEST_COMMAND instead. Shares its selection logic with
        # the pure-sub branch above via _select_subproject_pytest_targets
        # (task 2368 amendment — this was a third near-verbatim inline copy).
        sub_files = [f for f in py_files if f.startswith(mixed_sub + '/')]
        mixed_rel_targets = _select_subproject_pytest_targets(sub_files, mixed_sub)
        # Intentional coverage trade-off (task 2368; mirrors the pure-sub
        # branch's task-2344 precedent of test_command=None for a
        # source-only subproject diff): when the touched subproject's
        # portion of a mixed diff is source-only, mixed_rel_targets is empty
        # and TEST scopes to _ROOT_OWNING_TEST_COMMAND alone — the touched
        # subproject's own suite does NOT run to validate the source change.
        # This narrows gating versus the old fleet-chain-verbatim fallback
        # (which ran the subproject's whole suite, at the cost of dragging
        # in ~7400 unrelated tests). Scoping precision was chosen over
        # broader gating for the same reason the pure-sub branch made this
        # trade-off; revisit if source-only subproject regressions start
        # slipping through to merge.
        # Cold-verify dev-dep sync (task 2641): same twin-bug fix as the
        # pure-subproject branch above — carry test_command's --extra flags
        # into the touched subproject's own pytest segment only.
        # _ROOT_OWNING_TEST_COMMAND is a separate, dark_factory-specific
        # command and is left untouched. Same undeclared-extra assumption as
        # the pure-subproject branch above applies here too (see the
        # "Assumption (review follow-up, task 2641)" comment there).
        mixed_extras = _config_test_extras(config.test_command if config is not None else None)
        test_cmd = (
            'cd ' + mixed_sub + ' && uv run ' + ' '.join([*mixed_extras, 'pytest', *mixed_rel_targets])
            + ' && cd .. && ' + _ROOT_OWNING_TEST_COMMAND
            if mixed_rel_targets else _ROOT_OWNING_TEST_COMMAND
        )
        # Robustness (task 2368 amendment): _ROOT_OWNING_TEST_COMMAND only
        # runs tests/scripts/, but the root-owning files of a mixed diff can
        # include a collectable test file elsewhere (e.g. a touched
        # tests/e2e/test_x.py) that will then silently never run — unlike
        # the sub_has_test_data path above, this gap emitted no warning.
        # Surface it so a real, uncovered root-level test file isn't silent.
        root_files = [f for f in py_files if not f.startswith(mixed_sub + '/')]
        uncovered_root_tests = [
            f for f in root_files
            if _is_collectable_test_file(f) and not f.startswith('tests/scripts/')
        ]
        if uncovered_root_tests:
            logger.warning(
                '_build_fallback_config: root-owning test file(s) %s will not run '
                '— _ROOT_OWNING_TEST_COMMAND only covers tests/scripts/ (task 2368)',
                uncovered_root_tests,
            )
        return ModuleConfig(
            prefix=mixed_sub,
            lint_command=lint_cmd,
            type_check_command=type_cmd,
            test_command=test_cmd,
        )

    # Test command: when a non-default configured command exists (e.g.
    # `uv run --extra dev --extra web pytest -m 'not slow' --ignore=tests/e2e`),
    # use it as-is to avoid mangling multi-token flags.  The configured command
    # already encodes which extras, markers, and ignores are required.
    # has_test_data is included here: a data module (e.g. σ-allowlist) consumed
    # by real tests warrants running the full suite when a real suite exists —
    # the configured test_command has real tests, so rc≠5 (task 1852).
    if config is not None and config.test_command != 'pytest':
        test_cmd: str | None = config.test_command if (collectable_tests or has_conftest or has_test_data) else None
        return ModuleConfig(
            prefix='__fallback__',
            lint_command=lint_cmd,
            type_check_command=type_cmd,
            test_command=test_cmd,
        )

    if has_conftest:
        conftest_dirs = sorted({
            f.rsplit('/', 1)[0] if '/' in f else '.'
            for f in py_files
            if _is_conftest(f)
        })
        # Also include test files that live *outside* every conftest directory.
        # e.g. ['a/conftest.py', 'b/test_x.py'] → 'pytest a b/test_x.py' so
        # tests in b/ are not silently skipped.  A root-level conftest ('.')
        # shadows everything, so in that case no files are "outside".
        #
        # `collectable_tests` always contains file paths (e.g. 'a/sub/test_x.py'),
        # never bare directory paths — _is_collectable_test_file gates on filename
        # prefixes/suffixes, none of which match a directory entry.  That
        # guarantees `t.startswith(d + '/')` reliably means "t is inside
        # directory d" without false positives from a sibling like 'ab/'.
        if '.' not in conftest_dirs:
            outside = [
                t for t in collectable_tests
                if not any(t.startswith(d + '/') for d in conftest_dirs)
            ]
        else:
            outside = []
        targets = conftest_dirs + outside
        test_cmd = 'pytest ' + ' '.join(targets)
    elif collectable_tests:
        # Only collectable (test_*.py / *_test.py) files are targeted.
        # A lone data module under tests/ yields test_cmd = None (no rc=5).
        test_cmd = 'pytest ' + ' '.join(collectable_tests)
    else:
        test_cmd = None
        if has_test_data:
            # A test-tree data module (e.g. tests/some_data.py) is being skipped:
            # no collectable tests present and no configured suite to fall back
            # to.  The skip avoids a false-RED rc=5, but this change ships
            # UNVALIDATED.  Configure a non-default test_command for this project
            # so data-module changes are validated by a real suite (task 1852).
            _data_files = [
                f for f in py_files
                if _is_test_file(f) and not _is_collectable_test_file(f)
            ]
            logger.warning(
                '_build_fallback_config: test-tree data module(s) %s skipped '
                '— no tests will run for this change; configure a non-default '
                'test_command to validate data-module changes (task 1852)',
                _data_files,
            )

    return ModuleConfig(
        prefix='__fallback__',
        lint_command=lint_cmd,
        type_check_command=type_cmd,
        test_command=test_cmd,
    )


def _verify_duration_secs(runs: list[dict]) -> float:
    """Sum per-command ``duration_secs`` values from a verification runs list.

    Each entry is expected to have a ``duration_secs`` key (float); entries
    that are missing the key contribute 0.0.  Returns 0.0 for an empty list.

    This is the correct wall-clock measure when commands were run **serially**
    (sum of sequential durations).  For the **concurrent** branch (asyncio.gather
    of test/lint/type) the caller should use ``max(...)`` of the individual
    durations — mirroring the multi-module logic in ``_aggregate_results`` — to
    avoid overstating wall-clock by ~3×.
    """
    return sum(r.get('duration_secs', 0.0) for r in runs)


@dataclass
class CheckRun:
    """The execution outcome of a single check (test/lint/type).

    Introduced (task 2133) to collapse ``run_verification``'s 15 parallel
    per-check scalar locals (``{test,lint,type}_{rc,out,timed_out,
    started_at,duration}``) into one object per check, and — paired with
    :class:`VerifyAttempt` — to compute the pure-timeout-consistency formula
    in exactly one place instead of two hand-duplicated copies.
    """

    label: str
    cmd: 'str | None'
    rc: int
    output: str
    timed_out: bool
    started_at: 'str | None'
    duration_secs: float
    # Per-segment execution facts when this check ran as a SEGMENTED `&&`
    # chain (task 3338 / esc-3062-2); ``None`` when it did not. LAST field so
    # every pre-3338 positional construction site stays valid. A plain
    # JSON-native ``list[dict]`` rather than a nested dataclass, for the same
    # reason the rest of this schema is flat: ``to_dict()``'s output is written
    # straight into JSON, so anything needing its own serialisation step is a
    # second place for the shape to drift. ``[]`` would mean "segmented but
    # with no segments" — an impossible state ``split_and_chain_segments``'
    # fewer-than-2 refusal already prevents.
    segments: 'list[dict] | None' = None

    @classmethod
    def skipped(cls, label: str) -> 'CheckRun':
        """Build the CheckRun for a check whose command is ``None`` (module_config skip).

        Matches ``_run_or_skip_timed``'s early return for ``cmd is None``:
        vacuously passing (``rc=0``), no output, never timed out, no start
        time, zero duration.
        """
        return cls(
            label=label, cmd=None, rc=0, output='', timed_out=False,
            started_at=None, duration_secs=0.0,
        )

    def to_dict(self) -> dict:
        """Serialise to the runs-dict schema consumed by ``_persist_attempt_logs``/
        ``_build_summary_payload``/``_verify_duration_secs``/``_archive_merge_verify_logs``
        (all take ``list[dict]``) — the exact 8-key shape (label/cmd/rc/output/
        timed_out/started_at/duration_secs/segments), 7 of which were
        previously hand-built inline in ``run_verification``.

        ``started_at`` is normalised via ``or ''``: a skipped check's
        ``None`` serialises as ``''``, matching the pre-refactor
        ``test_started_at or ''`` pattern so downstream JSON consumers see
        the same shape as before.

        ``segments`` (task 3338) is emitted UNCONDITIONALLY, as ``None`` on an
        unsegmented run, and passed through verbatim. A conditionally-present
        key would leave a consumer unable to distinguish "this build predates
        segments" from "this run was not segmented" — reintroducing an
        absent-vs-null ambiguity in the very schema whose job is to make
        skipped-vs-passed unambiguous.
        """
        return {
            'label': self.label,
            'cmd': self.cmd,
            'rc': self.rc,
            'output': self.output,
            'timed_out': self.timed_out,
            'started_at': self.started_at or '',
            'duration_secs': self.duration_secs,
            'segments': self.segments,
        }


@dataclass
class VerifyAttempt:
    """The CheckRun results (test/lint/type) for one full verification attempt.

    Introduced (task 2133) so ``run_verification`` carries ONE object
    through its retry loop instead of 15 parallel scalar locals
    (``{test,lint,type}_{rc,out,timed_out,started_at,duration}``), and so
    ``passed``/``any_timed_out``/``pure_timeout_failure`` are each computed
    exactly ONCE. Previously this 6-clause formula was hand-duplicated at
    two call sites — the first-pass retry loop and the env-recovery branch
    (task 2048 added the second copy) — which is the exact drift surface
    that once let a recovery run's wall-clock timeout leave a stale
    ``timed_out=False`` while ``category`` flipped to ``infra_timeout``.
    With a single property definition read from every call site, first-pass
    and env-recovery are now structurally incapable of disagreeing.
    """

    checks: list[CheckRun]

    @property
    def passed(self) -> bool:
        return all(c.rc == 0 for c in self.checks)

    @property
    def any_timed_out(self) -> bool:
        return any(c.timed_out for c in self.checks)

    @property
    def pure_timeout_failure(self) -> bool:
        """True when the attempt failed and every failing check is a timeout.

        The generic (list-over-``checks``) form of the two hand-duplicated
        formulas this task collapses: ``not passed`` rules out a clean
        pass, ``any_timed_out`` requires at least one check to have hit the
        wall clock, and the trailing ``all(...)`` requires every OTHER
        failing check to also be a timeout — so a genuine non-timeout
        failure (e.g. a real lint error) alongside a timeout is never
        misclassified as a pure, retryable timeout.
        """
        return (
            not self.passed
            and self.any_timed_out
            and all(c.rc == 0 or c.timed_out for c in self.checks)
        )

    def _by_label(self, label: str) -> CheckRun:
        found = next((c for c in self.checks if c.label == label), None)
        if found is None:
            raise KeyError(
                f'no check labeled {label!r} in {[c.label for c in self.checks]}'
            )
        return found

    @property
    def test(self) -> CheckRun:
        return self._by_label('test')

    @property
    def lint(self) -> CheckRun:
        return self._by_label('lint')

    @property
    def type(self) -> CheckRun:
        # Shadows the `type` builtin, but this project's ruff config selects
        # ["E", "F", "UP", "B", "SIM", "I"] — flake8-builtins ("A") is not
        # enabled — so this is lint-safe; matches the CheckRun.label vocabulary
        # ('test'/'lint'/'type') used throughout verify.py.
        return self._by_label('type')


@dataclass
class VerifyResult:
    passed: bool
    test_output: str
    lint_output: str
    type_output: str
    summary: str
    timed_out: bool = False
    cause_hint: str = ''
    category: str = ''
    worktree_log_paths: list[str] = field(default_factory=list)
    archive_log_paths: list[str] = field(default_factory=list)
    # Machine-readable payload for a flock-contention outcome (task 2306 α):
    # {'host', 'holder_pgid', 'waiter_pgid'}.  Deliberately a plain JSON-native
    # dict (NOT a nested dataclass) so the generic codec (result_to_dict=asdict /
    # result_from_dict=VerifyResult(**d)) round-trips it losslessly — a nested
    # dataclass would come back as a bare dict from asdict but from_dict would
    # pass it straight into VerifyResult(**d) without reconstructing it.
    # None for every non-contention result (default / back-compat).
    contention: dict | None = None
    # Machine-readable record of the VerifyPlan (verify_plan.py, PRD task γ)
    # that drove this verification attempt: VerifyPlan.to_dict() — {'runs':
    # [...], 'needs_pipeline_guard_check': bool}. Deliberately a plain
    # JSON-native dict (NOT a nested VerifyPlan dataclass), mirroring
    # `contention` immediately above, so the generic codec
    # (result_to_dict=asdict / result_from_dict=VerifyResult(**d)) round-trips
    # it losslessly — a nested dataclass would come back as a bare dict from
    # asdict but from_dict would pass it straight into VerifyResult(**d)
    # without reconstructing it. None when no plan was derived (default /
    # back-compat / _trivial_pass and other non-planned results).
    #
    # Fidelity: `plan` IS the plan that drove execution on BOTH branches
    # (task κ, verify-scope-inversion-prd.md). On the module-config path,
    # derive_verify_plan is derived once and executed directly (see
    # _executed_module_configs_from_plan), so it faithfully records what
    # ran. On the fallback (no-module_configs) path, the same decision is
    # derived once and then reconciled against _build_fallback_config's
    # already-executed ModuleConfig via _executed_fallback_plan (see
    # _safe_derive_verify_plan_dict and its call site below), so the
    # attached record reflects the actual subproject/mixed-subproject
    # rescoping and OPAQUE-chain first-clause scoping that ran, not the
    # idealized flat '__fallback__' decision alone — see
    # derive_verify_plan's docstring ("Fidelity" paragraph) for what its raw,
    # unreconciled return value alone still omits.
    plan: dict | None = None
    # Machine-readable failing/errored pytest node ids, parsed from a
    # structured merge-role run's junitxml report (task μ,
    # verify-scope-inversion-prd.md — the baseline-attribution signal;
    # see _extract_failing_test_ids_from_junit and with_junitxml in
    # verify_cmd.py). Deliberately a plain JSON-native `list[str] | None`
    # (mirrors `contention`/`plan` immediately above), so it round-trips
    # losslessly through the generic codec (asdict / VerifyResult(**d)).
    #
    # None = "no junit collected" — role != 'merge', breadth != 'full', an
    # OPAQUE/raw-retained test command, or an unreadable/malformed junit
    # report — the B3 degrade signal callers fall back on. `[]` = "junit
    # collected, zero failing" (main/branch genuinely clean under this
    # run) and must NOT be conflated with None.
    failing_test_ids: list[str] | None = None
    # Task 3173: one FailureCategory per FAILING leg, in test/lint/type order,
    # exactly as `_summarize_checks` classified them (its fifth return
    # element). Deliberately a plain JSON-native `list[str] | None` (mirrors
    # `contention`/`plan`/`failing_test_ids` above), so it round-trips
    # losslessly through the generic codec (asdict / VerifyResult(**d)).
    #
    # WHY IT EXISTS SEPARATELY FROM `category`: `category` is the
    # severity-ranked WORST leg (_worst_category), which answers "how bad was
    # this run" — what the retry loop, the archive and the transient-infra
    # hold need. It cannot answer "did EVERY leg fail to produce a verdict",
    # because a rank-1 infra_kill hides a co-occurring rank-11 test_failure.
    # merge_queue's per-land cross-check needs the second question (only a run
    # in which every failing leg is verdict-less may decline to veto a remote
    # PASS), so the per-leg answer is CARRIED rather than inferred.
    #
    # None = NOT RECORDED, and is the fail-CLOSED default a consumer must
    # never treat as indeterminate: an older remote's wire payload simply
    # omits the key and lands here, as do `_trivial_pass`, the verify_runner
    # UNSCOPED_TYPECHECK_* sentinel results, and any hand-constructed result.
    # `[]` = "no failing legs" (a pass), which is likewise not a licence.
    failing_leg_categories: list[str] | None = None
    # Task 2823: True IFF this result came from _trivial_pass — a config-only
    # (no .py/.rs) merge that short-circuited verification WITHOUT running any
    # suite. The merge worker's pass path keys on this to refuse advancing main
    # over a known-red baseline (a trivial pass ran nothing, so it is no
    # evidence the red cleared), while still letting a NON-trivial pass (full
    # suite actually ran and passed) heal a red main. A plain JSON-native bool,
    # so it round-trips through the generic codec (asdict / VerifyResult(**d))
    # exactly like contention/plan/failing_test_ids — no allowlist to touch.
    # Comparable (participates in __eq__): unlike the compare=False
    # duration_secs below, it is invariant across two runs of the same logical
    # verification.
    trivial: bool = False
    # Wall-clock verify cost.  For a single-module run: max(test, lint, type)
    # when the three commands ran concurrently (asyncio.gather), or their sum
    # when run serially.  For a multi-module run: max across child
    # VerifyResults (set by _aggregate_results — modules run concurrently via
    # asyncio.gather so max approximates wall-time).  Defaults to 0.0 for
    # _trivial_pass and mocked results.
    #
    # compare=False: wall-clock duration differs between two independent runs of
    # the same logical verification, so it must NOT participate in __eq__ (else
    # cli_result == local can never hold — see test_cli
    # test_verify_merge_cli_wrapper_transparency). Folded in here to clear a
    # preexisting main red introduced by task 1802.
    duration_secs: float = field(default=0.0, compare=False)

    def failure_report(self) -> str:
        """Format all failures into a single report for the debugger."""
        sections = []
        if self.timed_out:
            # Lead with timeout info so the debugger knows the failure may not
            # be real code — list which commands actually hit the wall clock.
            timed_out_cmds = []
            if self.test_output and 'timed out' in self.test_output.lower():
                timed_out_cmds.append('test')
            if self.lint_output and 'timed out' in self.lint_output.lower():
                timed_out_cmds.append('lint')
            if self.type_output and 'timed out' in self.type_output.lower():
                timed_out_cmds.append('type')
            joined = ', '.join(timed_out_cmds) if timed_out_cmds else 'unknown'
            sections.append(
                f'## Verify Timed Out\n\nCommands that hit the timeout: {joined}.\n'
                f'This may indicate a cold build, resource contention, or a '
                f'genuinely hanging command — inspect the output below before '
                f'treating it as a real failure.'
            )
        if self.cause_hint:
            sections.append(f'## Failure Cause\n\n{self.cause_hint}')
        # ## Verify Logs — list on-disk paths so the reader can `cat` the full evidence.
        # Appears between ## Failure Cause and ## Test Failures.
        if self.worktree_log_paths or self.archive_log_paths:
            log_lines = ['## Verify Logs', '']
            if self.category:
                log_lines.append(f'Category: {self.category}')
                log_lines.append('')
            log_lines.append('Worktree:')
            for p in self.worktree_log_paths:
                log_lines.append(f'- {p}')
            if self.archive_log_paths:
                log_lines.append('')
                log_lines.append('Archive (durable, survives worktree cleanup):')
                for p in self.archive_log_paths:
                    log_lines.append(f'- {p}')
            sections.append('\n'.join(log_lines))
        if self.test_output and _FAILURE_MARKER_RE.search(self.test_output):
            excerpt = _failure_anchored_excerpt(self.test_output, cap=3000)
            sections.append(f'## Test Failures\n\n```\n{excerpt}\n```')
        if self.lint_output and self.lint_output.strip():
            sections.append(f'## Lint Issues\n\n```\n{self.lint_output[-2000:]}\n```')
        if self.type_output and 'error' in self.type_output.lower():
            sections.append(f'## Type Errors\n\n```\n{self.type_output[-2000:]}\n```')
        return '\n\n'.join(sections) if sections else self.summary


def _scope_tag_for(project_root: Path) -> str:
    """Derive a deterministic, systemd-name-safe per-project scope tag.

    All ``orchestrator-*.service`` units (reify, dark-factory, know-live, …)
    run the SAME orchestrator package under ONE shared ``systemctl --user``
    session, so ``df-verify-*.scope`` is a single per-user namespace shared
    across projects.  Embedding this tag in the verify-scope unit name
    (``df-verify-{tag}-{uuid}.scope``) lets each orchestrator's startup sweep
    reap ONLY its own leftovers — a bare-glob sweep would reap a sibling
    project's LIVE in-flight verify scope during a rolling fleet restart.

    The tag is ``{basename-slug}-{path-hash}``:

    - the ``project_root`` basename, lowercased and sanitized to ``[a-z0-9-]``
      (operator-legible), bounded in length to keep the total unit name within
      systemd's limit; and
    - the first 8 hex chars of ``sha1(str(resolved_path))`` so two projects
      that share a basename but live at different absolute paths still get
      distinct tags (collision-resistant disambiguation).

    Pure and deterministic: the same ``project_root`` always yields the same
    tag, so a fresh boot reaps its dead predecessor's same-tagged scopes.
    """
    resolved = Path(project_root).resolve()
    slug = re.sub(r'[^a-z0-9-]+', '-', resolved.name.lower()).strip('-')
    # Bound the slug so the whole df-verify-{slug}-{hash}-{uuid}.scope name
    # stays well within systemd's unit-name length limit.
    slug = slug[:32] or 'proj'
    digest = hashlib.sha1(str(resolved).encode()).hexdigest()[:8]
    return f'{slug}-{digest}'


def _verify_scope_name(scope_tag: str) -> str:
    """Build a per-project, uuid-unique transient verify-scope unit name.

    Shape: ``df-verify-{scope_tag}-{uuid}.scope``.  The ``df-verify-`` prefix
    is preserved (existing prefix matchers are unaffected); ``scope_tag`` (see
    ``_scope_tag_for``) confines the leftover-scope startup sweep to THIS
    project; the 12-hex uuid segment keeps concurrent verifies from colliding
    on a unit name.
    """
    return f'df-verify-{scope_tag}-{uuid.uuid4().hex[:12]}.scope'


async def _kill_cgroup_scope(unit: str) -> None:
    """Force-kill a transient systemd ``--user`` scope unit and reap its cgroup.

    Sends SIGKILL to every process in the scope's cgroup, then stops the unit.
    Used when a verify command was spawned inside a transient scope (see
    ``_run_cmd``'s ``use_cgroup_scope`` path): killing the cgroup reaps the
    ENTIRE subtree (bash → cargo → rustc, and any inner ``timeout`` that
    setpgid'd cargo into a separate process group), which a plain ``killpg`` on
    the spawn pgid cannot reach.  Best-effort: every systemctl call is wrapped
    in ``suppress`` and bounded by a short timeout so a hung manager cannot
    block the verify path.
    """
    for action in (['kill', '--signal=SIGKILL', unit], ['stop', unit]):
        with contextlib.suppress(Exception):
            p = await asyncio.create_subprocess_exec(
                'systemctl', '--user', *action,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(p.wait(), 10)


async def _scope_is_gone(unit: str) -> bool:
    """Best-effort liveness probe: ``True`` iff ``systemctl --user is-active``
    reports *unit* is no longer active (``inactive`` / ``failed`` / unknown),
    ``False`` if it is still ``active`` (or mid-transition).

    Used by :func:`reap_leftover_verify_scopes` to CONFIRM a reap rather than
    assume it: :func:`_kill_cgroup_scope` is best-effort and can leave a
    genuinely un-killable scope alive, so a startup crash-recovery sweep must
    verify before it reports a scope reaped.  Fully fail-soft — if the probe
    itself cannot be run (systemctl absent, timeout, manager fault) it returns
    ``True`` (assume gone) so the sweep degrades no worse than the previous
    unconditional behaviour and a systemd fault never blocks startup.
    """
    with contextlib.suppress(Exception):
        p = await asyncio.create_subprocess_exec(
            'systemctl', '--user', 'is-active', unit,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(p.communicate(), 10)
        state = out.decode('utf-8', 'replace').strip() if out else ''
        return state not in ('active', 'activating', 'deactivating')
    return True


async def reap_leftover_verify_scopes(project_root: Path) -> list[str]:
    """Stop THIS project's leftover transient verify-scope units at startup.

    A crash/SIGKILL of a prior orchestrator incarnation can strand a
    ``df-verify-{tag}-{uuid}.scope`` whose processes keep running (the
    controller died but the scope's cgroup subtree — bash → cargo → rustc —
    lives on).  Run once before the first dispatch, this sweep enumerates and
    reaps ONLY this project's leftovers, returning the names of the scopes
    CONFIRMED gone after the reap.  A scope that survives the reap attempt (a
    genuinely un-killable, still-``active`` unit) is NOT returned and is logged
    loudly at WARNING, rather than being silently over-reported as reaped.

    Cross-project safety: every ``orchestrator-*.service`` unit shares ONE
    per-user ``systemctl --user`` session, so ``df-verify-*.scope`` is a single
    shared namespace.  The enumeration is TAG-SCOPED to
    ``df-verify-{tag}-*.scope`` (see ``_scope_tag_for``) AND the returned names
    are DEFENSIVELY re-filtered to the same ``df-verify-{tag}-…\\.scope`` shape,
    so a sibling project's LIVE in-flight verify scope can never be reaped even
    if the ``systemctl`` glob were to over-return.  systemd guarantees one
    incarnation per unit, and this runs before this incarnation's first
    dispatch, so any same-tag scope is necessarily a dead predecessor's leak.

    Fully fail-soft: returns ``[]`` (never raises) when ``systemctl`` /
    ``systemd-run`` are unavailable, or on any enumeration/parse error — a
    systemd fault must never abort startup.
    """
    if shutil.which('systemctl') is None or shutil.which('systemd-run') is None:
        return []
    tag = _scope_tag_for(project_root)
    pattern = f'df-verify-{tag}-*.scope'
    listing = ''
    with contextlib.suppress(Exception):
        proc = await asyncio.create_subprocess_exec(
            'systemctl', '--user', 'list-units', '--all', '--plain',
            '--no-legend', pattern,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), 30)
        listing = out.decode('utf-8', 'replace') if out else ''
    keep = re.compile(rf'^df-verify-{re.escape(tag)}-.*\.scope$')
    reaped: list[str] = []
    survivors: list[str] = []
    for line in listing.splitlines():
        parts = line.split()
        if not parts:
            continue
        unit = parts[0]
        # Defensive re-filter: never stop a name outside this project's tag,
        # regardless of any surprise in systemctl glob semantics.
        if not keep.match(unit):
            continue
        # `_kill_cgroup_scope` already suppresses every systemctl failure
        # internally and never raises an Exception, so no outer suppress is
        # needed here (it would be dead code).  CONFIRM the reap before
        # reporting it: a best-effort kill can leave a genuinely un-killable
        # scope alive, so only count a unit as reaped once it is verified gone
        # and surface any survivor LOUDLY instead of over-reporting success.
        await _kill_cgroup_scope(unit)
        if await _scope_is_gone(unit):
            reaped.append(unit)
        else:
            survivors.append(unit)
    if survivors:
        logger.warning(
            'Verify-scope reaper: %d leftover scope(s) survived the reap '
            'attempt and are STILL ACTIVE (manual cleanup may be needed): %s',
            len(survivors),
            ', '.join(survivors),
        )
    return reaped


# Environment variables that activate a *specific* Python virtualenv / uv
# project.  The orchestrator runs under `uv run --project orchestrator`, which
# activates dark-factory/.venv and exports these into our process env.  If they
# leak into a TARGET project's verify/build/test subprocess, the target's `uv`
# resolves THIS venv instead of its own and a target `uv sync` writes the
# target's deps into the orchestrator's runtime interpreter — the 2026-05-29
# ghost-venv incident (autopilot-video's torch/insightface stack got synced into
# dark-factory/.venv, flipping it 3.13->3.12 and deleting the live interpreter's
# stdlib out from under the running orchestrator, which then hit FileNotFoundError
# every scheduler cycle and stopped dispatching).
#
# Denylist, NOT allowlist: reify's cargo verify depends on a broad, evolving set
# of toolchain/sccache/jobserver vars (RUSTC_WRAPPER, CARGO_*, the jobserver
# FIFO, ...); an allowlist would silently break Rust verify the first time a new
# var is needed.  We remove ONLY the python-env-selection vars below plus the
# orchestrator's own ORCH_* control-plane namespace (see _ORCH_ENV_PREFIX) —
# the vars that cause a leak — and pass everything else through untouched.
_VENV_ISOLATION_KEYS: frozenset[str] = frozenset({
    'VIRTUAL_ENV',
    'UV_PROJECT_ENVIRONMENT',
    'UV_PROJECT',
    'UV_ACTIVE',
    # Fix 3 may run the orchestrator unit with --frozen; UV_FROZEN / UV_NO_SYNC
    # must NOT leak to a target, which has to stay free to sync its own deps.
    'UV_FROZEN',
    'UV_NO_SYNC',
    # `uv run` (our ExecStart) sets UV_RUN_RECURSION_DEPTH in our env to guard
    # against runaway nested invocations.  A target's verify is a logically
    # independent uv invocation tree, so it must start that counter fresh rather
    # than inherit our depth — otherwise a uv-based target (dark-factory /
    # autopilot-video) would `uv run` "pre-loaded" at our depth.
    'UV_RUN_RECURSION_DEPTH',
    'CONDA_PREFIX',
    'CONDA_DEFAULT_ENV',
    'PYTHONHOME',
})

# The orchestrator's own control-plane env namespace.  ``OrchestratorConfig`` is
# a pydantic-settings ``BaseSettings`` whose ``env_settings`` source reads the
# ENTIRE ``ORCH_`` prefix as config overrides, so ANY ambient ``ORCH_*`` var
# must NOT leak into a TARGET verify subprocess.  In particular ``load_config``
# stamps ``os.environ['ORCH_CONFIG_PATH']`` in-process; if that leaks, a
# snapshot-era env-sensitive test (e.g. an ``OrchestratorConfig()`` defaults
# assertion frozen before main's autouse ``_isolate_orch_config`` hardening)
# loads the PRODUCTION ``dark-factory-orchestrator.yaml`` and fails — falsifying
# the eval metric collector's ``tests_pass`` on every cell (task 2957 / the RCA
# doc ``plans/eval-metric-collector-orch-config-leak-rca-2026-07-22.md``).  We
# scrub the WHOLE prefix (not just ORCH_CONFIG_PATH) so a future ORCH_* var
# can't reintroduce the same leak class.
_ORCH_ENV_PREFIX: str = 'ORCH_'


def _strip_venv_bin_from_path(path: str | None, venv: str | None) -> str | None:
    """Drop the active venv's ``bin`` directory from a PATH string.

    ``uv run`` prepends ``$VIRTUAL_ENV/bin`` to PATH in the orchestrator
    process, so removing ``VIRTUAL_ENV`` alone is insufficient: the venv's bin
    dir is still first on PATH and a target's bare ``python`` / ``uv`` / ``pip``
    would still resolve into the orchestrator venv.  Remove exactly that one
    component (matched as ``<venv>/bin`` via normpath); every other PATH entry
    (cargo, sccache, system bins) keeps its order.  Returns *path* unchanged
    when *path* or *venv* is falsy.
    """
    if not path or not venv:
        return path
    venv_bin = os.path.normpath(os.path.join(venv, 'bin'))
    kept = [
        p for p in path.split(os.pathsep)
        if p and os.path.normpath(p) != venv_bin
    ]
    return os.pathsep.join(kept)


def _target_subprocess_env(extra: dict[str, str] | None) -> dict[str, str]:
    """Build the subprocess env for a TARGET project's verify/build/test spawn.

    Starts from ``os.environ`` minus the orchestrator's own venv/uv activation
    vars (``_VENV_ISOLATION_KEYS``), minus the orchestrator's ``ORCH_*``
    control-plane namespace (``_ORCH_ENV_PREFIX`` — so an ambient, leaked
    ``ORCH_CONFIG_PATH`` can't make a snapshot-era test load the production
    config; task 2957), and minus the venv ``bin`` dir on PATH, so the target's
    toolchain resolves the target's OWN .venv.  Then injects
    ``PYTHONUNBUFFERED=1`` (the partial-log invariant — see ``_run_cmd``) and
    finally overlays *extra* (the caller's ``_resolve_verify_env`` result:
    ``DF_VERIFY_ROLE`` plus reify's ``RUSTC_WRAPPER`` / ``CARGO_*`` / jobserver
    vars) LAST, so target-supplied vars always win — an ``ORCH_*`` var a caller
    intentionally injects therefore survives the scrub.
    """
    venv = os.environ.get('VIRTUAL_ENV')
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in _VENV_ISOLATION_KEYS and not k.startswith(_ORCH_ENV_PREFIX)
    }
    stripped_path = _strip_venv_bin_from_path(env.get('PATH'), venv)
    if stripped_path is not None:
        env['PATH'] = stripped_path
    env['PYTHONUNBUFFERED'] = '1'
    if extra:
        env.update(extra)
    return env


@dataclass(frozen=True)
class ClockStopConfig:
    """Configuration for the clock-stop verify timeout seam (task 1916).

    Bundles the marker strings and timing limits used by the marker-aware
    streamed loop in ``_run_cmd``.  Constructed from ``OrchestratorConfig``
    fields in ``_run_or_skip_timed`` and passed as ``clock_stop=...``.

    Modelled on the ``ModuleConfig`` dataclass style (plain frozen dataclass
    holding verify-related overrides, defined next to its consumer).

    Fields
    ------
    marker_stop : str
        Substring matched against complete output lines to enter STOPPED state.
    marker_heartbeat : str
        Substring matched to reset the heartbeat-idle deadline while STOPPED.
    marker_start : str
        Substring matched to resume RUNNING state (wall-clock resumes).
    heartbeat_idle_max : float
        Max seconds between heartbeats (or after STOP) before the idle backstop
        kills the subprocess.  Must be > 0.
    max_total_secs : float
        Max cumulative seconds in STOPPED state across all stop/start cycles.
        0 means unlimited (no total cap).
    """

    marker_stop: str
    marker_heartbeat: str
    marker_start: str
    heartbeat_idle_max: float
    max_total_secs: float = 0.0


class _ScopeKw(TypedDict, total=False):
    """Keyword arguments for the cgroup-scope path in ``_run_cmd``."""

    use_cgroup_scope: bool
    scope_tag: str


class _ClockKw(TypedDict, total=False):
    """Keyword arguments for the clock-stop path in ``_run_cmd``."""

    clock_stop: ClockStopConfig | None


def _match_clock_marker(line: str, cfg: ClockStopConfig) -> str | None:
    """Return 'stop', 'heartbeat', or 'start' if *line*, after stripping leading
    whitespace, STARTS WITH the respective configured marker string; else None.

    Matching is ANCHORED to line start (``line.lstrip().startswith(marker)``), not
    substring-anywhere.  The reify emitter (``scripts/lib_clock_stop.sh``) always
    writes a marker at column 0 as the first token of its own line
    (``@@REIFY_CLOCK_STOP@@ reason=… pid=…``), so anchoring matches every genuine
    marker while ignoring the token wherever it appears MID-LINE — e.g. quoted in a
    test's assertion prose (``PASS: … stderr contains @@REIFY_CLOCK_STOP@@ …``).

    Why anchored, not substring (reify task 4998 / esc-4791-52): a per-task verify
    runs ``run_all.sh`` (``--include-infra``), whose ~100 infra tests exercise the
    clock-stop machinery and print the marker tokens as assertion text.  Under the
    old substring match those quotes were misread as REAL STOP/START transitions,
    leaving this parser wrongly STOPPED going into the heavy ``cargo nextest``
    compile; a >180s silent native-kernel link gap then tripped the heartbeat-idle
    backstop and false-killed a healthy, code-complete compile.  Anchoring is the
    wire-contract fix that defuses quoted-in-prose pollution for every project.
    (reify ships the complementary Layer-1 fix — run_all.sh neutralizes the tokens
    it re-emits — so either half alone closes the hole.)

    Tolerant of trailing fields (reason=…/waited=…/pid=…) and of leading
    WHITESPACE only — deliberately NOT of an arbitrary leading log/harness prefix
    (the tightening that removes the pollution; if a project ever needs prefix
    support, add a configurable strip pattern rather than reinstating
    substring-anywhere).  A hypothetical genuine marker that fails to match
    degrades gracefully to the wall-clock budget — never a false kill.

    The three marker strings are guaranteed pairwise non-substrings (enforced by
    the ``OrchestratorConfig`` validator when enabled), which a fortiori guarantees
    none is a PREFIX of another — so the stop→heartbeat→start priority order below
    cannot misclassify one anchored marker as another.

    Parameters
    ----------
    line : str
        A complete, newline-stripped output line from the verify subprocess.
    cfg : ClockStopConfig
        The active clock-stop configuration.

    Returns
    -------
    'stop' | 'heartbeat' | 'start' | None
    """
    stripped = line.lstrip()
    if stripped.startswith(cfg.marker_stop):
        return 'stop'
    if stripped.startswith(cfg.marker_heartbeat):
        return 'heartbeat'
    if stripped.startswith(cfg.marker_start):
        return 'start'
    return None


async def _run_cmd(
    cmd: str,
    cwd: Path,
    timeout: float,
    env: dict[str, str] | None = None,
    log_path: 'Path | None' = None,
    *,
    use_cgroup_scope: bool = False,
    scope_tag: str = '',
    clock_stop: ClockStopConfig | None = None,
) -> tuple[int, str, bool]:
    """Run a shell command, return (returncode, combined output, timed_out).

    When *env* is non-None, it is merged on top of ``os.environ`` and passed
    to the subprocess so callers can inject build accelerators like
    ``RUSTC_WRAPPER=sccache`` without mutating the parent process's env.

    When *use_cgroup_scope* is True and ``systemd-run`` is available, the
    command is launched inside a transient systemd ``--user --scope`` (its own
    cgroup) so a timeout/cancel can reap the WHOLE subtree by cgroup
    (``_kill_cgroup_scope``), regardless of process-group escapes — e.g. an
    inner GNU ``timeout`` that setpgid'd cargo into a separate group, which
    defeats the ``killpg``-on-spawn-pgid fallback and was the leak that let a
    defeated post-merge verify strand live ``cargo`` for up to 30 minutes.
    Falls back to the plain ``start_new_session`` + ``killpg`` path when the
    flag is off or ``systemd-run`` is missing, so the default behaviour and the
    existing test suite are unchanged.

    When *log_path* is provided, subprocess output is streamed (read in 4 KiB
    chunks and flushed) to that file as it arrives, so a timeout-killed child
    leaves the partial buffer on disk instead of producing a 0-byte file.  The
    accumulated buffer is also returned via the second tuple slot, identical
    to the legacy ``proc.communicate()`` contract.  When *log_path* is None
    no file is created.

    When *clock_stop* is provided AND *log_path* is not None, the streamed
    path uses a marker-aware state machine instead of the single
    ``asyncio.wait_for``.  The loop recognises the configured STOP /
    HEARTBEAT / START marker family and EXCLUDES the declared admission-wait
    span from *timeout* (the wall-clock deadline is shifted forward by the
    stopped duration on START).  A heartbeat-idle backstop fires if no
    heartbeat arrives within ``clock_stop.heartbeat_idle_max`` seconds of
    the last STOP or HEARTBEAT.  Any deadline breach raises ``TimeoutError``,
    which the existing kill path catches (``timed_out=True → infra_timeout``).

    ``PYTHONUNBUFFERED=1`` is unconditionally injected into the subprocess env
    so that python children (pytest, ruff, pyright via uv) flush their stdout
    per-line — necessary for the partial-log invariant under heavy buffering.
    """
    # PYTHONUNBUFFERED is the cheap-but-decisive lever: without it pytest's
    # progress dots stay in stdio buffers and never reach our streaming loop,
    # so a hanging subprocess produces an opaque ``Command timed out after …``
    # cause hint with no actionable signal.
    # Build the target subprocess env via the venv-isolation scrub so the
    # TARGET's `uv` resolves the TARGET's .venv, never dark-factory/.venv (the
    # 2026-05-29 ghost-venv coupling).  The scrub strips the orchestrator's
    # venv/uv activation vars + the venv bin dir from PATH, sets
    # PYTHONUNBUFFERED, and reapplies the caller overlay (`env`) LAST so reify's
    # RUSTC_WRAPPER/CARGO_*/jobserver vars and DF_VERIFY_ROLE always win.
    subprocess_env: dict[str, str] = _target_subprocess_env(env)

    proc = None
    pgid: int | None = None
    scope_unit: str | None = None
    if use_cgroup_scope and shutil.which('systemd-run') is not None:
        scope_unit = _verify_scope_name(scope_tag)
    # Populated by the clock-stop loop before raising TimeoutError so the except
    # handler can emit a richer message (actual wall time + which deadline fired).
    _cs_timeout_msg: list[str] = []
    try:
        if scope_unit is not None:
            # Launch inside a transient --user scope (its own cgroup) so a
            # timeout/cancel can reap the WHOLE subtree by cgroup.  --scope runs
            # bash as a direct child of systemd-run, inheriting our cwd/env and
            # forwarding stdio to our pipe; --collect auto-removes the scope when
            # it exits (no unit leak on the normal-completion path).
            proc = await asyncio.create_subprocess_exec(
                'systemd-run', '--user', '--scope', '--quiet', '--collect',
                f'--unit={scope_unit}',
                '/bin/bash', '-c', cmd,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=subprocess_env,
                start_new_session=True,
            )
        else:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                executable='/bin/bash',
                env=subprocess_env,
                start_new_session=True,
            )
        # Capture pgid at spawn; start_new_session guarantees pgid == pid.
        pgid = proc.pid

        if log_path is None:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            rc = proc.returncode if proc.returncode is not None else 1
            return rc, stdout.decode(errors='replace'), False

        # Streamed path: chunked read + per-chunk flush so the kill on timeout
        # cannot strand the partial output in kernel buffers.  The ``with``
        # block guarantees the FD closes even when wait_for/CancelledError
        # unwinds through the outer try; the close happens before the except
        # handler runs, which is exactly what we want.
        buf = bytearray()
        with open(log_path, 'wb') as log_fh:
            assert proc.stdout is not None
            if clock_stop is None:
                # ── Legacy streamed path (unchanged) ────────────────────────
                async def _stream() -> None:
                    assert proc is not None and proc.stdout is not None
                    while True:
                        chunk = await proc.stdout.read(4096)
                        if not chunk:
                            break
                        buf.extend(chunk)
                        log_fh.write(chunk)
                        log_fh.flush()
                    await proc.wait()

                await asyncio.wait_for(_stream(), timeout=timeout)
            else:
                # ── Marker-aware clock-stop loop (task 1916) ────────────────
                # State machine: RUNNING enforces a wall-clock deadline that
                # is shifted forward by each STOP→START span so admission
                # waits are excluded.  STOPPED enforces a heartbeat-idle
                # backstop; any deadline breach raises TimeoutError → existing
                # kill path → timed_out=True.  Raw bytes are written verbatim
                # to log_fh/buf (byte-identical on-disk log).  A SEPARATE
                # line_buf decodes complete lines for marker scanning without
                # disturbing the raw stream.
                _CS_RUNNING = 'running'
                _CS_STOPPED = 'stopped'
                state = _CS_RUNNING
                t0 = time.monotonic()
                # Wall-clock deadline: start + timeout, shifted forward on each
                # STOP→START transition by the duration of the stopped span.
                deadline = t0 + timeout
                stop_entered: float = 0.0
                idle_deadline: float = 0.0
                # Cumulative stopped time across all completed stop/start cycles
                # (step-10: max_total_secs cap).
                total_stopped: float = 0.0
                line_buf = bytearray()
                # Human-readable reason for the binding deadline (overwritten each
                # iteration; captured in _cs_timeout_msg before raising TimeoutError
                # so the except handler can emit an informative message with the
                # actual wall time and which limit fired).
                _cs_reason = f'wall-clock budget ({timeout:.0f}s)'

                while True:
                    now = time.monotonic()
                    if state == _CS_RUNNING:
                        read_timeout = deadline - now
                        _cs_reason = (
                            f'wall-clock budget ({timeout:.0f}s), '
                            f'wall time {now - t0:.1f}s'
                        )
                    else:  # _CS_STOPPED
                        read_timeout = idle_deadline - now
                        _cs_reason = (
                            f'heartbeat-idle backstop ({clock_stop.heartbeat_idle_max:.0f}s), '
                            f'wall time {now - t0:.1f}s'
                        )
                        # max_total_secs cap (step-10): when > 0, also bound
                        # the read timeout by the remaining total-stopped budget
                        # so we don't stay STOPPED past the cumulative cap.
                        if clock_stop.max_total_secs > 0:
                            cumulative = total_stopped + (now - stop_entered)
                            remaining_total = clock_stop.max_total_secs - cumulative
                            if remaining_total < read_timeout:
                                read_timeout = remaining_total
                                _cs_reason = (
                                    f'max-total-stopped cap ({clock_stop.max_total_secs:.0f}s), '
                                    f'wall time {now - t0:.1f}s'
                                )

                    if read_timeout <= 0:
                        _cs_timeout_msg.append(_cs_reason)
                        raise TimeoutError()

                    try:
                        chunk = await asyncio.wait_for(
                            proc.stdout.read(4096),
                            timeout=read_timeout,
                        )
                    except TimeoutError:
                        _cs_timeout_msg.append(_cs_reason)
                        raise

                    if not chunk:
                        # EOF: process finished; wait for exit code within
                        # remaining wall-clock budget (fast in practice).
                        now = time.monotonic()
                        wait_budget = max(5.0, deadline - now)
                        await asyncio.wait_for(proc.wait(), timeout=wait_budget)
                        break

                    # Write raw bytes verbatim (byte-identical on-disk log).
                    buf.extend(chunk)
                    log_fh.write(chunk)
                    log_fh.flush()

                    # Scan complete lines for clock-stop markers.  Split on
                    # '\n' in one pass (O(n) per chunk) rather than the
                    # index+re-slice loop that is O(n^2) when a chunk contains
                    # many newlines.  The last element is a partial line (no
                    # trailing newline yet) and becomes the new line_buf.
                    line_buf.extend(chunk)
                    parts = line_buf.split(b'\n')
                    line_buf = bytearray(parts[-1])
                    for line_bytes in parts[:-1]:
                        line = line_bytes.decode(errors='replace')
                        marker = _match_clock_marker(line, clock_stop)
                        if marker is None:
                            continue

                        now = time.monotonic()
                        if state == _CS_RUNNING:
                            if marker == 'stop':
                                state = _CS_STOPPED
                                stop_entered = now
                                idle_deadline = now + clock_stop.heartbeat_idle_max
                        else:  # _CS_STOPPED
                            if marker in ('heartbeat', 'stop'):
                                # Heartbeat (or duplicate STOP) resets idle backstop.
                                idle_deadline = now + clock_stop.heartbeat_idle_max
                            elif marker == 'start':
                                # Resume: shift wall-clock deadline forward by
                                # the duration of this stopped span; accumulate
                                # total_stopped for the max_total_secs cap.
                                stopped_duration = now - stop_entered
                                total_stopped += stopped_duration
                                deadline += stopped_duration
                                state = _CS_RUNNING

        rc = proc.returncode if proc.returncode is not None else 1
        return rc, buf.decode(errors='replace'), False
    except TimeoutError:
        # cgroup kill (primary, reaps process-group escapes) then killpg backstop.
        if scope_unit is not None:
            await _kill_cgroup_scope(scope_unit)
        if proc is not None and pgid is not None:
            await terminate_process_group(proc, pgid, grace_secs=5.0)
        if _cs_timeout_msg:
            # Clock-stop path: include actual wall time and which deadline fired
            # (idle backstop / wall-clock budget / max-total cap) so infra_timeout
            # incidents are distinguishable in the verify log.
            return 1, f'Command clock-stop timed out ({_cs_timeout_msg[0]}): {cmd}', True
        return 1, f'Command timed out after {timeout}s: {cmd}', True
    except asyncio.CancelledError:
        if scope_unit is not None:
            await _kill_cgroup_scope(scope_unit)
        if proc is not None and pgid is not None:
            await terminate_process_group(proc, pgid, grace_secs=5.0)
        raise
    except Exception as e:
        return 1, f'Command failed: {e}', False


_CHAIN_BUDGET_EXHAUSTED = 'chain wall-clock budget exhausted before this segment started'


async def _run_segmented(
    segments: 'list[ChainSegment]',
    *,
    run_one: 'Callable[[str, Path, float, str], Awaitable[tuple[int, str, bool]]]',
    worktree: Path,
    budget_secs: float,
    now: 'Callable[[], float]' = time.monotonic,
) -> 'tuple[int, str, bool, list[dict]]':
    """Run EVERY segment of a decomposed `&&` chain — no short-circuit.

    **Running every segment, rather than stopping at the first red, is the
    whole point of task 3338.** A later "optimisation" that reintroduces the
    short-circuit is not a speed-up; it is a reintroduction of the defect
    esc-3062-2 reports, and
    ``test_verify_segmented_fallback.TestRunSegmentedRunsEverySegment`` is what
    should catch it.

    The defect: the fallback verify runs the fleet-wide `&&` chain as one
    ``/bin/bash -c`` string, so an unrelated earlier subproject's red makes the
    SHELL skip every later clause — including the one a task's own assigned
    files live in. The orchestrator sees a single rc and cannot tell "skipped"
    from "passed", so the triaging agent's job becomes proving an unrelated
    red is unrelated instead of reading its own result.

    Returns ``(rc, output, timed_out, segments)`` where:

    * ``rc`` is the FIRST non-zero segment rc, else 0 — deliberately the same
      number the shell's `&&` chain would have returned, so every downstream rc
      consumer keeps reading what it read before. Running the later segments
      buys information, never leniency.
    * ``timed_out`` is true if a segment ACTUALLY timed out at a point the old
      `&&` chain would have reached (nothing red before it), or — only when no
      segment produced a genuine red — if the shared budget left some segment
      unrun. Both halves are conditional on there being no earlier red, because
      raising the flag on top of a real failure relabels that failure
      `infra_timeout`; see the loop's ``chain_broken`` guard and the comment on
      the return statement.
    * ``segments`` is one flat JSON-native dict per segment
      (index/label/cwd/cmd/status/rc/timed_out/duration_secs/skip_reason),
      which rides on ``CheckRun.segments`` into the persisted
      ``.task/verify/attempt-N[.<prefix>].summary.json`` (via ``to_dict`` ->
      ``_build_summary_payload``'s per-command entries) and into the aggregated
      ``.log`` text.

    *run_one* is INJECTED rather than calling ``_run_cmd`` directly, so this
    aggregator is unit-testable with a recording fake and spawns no
    subprocesses. Its production binding wraps ``_run_cmd`` with the caller's
    per-segment cpu-governance, nice prefix and streamed log path.
    """
    results: list[dict] = []
    first_nonzero = 0
    any_timed_out = False
    any_not_run = False
    # True once a segment has produced the result that would have STOPPED the
    # old `&&` chain (non-zero rc, or a timeout). Everything after that point is
    # information the shell never had; see the `any_timed_out` guard below.
    chain_broken = False
    blocks: list[str] = []
    deadline = now() + budget_secs

    for index, segment in enumerate(segments, start=1):
        remaining = deadline - now()
        if remaining <= 0:
            # Budget gone. Record this segment and every one after it as
            # NOT RUN without spawning anything: `rc=None` (never 0) is the
            # unconflatable encoding — a segment that never ran must be
            # structurally impossible to read as a pass.
            any_not_run = True
            results.append({
                'index': index,
                'label': segment.label,
                'cwd': segment.cwd_rel,
                'cmd': segment.command,
                'status': 'not_run',
                'rc': None,
                'timed_out': False,
                'duration_secs': 0.0,
                'skip_reason': _CHAIN_BUDGET_EXHAUSTED,
            })
            blocks.append(_segment_output_block(results[-1], len(segments), ''))
            continue
        started = now()
        rc, out, seg_timed_out = await run_one(
            segment.command,
            worktree / segment.cwd_rel,
            remaining,
            segment.label,
        )
        duration = now() - started
        if seg_timed_out:
            status = 'timed_out'
        elif rc != 0:
            status = 'failed'
        else:
            status = 'passed'
        if not chain_broken:
            # A timeout only counts as a GENUINE timeout when the old `&&`
            # chain would actually have reached this segment — i.e. nothing
            # before it already went red. Past the first red, every segment is
            # running on a deadline the earlier segments have already eaten
            # into (`remaining = deadline - now()`, never the full budget), so
            # the dominant way a late segment reports `timed_out=True` is
            # SHARED-BUDGET exhaustion mid-flight, not a hang of its own. That
            # is the same fact `not_run` encodes one segment later, and it must
            # be treated the same way: as budget exhaustion, which never
            # outranks a real red. Letting it through here would reintroduce
            # exactly the `infra_timeout` relabelling the return-statement
            # comment below exists to prevent — through the unconditional flag
            # rather than the synthetic one. Pinned by
            # `test_a_red_segment_then_a_deadline_bound_timeout_is_a_test_failure`.
            any_timed_out = any_timed_out or seg_timed_out
        if seg_timed_out or rc != 0:
            chain_broken = True
        if rc != 0 and first_nonzero == 0:
            first_nonzero = rc
        results.append({
            'index': index,
            'label': segment.label,
            'cwd': segment.cwd_rel,
            'cmd': segment.command,
            'status': status,
            'rc': rc,
            'timed_out': seg_timed_out,
            'duration_secs': duration,
            'skip_reason': None,
        })
        blocks.append(_segment_output_block(results[-1], len(segments), out))

    # A genuine red outranks the synthetic not-run rc: the cause_hint a
    # triaging agent needs is the real failure, not the budget. But a
    # green-so-far run with unrun segments is NEVER green — reporting 0 would
    # claim a fleet-wide pass on the strength of whatever happened to fit.
    rc_total = first_nonzero or (1 if any_not_run else 0)
    # The SYNTHETIC not-run timeout flag is raised only when there is no genuine
    # red to report. `verify_classify.classify_failure`'s guard 2 (`if
    # timed_out: return INFRA_TIMEOUT`) wins over EVERY output pattern, so
    # raising it unconditionally on budget exhaustion would relabel a real test
    # failure as a timeout — and `VerifyAttempt.pure_timeout_failure` (whose
    # `all(c.rc == 0 or c.timed_out ...)` clause a rc!=0 + timed_out=True check
    # satisfies) would then re-run the whole thing `max_retries` times at a full
    # budget each and route it to infra-hold instead of to the debugger.
    #
    #   * A genuine non-timeout failure keeps its `test_failure` classification.
    #     Nothing is silently greened by dropping the flag: the unrun segments
    #     stay fully visible via `CheckRun.segments` (`status='not_run'`,
    #     `rc=None`, non-empty `skip_reason`), via the NOT RUN output blocks and
    #     roster lines, and via run_verification's `| segments not run: <labels>`
    #     cause_hint suffix. Only the CLASSIFICATION changes, and only toward
    #     the truth.
    #   * A green-so-far-but-truncated run still reports rc!=0 AND
    #     timed_out=True — that is the case this synthetic flag exists for, and
    #     it classifies as `infra_timeout` (retryable) exactly as a single-chain
    #     timeout does today, leaving the retry/env-recovery machinery untouched.
    #   * A segment that ACTUALLY hit its wall clock BEFORE anything went red
    #     still forces `timed_out=True` through `any_timed_out` — including the
    #     lone-hang case (segment 1 wedges on the full budget), which stays
    #     `infra_timeout` exactly as the single `&&` chain reported it. Only a
    #     timeout in a segment the shell would never have reached is folded
    #     away, because past the first red that flag reports the shared budget,
    #     not a hang; see the `chain_broken` guard in the loop above.
    #
    # Load-bearing, not an edge case: removing the `&&` short-circuit (the whole
    # point of task 3338) makes budget exhaustion strictly MORE likely, since
    # all 8 segments now always run where the shell previously stopped at the
    # first red — and the committed config's own measured table already records
    # five of seven segments costing 1838.60s. So red-plus-exhausted is the
    # COMMON shape of a red fallback verify. Under the old `&&` chain the shell
    # short-circuited at the red and the check finished fast with
    # rc=1/timed_out=False/`test_failure`; without this conditional, segmenting
    # would be a regression against that baseline. Pinned as a pair by
    # test_verify_segmented_fallback's
    # `test_a_red_segment_before_the_deadline_still_wins_the_rc` and
    # `test_a_green_so_far_run_whose_tail_is_unrun_still_reports_timed_out`.
    timed_out_total = any_timed_out or (any_not_run and first_nonzero == 0)
    output = '\n'.join(_segment_roster(results) + blocks)
    return rc_total, output, timed_out_total, results


# Roster status words, chosen to be INERT to `_extract_cause_hint`'s ladder and
# `verify_classify.classify_failure`'s scanners: no `FAILED`, no `ERROR`, no
# `error:`, no tool-specific token. Paired with the `#` line prefix, that is
# what stops the roster from shadowing the genuine failing segment's hint.
# `test_verify_segmented_fallback.TestRunSegmentedRoster` pins it — an edit
# that "improves" this wording MUST re-run
# `test_roster_is_inert_to_the_cause_hint_and_category_scanners`.
_ROSTER_STATUS_WORDS = {
    'passed': 'ok',
    'failed': 'RED',
    'timed_out': 'TIMED OUT',
    'not_run': 'NOT RUN',
}


def _segment_roster(results: 'list[dict]') -> 'list[str]':
    """One `#`-prefixed line per segment: index, label, status, rc, duration.

    Leads the combined output so a reader hits the whole picture FIRST, rather
    than after scrolling six subprojects of pytest chatter. This is what turns
    the triaging agent's job from "prove this unrelated red is unrelated" into
    "read your own segment's line" — the human-time cost esc-3062-2 is about.
    """
    total = len(results)
    lines = []
    for entry in results:
        word = _ROSTER_STATUS_WORDS.get(entry['status'], entry['status'])
        if entry['status'] == 'not_run':
            detail = f' — {entry["skip_reason"]}'
        else:
            detail = f' (rc={entry["rc"]}, {entry["duration_secs"]:.1f}s)'
        lines.append(
            f'# [{entry["index"]}/{total}] {entry["label"]}  cwd={entry["cwd"]}  '
            f'{word}{detail}',
        )
    return lines


def _segment_output_block(entry: dict, total: int, out: str) -> str:
    """One delimited output block for a segment, headed by its own facts.

    The header names index/label/cwd/rc so a reader scrolling a concatenated
    multi-subproject log can always tell which subproject the surrounding lines
    came from — the thing a single `&&`-chained blob cannot say.

    A `not_run` segment gets an explicit NOT RUN body saying its result is
    UNKNOWN — not a pass. Its wording is deliberately inert to
    ``_extract_cause_hint``'s ladder and ``classify_failure``'s scanners (no
    `FAILED`, no `^error: `, no `^ERROR: `), so it can never shadow the real
    failing segment's hint.
    """
    header = (
        f'===== segment {entry["index"]}/{total} [{entry["label"]}] '
        f'cwd={entry["cwd"]} rc={entry["rc"]} status={entry["status"]} '
        f'({entry["duration_secs"]:.1f}s) ====='
    )
    if entry['status'] == 'not_run':
        body = (
            f'  NOT RUN: {entry["skip_reason"]}.\n'
            '  This segment produced no result. Its outcome is unknown — '
            'unknown is not a pass.'
        )
        return f'{header}\n{body}'
    return f'{header}\n{out}'


# Marker file that records a worktree has completed at least one non-timeout verify.
_VERIFY_WARM_MARKER = 'verify_warmed'


def _warm_marker_name(module_prefix: str | None) -> str:
    """Return the marker filename for the given module prefix.

    When *module_prefix* is ``None`` the shared worktree marker is used
    (``verify_warmed``).  When a prefix is provided the marker is scoped to
    that subproject (``verify_warmed_<safe_prefix>``), preventing a successful
    subproject A from hiding a cold-build need for concurrently-run subproject B.
    Path separators and spaces are replaced with underscores.
    """
    if module_prefix is None:
        return _VERIFY_WARM_MARKER
    safe = module_prefix.replace('/', '_').replace(' ', '_')
    return f'{_VERIFY_WARM_MARKER}_{safe}'


def _is_verify_cold(worktree: Path, module_prefix: str | None = None) -> bool:
    """Return True when *worktree* has never completed a non-timeout verify.

    A worktree is considered cold when its ``.task/`` scratch directory exists
    but the ``verify_warmed`` marker inside it does not.  Paths without
    ``.task/`` (e.g., the project root used by review checkpoints) are treated
    as warm so that review-checkpoint verifies always use the standard timeout.

    When *module_prefix* is provided the check uses a per-subproject marker
    (``verify_warmed_{prefix}``), so one subproject completing successfully
    does not falsely warm-classify a concurrently-run sibling subproject.
    """
    task_dir = worktree / '.task'
    if not task_dir.is_dir():
        return False
    return not (task_dir / _warm_marker_name(module_prefix)).exists()


def _mark_verify_warm(worktree: Path, module_prefix: str | None = None) -> None:
    """Atomically mark *worktree* as warm by touching the verify_warmed marker.

    No-op when ``.task/`` is absent — we never create the scratch directory
    from within the verify path.  Idempotent (``exist_ok=True``).

    When *module_prefix* is provided the per-subproject marker is touched
    (``verify_warmed_{prefix}``) rather than the shared worktree marker.
    """
    task_dir = worktree / '.task'
    if not task_dir.is_dir():
        return
    marker_path = task_dir / _warm_marker_name(module_prefix)
    try:
        marker_path.touch(exist_ok=True)
    except OSError as exc:
        if _is_infra_oserror(exc):
            raise VerifyInfraError(phase='warm_marker', errno=exc.errno) from exc
        # Non-infra OSError — the warm marker is advisory; a passing verify
        # must not be sunk by a failed marker write.  Log and swallow.
        # Precedent: _write_run_log / _persist_attempt_logs do the same.
        logger.warning(
            'verify warm marker write failed (non-infra, swallowed): %s: %s',
            marker_path,
            exc,
        )
        return
    logger.debug('verify warm marker set: %s', marker_path)


def _resolve_verify_timeout(
    config: OrchestratorConfig,
    module_config: ModuleConfig | None,
    *,
    is_cold: bool,
    is_merge_verify: bool = False,
) -> float:
    """Return the effective per-command verify timeout.

    When *is_cold* is False (warm cache), the warm timeout is returned:
    ``module_config.verify_command_timeout_secs`` takes precedence over
    ``config.verify_command_timeout_secs``.

    When *is_cold* is True (first verify in a fresh worktree), the cold
    timeout is resolved via the cascade:
      0. ``config.merge_verify_cold_command_timeout_secs`` (if set AND
         *is_merge_verify* is True) — merge-verify-specific cold budget;
         wins before the per-module and general cold knobs.
      1. ``module_config.verify_cold_command_timeout_secs`` (if set)
      2. ``config.verify_cold_command_timeout_secs`` (if set)
      3. The warm timeout computed above (fallback when cold knob is unset
         at every level — preserves existing behaviour for deployments that
         don't configure the cold window).

    *is_merge_verify* only affects the cold track; for warm resolves (or
    when ``config.merge_verify_cold_command_timeout_secs`` is None) the
    resolver falls through to the existing cascade unchanged.
    """
    # Warm track: module override wins over top-level.
    warm: float
    if module_config is not None and module_config.verify_command_timeout_secs is not None:
        warm = module_config.verify_command_timeout_secs
    else:
        warm = config.verify_command_timeout_secs

    if not is_cold:
        return warm

    # Cold track: merge-verify budget wins first, then cascade module → top → warm.
    if is_merge_verify and config.merge_verify_cold_command_timeout_secs is not None:
        return config.merge_verify_cold_command_timeout_secs
    if module_config is not None and module_config.verify_cold_command_timeout_secs is not None:
        return module_config.verify_cold_command_timeout_secs
    if config.verify_cold_command_timeout_secs is not None:
        return config.verify_cold_command_timeout_secs
    return warm


def _resolve_concurrent_verify(
    config: OrchestratorConfig,
    module_config: ModuleConfig | None,
) -> bool:
    """Return whether test/lint/type should run concurrently.

    Module override wins over top-level config.
    """
    if module_config is not None and module_config.concurrent_verify is not None:
        return module_config.concurrent_verify
    return config.concurrent_verify


def _resolve_sequential_lint_first(
    config: OrchestratorConfig,
    module_config: ModuleConfig | None,
) -> bool:
    """Return whether the merge-role sequential branch should run lint first.

    Module override wins over top-level config.  Consulted only for the
    merge-role sequential lint-first branch in ``run_verification`` (gated
    additionally on ``role == 'merge'`` and ``not concurrent``).
    """
    if module_config is not None and module_config.sequential_lint_first is not None:
        return module_config.sequential_lint_first
    return config.sequential_lint_first


def _resolve_verify_env(
    config: OrchestratorConfig,
    module_config: ModuleConfig | None,
    *,
    role: Literal['merge', 'task', 'background'] = 'task',
) -> dict[str, str]:
    """Return the effective env injected into verify commands.

    Merges ``config.verify_env`` with ``module_config.verify_env``; module
    keys override top-level keys.  The orchestrator-supplied *role* is then
    stamped in as ``DF_VERIFY_ROLE`` and is always authoritative — it overrides
    any ``DF_VERIFY_ROLE`` entry that may appear in static config.
    """
    merged: dict[str, str] = {}
    merged.update(config.verify_env or {})
    if module_config is not None and module_config.verify_env:
        merged.update(module_config.verify_env)
    merged['DF_VERIFY_ROLE'] = role
    return merged


def _resolve_governed_exec_path(
    config: OrchestratorConfig,
    worktree: 'Path | None',
    role: str,
) -> 'str | None':
    """Return the resolved cpu-governed-exec path to apply for *role*, or ``None``.

    Only ``role == 'merge'`` uses the merge-weighted cgroup scope. Returns
    ``None`` (fail-open) when ``config.cpu_governance`` is absent/disabled or
    ``resolved_exec_path`` cannot resolve an executable path (non-executable,
    missing, or *worktree* is ``None``) — the caller (``_govern_cpu_str``)
    then no-ops on a falsy *exec_path*.
    """
    if role != 'merge':
        return None
    gov = getattr(config, 'cpu_governance', None)
    if gov is None or not gov.enabled:
        return None
    return gov.resolved_exec_path(worktree)


def _govern_cpu_str(cmd: 'str | None', exec_path: 'str | None') -> 'str | None':
    """Wrap *cmd* in a cpu-governed-exec.sh invocation via VerifyCmd, when *exec_path* resolves.

    Thin string-level wrapper around ``parse_config_command`` -> ``govern_cpu``
    -> ``render`` (replaces ``_maybe_govern_merge_cmd``'s bash-wrap). Renders as::

        <shlex.quote(exec_path)> --role merge -- /bin/bash -c <shlex.quote(rendered_cmd)>

    so that shell operators (``&&``, ``|``, leading env assignments) in *cmd*
    survive intact inside the merge-weighted cgroup scope.  The inner
    ``/bin/bash -c <quoted>`` makes the whole rendered command a single argv
    payload for ``cpu-governed-exec.sh``.

    Returns *cmd* unchanged when: *cmd* is ``None``; *exec_path* is falsy
    (governance disabled/unresolved — see ``_resolve_governed_exec_path``,
    fail-open); or *cmd* parses OPAQUE (P1 — ``govern_cpu`` no-ops on an
    unparseable command rather than blindly bash-wrapping it).

    Does NOT alter ``_run_cmd``'s signature, the ``use_cgroup_scope`` path, or
    any merge PSI/semaphore bypass.

    **Interaction with verify_use_cgroup_scope**: when both
    ``config.cpu_governance.enabled`` *and* ``config.verify_use_cgroup_scope``
    are ``True``, ``_run_or_skip_timed`` wraps the command here first
    (so ``cpu-governed-exec.sh`` becomes ``argv[0]``), then passes
    ``use_cgroup_scope=True`` to ``_run_cmd``.  ``_run_cmd`` in turn launches
    the already-wrapped command inside a ``systemd-run --user --scope``
    (outer ``df-verify`` scope).  ``cpu-governed-exec.sh``, on its governed
    path, tries to create an *inner* ``systemd-run --user --scope`` scope —
    a nested transient scope inside the outer ``df-verify`` scope.  Nested
    ``--user --scope`` invocations are allowed by systemd (each creates a
    distinct cgroup slice), so this is not a correctness or leak bug; the
    outer scope's cgroup kill still reaps the entire subtree regardless.
    The live reify deployment currently sets ``verify_use_cgroup_scope=False``,
    so this combination does not occur in practice.  ``cpu-governed-exec.sh``
    also has a runtime probe + fail-open, so a nested-scope failure degrades
    gracefully.
    """
    if cmd is None or not exec_path:
        return cmd
    parsed = parse_config_command(cmd)
    governed = govern_cpu(parsed, exec_path)
    if governed is parsed:
        return cmd
    return render(governed)


def _resolve_nice_prefix(config: OrchestratorConfig, role: str) -> list[str]:
    """Return the argv ``nice``/``ionice`` prefix to apply for *role*.

    A non-empty per-role override knob (``verify_admission_nice_{merge,task,
    background}``) wins, ``shlex.split``. Empty (default) defers to T1's
    canonical ``shared.verify_admission.nice_prefix(role)`` tier table —
    ``offline`` and any unrecognized role resolve to ``[]`` (no adjustment).
    """
    overrides = {
        'merge': config.verify_admission_nice_merge,
        'task': config.verify_admission_nice_task,
        'background': config.verify_admission_nice_background,
    }
    override = overrides.get(role, '')
    if override:
        return shlex.split(override)
    return nice_prefix(role)


def _verify_admission_active(config: OrchestratorConfig) -> bool:
    """Whether the verify-admission gate (flock slot + nice tier) is active.

    The single module seam the autouse ``_neutralize_verify_admission``
    conftest fixture (task 2390 pre-1) patches to force every pre-existing
    verify test to run ungated, regardless of ``config.verify_admission_enabled``.
    """
    return config.verify_admission_enabled


_ADMISSION_EXECUTOR_MAX_WORKERS = 64

_admission_executor_singleton: concurrent.futures.ThreadPoolExecutor | None = None


def _admission_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Dedicated thread pool for the admission mkdir + flock poll-wait.

    Kept separate from asyncio's shared default executor (process-wide,
    capped at ``min(32, cpu_count+4)`` and used by unrelated
    ``asyncio.to_thread`` callers throughout the orchestrator) so a burst of
    concurrent task-role verifies polling ``acquire_task_slot`` can never
    starve that unrelated work — or each other's ``slots_dir.mkdir``. Workers
    spend nearly all their time asleep in T1's 0.1s poll loop, so a fixed
    size decoupled from the cpu-bound default-executor formula is cheap.
    Lazily created, never torn down (matches the default executor's
    process-lifetime scope).
    """
    global _admission_executor_singleton
    if _admission_executor_singleton is None:
        _admission_executor_singleton = concurrent.futures.ThreadPoolExecutor(
            max_workers=_ADMISSION_EXECUTOR_MAX_WORKERS,
            thread_name_prefix='df-verify-admission',
        )
    return _admission_executor_singleton


@contextlib.asynccontextmanager
async def _admission_slot(role: str, config: OrchestratorConfig):
    """Async CM around T1's ``shared.verify_admission.acquire_task_slot``.

    Gates only the test leg of a verify (callers decide that; this CM itself
    is role-agnostic and always attempts acquisition uniformly — T1's
    ``acquire_task_slot`` internally no-ops for ``role`` values other than
    ``'task'``/``'background'`` and always yields ``held=False`` immediately
    for them, so ``merge`` can never be starved by ``task`` — C-merge-priority
    is owned entirely by T1, not re-implemented here).

    T1 never creates ``slots_dir`` itself (fails open when absent) and never
    even inspects it for roles it can't acquire for (its own role check
    short-circuits first), so this CM only mkdirs it for roles that actually
    attempt acquisition (``task``/``background``) — leaving ``merge`` (and any
    other role) with no filesystem side effect. The mkdir and the blocking,
    potentially-unbounded ``acquire_task_slot(...).__enter__`` (a synchronous
    flock poll-loop) both run on the dedicated ``_admission_executor`` so the
    wait never blocks the event loop nor contends with unrelated
    ``asyncio.to_thread`` work — a loop-blocking acquire would otherwise stall
    the holder's own subprocess-exit callback from ever firing on this same
    loop, deadlocking cross-verify contention.

    The acquire await is shielded from cancellation (``asyncio.shield``): if
    the awaiting coroutine is cancelled mid-wait (e.g. orchestrator shutdown,
    or a sibling verify's failure cancelling this one via ``asyncio.gather``),
    the worker thread's poll loop keeps running in the background regardless
    — it cannot be interrupted mid-``time.sleep`` — so a bare cancellation
    would otherwise leave a slot acquired-but-never-released if the thread
    goes on to succeed after we stopped waiting. A done-callback releases it
    in that case instead. Release on the normal path (``os.close`` under the
    hood) is synchronous and instant, so it runs directly in ``finally``
    without needing an executor thread.

    Fails open (runs ungated) on any ``OSError`` — most commonly a
    ``slots_dir`` that cannot be created (C-fail-open, mirroring T1's own
    fail-open contract for acquisition itself).
    """
    slots_dir = Path(config.verify_admission_slots_dir)
    n = config.verify_admission_task_slots
    loop = asyncio.get_running_loop()
    executor = _admission_executor()
    cm = None
    try:
        if role in {'task', 'background'}:
            await loop.run_in_executor(
                executor, lambda: slots_dir.mkdir(parents=True, exist_ok=True),
            )
        cm = acquire_task_slot(role, slots_dir=slots_dir, n=n, wait=True)
        enter_future = loop.run_in_executor(executor, cm.__enter__)
        try:
            await asyncio.shield(enter_future)
        except asyncio.CancelledError:
            def _release_if_acquired(fut: 'asyncio.Future[bool]') -> None:
                if cm is None or fut.cancelled() or fut.exception() is not None:
                    return
                with contextlib.suppress(OSError):
                    cm.__exit__(None, None, None)
            enter_future.add_done_callback(_release_if_acquired)
            raise
    except OSError:
        cm = None
    try:
        yield
    finally:
        if cm is not None:
            with contextlib.suppress(OSError):
                cm.__exit__(None, None, None)


# Cold-verify shared-venv pre-provision coalescing guard (task 2997).
# run_verification is invoked N times concurrently in fan-outs (run_scoped_
# verification's per-module gather; the post-merge type-only pyright gather in
# merge_queue.py).  Without coalescing, each call would spawn its own `uv sync`
# on the same cold `.venv` — itself the concurrent-uv-on-cold-venv operation
# this task exists to eliminate.  A per-worktree lock + a TTL'd completed-path
# map (both keyed by the RESOLVED worktree path) make overlapping callers await a
# single in-flight provision and later callers skip.  Follows the module-level
# cache + TTL precedent (_PROBE_CACHE / _PROBE_CACHE_TTL).
#
# No separate guard lock protects the dict: the get-or-create below runs with no
# `await` between the read and the write, so under asyncio's cooperative
# scheduling it is atomic (two coroutines cannot interleave within it).  A
# module-level `asyncio.Lock()` created at import would instead bind to the first
# event loop that touched it and raise cross-loop under pytest's per-test loops.
#
# The completed-path map records a monotonic completion time (not a bare set) so
# entries EXPIRE.  Two failure modes a permanent set would have:
#   * LEAK — a fresh-per-merge `.worktrees/_merge-<uuid>` path is never revisited,
#     so an unbounded map/set would pin one entry per merge forever.
#   * STALE SKIP — a warm-lane path later reset IN PLACE to a cold state (its
#     `.venv` wiped and the `verify_warmed` marker cleared, so `_is_verify_cold`
#     reclassifies it cold) would be SKIPPED on its retained "done" key and never
#     re-provision, reintroducing the very concurrent-uv-on-empty-venv race this
#     guard exists to prevent.
# TTL-expiring the entry on read + opportunistically pruning on each completion
# (`_prune_preprovision_guard`) handles both.
_PREPROVISION_LOCKS: dict[str, asyncio.Lock] = {}
_PREPROVISION_DONE: dict[str, float] = {}
_PREPROVISION_DONE_TTL: float = 300.0  # 5 min; >> any single verify fan-out's coalescing window


def _prune_preprovision_guard() -> None:
    """Evict TTL-expired entries from the pre-provision coalescing guard.

    Called opportunistically after each completion so the guard stays bounded:
    a fresh-per-merge ``.worktrees/_merge-<uuid>`` path is never revisited, so
    without a sweep its DONE + LOCKS entries would leak forever.  A DONE entry
    older than :data:`_PREPROVISION_DONE_TTL` is dropped (its worktree either no
    longer exists or, if reset in-place to a cold state, SHOULD re-provision
    rather than skip on the stale key); the companion lock is dropped only when
    uncontended, so an active provision/fan-out still coalescing on it is never
    disturbed.
    """
    now = time.monotonic()
    for key, done_at in list(_PREPROVISION_DONE.items()):
        if now - done_at >= _PREPROVISION_DONE_TTL:
            del _PREPROVISION_DONE[key]
            lock = _PREPROVISION_LOCKS.get(key)
            if lock is not None and not lock.locked():
                del _PREPROVISION_LOCKS[key]


async def _preprovision_shared_venv(
    worktree: Path,
    config: OrchestratorConfig,
    *,
    verify_env: dict[str, str],
    timeout: float,
) -> None:
    """Synchronously populate the shared ``.venv`` before the test/lint/type gather.

    On a COLD verify worktree the shared ``.venv`` is populated only as a SIDE
    EFFECT of the TEST leg's ``cd <module> && uv run pytest``.  The full-repo-
    scope root LINT (``uv run ruff check …``) and TYPE (``… npx pyright``)
    commands race that sync in the concurrent gather and fail spuriously
    (``Failed to spawn: ruff``; ``Import "pytest" could not be resolved``).
    Running ``config.verify_cold_preprovision_command`` here, synchronously,
    before the gather closes that race by populating the venv first.

    No-op when the knob is empty (the project-agnostic default — the deployed
    uv-workspace value lives only in dark-factory-orchestrator.yaml).  Runs
    through ``_run_cmd`` so it inherits the ghost-venv isolation scrub from
    ``_target_subprocess_env`` (the TARGET's ``uv`` resolves the TARGET's
    ``.venv``) with no new isolation code.
    """
    cmd = config.verify_cold_preprovision_command
    if not cmd:
        return
    key = str(worktree.resolve())
    # Fast path: recently provisioned for this worktree — skip without taking the
    # lock (the common case once a worktree's first cold verify has run).  TTL'd
    # so a path later reset in-place to a cold state re-provisions instead of
    # skipping on a stale "done" key.
    done_at = _PREPROVISION_DONE.get(key)
    if done_at is not None and time.monotonic() - done_at < _PREPROVISION_DONE_TTL:
        return
    # Get-or-create the per-worktree lock.  Synchronous (no await between the get
    # and the set) → atomic under asyncio, so no separate guard lock is needed.
    lock = _PREPROVISION_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _PREPROVISION_LOCKS[key] = lock
    async with lock:
        # Double-checked: a concurrent caller may have completed the provision
        # while we awaited the lock — coalesce onto its single sync and skip
        # (freshness re-checked against the TTL, as in the fast path).
        done_at = _PREPROVISION_DONE.get(key)
        if done_at is not None and time.monotonic() - done_at < _PREPROVISION_DONE_TTL:
            return
        logger.info('Cold-verify shared-venv pre-provision: running %r in %s', cmd, worktree)
        rc, out, _ = await _run_cmd(cmd, worktree, timeout, env=verify_env or None)
        if rc != 0:
            # Fail-open + loud (honours the loud-over-silent-degradation norm;
            # mirrors the _govern_cpu_str fail-open convention).  A SUCCESSFUL
            # sync is what closes the race; a FAILED sync is a real infra problem
            # the gather's own tool spawn + existing failure classification will
            # surface, so the pre-provision must NEVER become a NEW failure
            # source — warn and return normally, always proceeding to the gather,
            # never raising.
            tail = out[-500:] if out else ''
            logger.warning(
                'Cold-verify shared-venv pre-provision failed (rc=%d), proceeding '
                'to the verify gather anyway: %r\noutput tail: %s',
                rc, cmd, tail,
            )
        # Mark done on completion (success OR fail-open failure) so overlapping
        # callers coalesce onto this single in-flight sync and later callers skip.
        # A failed sync is deliberately not retried per-caller (fail-open; the
        # gather surfaces any real breakage) — retrying N times on a cold
        # throwaway worktree would not help and reintroduces the concurrent-uv
        # race this guard exists to prevent.  Record the completion TIME (not a
        # bare membership marker) so the entry TTL-expires, then opportunistically
        # prune stale entries so the guard stays bounded across many worktrees.
        _PREPROVISION_DONE[key] = time.monotonic()
        _prune_preprovision_guard()


async def run_verification(
    worktree: Path,
    config: OrchestratorConfig,
    module_config: ModuleConfig | None = None,
    *,
    allow_cold_cache: bool = True,
    max_retries: int | None = None,
    is_merge_verify: bool = False,
    attempt_id: int | None = None,
    task_id: str | None = None,
    archive_root: Path | None = None,
    role: Literal['merge', 'task', 'background'] = 'task',
    segment_chained_test: bool = False,
) -> VerifyResult:
    """Run test suite, linter, and type checker. Return structured result.

    *segment_chained_test* (task 3338 / esc-3062-2) opts the TEST leg into
    per-segment execution: when the configured command is an `&&` chain
    ``split_and_chain_segments`` accepts, every clause is run separately so
    an unrelated earlier subproject's red can no longer make the SHELL skip
    the clause a task's OWN assigned files live in. Default OFF, and passed
    True from exactly one call site (``run_scoped_verification``'s fallback
    branch): segmenting every chain would silently change the global tail,
    the cargo-scoped path, ``merge_queue._run_unscoped_typechecks`` and
    every module_configs run. When the segmenter REFUSES, this is a no-op
    and the chain runs exactly as it does today.

    Each segment is its own subprocess, so the per-execution details are
    re-applied per segment rather than once for the chain: cpu-governance,
    the admission nice prefix, its own streamed log path (keyed off the
    index-suffixed ``ChainSegment.label``, so two segments sharing a cwd
    cannot collide) and — since task 3478 — the ``verify_admission_pytest_n``
    ``-n`` worker cap. ``CheckRun.cmd`` stays the operator's configured
    chain throughout; all of the above are execution details layered onto
    the segment, not rewrites of what was configured. Per-segment junitxml
    is deliberately NOT among them: see the guard and the CheckRun comment
    at the construction site.

    When *module_config* is provided, a ``None`` command means "skip that check"
    (the subproject doesn't define it).  When *module_config* is ``None``,
    global config commands are used for every check.

    If any enabled command times out while the others pass, the whole verify
    is retried up to *max_retries* times (default ``config.verify_timeout_retries``).
    Pass ``max_retries=0`` to disable retries entirely — appropriate for
    merge-queue post-merge verification, where a deterministic hang would
    otherwise triple the queue-wide stall.  A retry that surfaces a genuine
    failure (e.g., a real lint error) is returned immediately instead of
    being retried further.

    When *allow_cold_cache* is ``False`` cold-timeout detection is disabled
    entirely regardless of filesystem state, and the warm timeout is always
    used.  Useful for review-checkpoint or eval callers that pass arbitrary
    paths which may happen to contain a ``.task/`` directory.  Defaults to
    ``True`` (auto-detect from filesystem).

    When *is_merge_verify* is ``True`` the verify is treated as always cold,
    regardless of filesystem state (merge worktrees are freshly created per
    merge — no ``.task/`` dir, but also no warm build cache — and so the
    ``_is_verify_cold`` filesystem heuristic mis-classifies them as warm).
    This bypasses ``_is_verify_cold`` entirely and uses the cold-track timeout
    cascade (``verify_cold_command_timeout_secs``).  Also implies
    ``allow_cold_cache=True`` semantics for the timeout; the warm marker is
    NOT written on success because merge worktrees are ephemeral.  Defaults
    to ``False`` for all non-merge callers so existing cold-detection
    behaviour is preserved.
    """
    if module_config is not None:
        # Scoped: use module command; None → skip
        test_cmd = module_config.test_command
        lint_cmd = module_config.lint_command
        type_cmd = module_config.type_check_command
    else:
        # Global fallback
        test_cmd = config.test_command
        lint_cmd = config.lint_command
        type_cmd = config.type_check_command

    module_prefix = module_config.prefix if module_config is not None else None

    # Task μ (verify-scope-inversion-prd.md): under role=='merge' AND
    # merge_verify_breadth=='full', inject --junitxml into the test leg (see
    # _run_or_skip_timed below) and parse the report afterward into
    # VerifyResult.failing_test_ids — the baseline-attribution signal shared
    # by verify_failure_is_preexisting_on_main. Gated on breadth=='full'
    # because only then does a passing verify mean "this module's suite is
    # genuinely clean" (λ, task 2589); under 'scoped' a pass says nothing
    # about the rest of main, so seeding an empty baseline would be wrong.
    #
    # junit_path is computed here whenever role+breadth match, independent
    # of whether test_cmd is actually a structured pytest command — deciding
    # eligibility is left entirely to with_junitxml's own no-op guard
    # (OPAQUE / raw-retained chain / non-pytest tool / cmd is None all leave
    # the flag uninjected), so this never duplicates that check. When
    # nothing ever injects the flag, pytest never writes the report, and
    # _extract_failing_test_ids_from_junit(junit_path) below degrades to
    # None (file not found) — the same B3 degrade signal as an unreadable
    # report.
    #
    # The path is worktree-internal (merge worktrees lack `.task/` —
    # git_ops.py scrubs it) and per-module_prefix (mirrors _stream_log_path's
    # infix immediately below) so concurrent per-module fan-out within one
    # worktree can't collide. Absolute: module commands may `cd <prefix>`,
    # so a relative --junitxml would land in the wrong directory.
    junit_path: Path | None = None
    if role == 'merge' and verify_plan._merge_breadth_is_full(config):
        # Shape-2 husk guard (task 2922): _prepare_junit_report_path returns
        # None WITHOUT re-creating a torn-down worktree as an empty husk when a
        # late merge-role verify writer fires after teardown.
        junit_path = _prepare_junit_report_path(worktree, module_prefix)

    if is_merge_verify:
        # Merge worktrees are freshly created per merge — cargo caches are
        # cold and the ``.task/`` marker is absent — so the filesystem
        # heuristic mis-classifies them as warm.  Force cold semantics.
        is_cold = True
    elif allow_cold_cache:
        is_cold = _is_verify_cold(worktree, module_prefix)
    else:
        is_cold = False
    timeout = _resolve_verify_timeout(config, module_config, is_cold=is_cold, is_merge_verify=is_merge_verify)
    if max_retries is None:
        max_retries = config.verify_timeout_retries

    if is_cold:
        warm_timeout = _resolve_verify_timeout(config, module_config, is_cold=False)
        if timeout != warm_timeout:
            logger.info(
                'Cold-cache verify: using %ds timeout (warm would be %ds)',
                int(timeout), int(warm_timeout),
            )
    concurrent = _resolve_concurrent_verify(config, module_config)
    # Merge-role fail-fast: lint first, short-circuit test+type on a lint
    # failure.  Only meaningful on the sequential branch (never concurrent) and
    # only for role=='merge' — task/background keep today's test→lint→type order.
    lint_first = (
        role == 'merge'
        and not concurrent
        and _resolve_sequential_lint_first(config, module_config)
    )
    verify_env = _resolve_verify_env(config, module_config, role=role)

    # DF_VERIFY_ROLE is always present (injected by _resolve_verify_env); log at
    # INFO only when user-configured keys also exist so we don't inflate INFO
    # volume on the hot verify path for plain task verifies.
    user_env_keys = set(verify_env.keys()) - {'DF_VERIFY_ROLE'}
    if user_env_keys:
        logger.info(
            'Verification env (mode=%s): %s',
            'concurrent' if concurrent else 'sequential',
            sorted(verify_env.keys()),
        )
    else:
        logger.debug(
            'Verification mode: %s',
            'concurrent' if concurrent else 'sequential',
        )

    # Resolve the streaming log path for each label.  Identical to the
    # filename computed in ``_persist_attempt_logs`` so that ``_run_cmd``'s
    # streamed file is the same file ``_persist_attempt_logs`` would have
    # written via ``write_text`` — the latter now skips the rewrite when the
    # streamed file already exists on disk.  Returns None when ``.task/`` is
    # absent (review-checkpoint / merge-queue paths), preserving the legacy
    # buffered behaviour for those callers.
    def _stream_log_path(label: str, current_attempt: int) -> 'Path | None':
        if attempt_id is None:
            return None
        task_dir = worktree / '.task'
        if not task_dir.is_dir():
            return None
        verify_dir = task_dir / 'verify'
        try:
            verify_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        if module_prefix is not None:
            safe = module_prefix.replace('/', '_').replace(' ', '_')
            infix = f'.{safe}'
        else:
            infix = ''
        return verify_dir / f'attempt-{current_attempt}{infix}.{label}.log'

    async def _run_or_skip_timed(
        cmd: str | None,
        *,
        label: str,
        current_attempt: int,
    ) -> CheckRun:
        """Like _run_cmd but returns a CheckRun capturing (rc, output, timed_out,
        started_at, duration_secs) for this check.

        When *cmd* is None (skipped check), returns ``CheckRun.skipped(label)``.
        """
        if cmd is None:
            return CheckRun.skipped(label)
        # Capture the un-governed config command before _govern_cpu_str/the
        # nice-wrap below reassign the local `cmd` — CheckRun.cmd must reflect
        # what the caller configured, not the governed/wrapped string handed
        # to _run_cmd, so persisted logs and _summarize_checks see the same
        # command they always have.
        config_cmd = cmd
        # junitxml injection (task μ, verify-scope-inversion-prd.md): only
        # the 'test' leg, only when junit_path was computed above (role
        # =='merge' and breadth=='full'). _with_junitxml_str keeps the
        # identity-check semantics this site always had — with_junitxml
        # no-ops for OPAQUE/raw-retained/non-pytest commands, so the
        # parse->render round-trip is skipped and a no-op stays
        # byte-identical — and additionally reports a suppressed injection
        # on a pytest command at INFO (task 3218), since reaching here means
        # a junit report was expected and will not be written. MUST run
        # before the cpu-governance wrap immediately below: once governed,
        # cmd is an opaque outer `<exec> -- /bin/bash -c '...'` string that
        # parse_config_command can no longer see as pytest.
        if junit_path is not None and label == 'test':
            cmd = _with_junitxml_str(cmd, str(junit_path))
            assert cmd is not None  # None only when the input is None; guarded above
        # Admission gate (task 2390 T2): only the pytest ('test') leg is
        # gated by the shared.verify_admission flock semaphore + role nice
        # tier; lint/type ride alongside within the same verify, ungated.
        # Resolved here rather than after the governance wrap below because
        # the `-n` gate reads it and must itself precede that wrap; it
        # depends on nothing but config/label, so the move is inert.
        admission = _verify_admission_active(config) and label == 'test'
        # -n cap (task 2394 T6): applies only to roles {task, background} —
        # 'merge' is never -n-capped (bypasses admission slot-counting,
        # latency-critical). No-op when the knob is '' or 'auto' (the
        # apply_pytest_numprocesses no-op guard) — byte-identical to today.
        # config_cmd above intentionally stays un-rewritten (same treatment
        # as the nice prefix: an execution detail layered onto cmd, not the
        # persisted config command).
        #
        # Hoisted into ONE local (task 3478) because the segmented branch
        # below applies the same cap per segment: a second copy of this
        # predicate could drift, letting role/admission/knob eligibility
        # disagree between the segmented and unsegmented paths.
        pytest_n_capped = (
            admission
            and role in {'task', 'background'}
            and config.verify_admission_pytest_n not in {'', 'auto'}
        )
        # _with_pytest_numprocesses_str identity-checks the mutation before
        # rendering (mirrors _govern_cpu_str's `governed is parsed` guard
        # below), so a no-op leaves cmd byte-identical rather than
        # reformatting a command that wasn't actually touched.
        #
        # MUST run before the cpu-governance wrap immediately below and after
        # the junitxml injection above — the same ordering constraint, for
        # the same reason: once governed, cmd is an opaque outer `<exec> --
        # /bin/bash -c '...'` string that parse_config_command can no longer
        # see as pytest, so the cap would silently vanish. Both gates are
        # disjoint today (governance resolves only for role=='merge', the cap
        # only for role in {'task','background'}), so this is defence in
        # depth; ordering it identically to _run_one_segment below is what
        # keeps the two paths from disagreeing if either gate ever widens.
        if pytest_n_capped:
            cmd = _with_pytest_numprocesses_str(cmd, config.verify_admission_pytest_n)
            assert cmd is not None  # None only when the input is None; guarded above
        # Wrap the command in cpu-governed-exec.sh when role=='merge' and
        # cpu_governance is enabled + exec resolves.  Fail-open: returns cmd
        # unchanged when governance is disabled or the path is non-executable,
        # so a misconfig never makes a verify spawn fail.
        cmd = _govern_cpu_str(cmd, _resolve_governed_exec_path(config, worktree, role))
        assert cmd is not None  # _govern_cpu_str returns None only when cmd is None; guarded above
        if admission:
            prefix = _resolve_nice_prefix(config, role)
            if prefix:
                cmd = f'{shlex.join(prefix)} /bin/bash -c {shlex.quote(cmd)}'
        async with (_admission_slot(role, config) if admission else contextlib.nullcontext()):
            started_at = datetime.now(UTC).isoformat()
            t0 = time.monotonic()
            # Pass use_cgroup_scope only when enabled so the default-off call
            # signature stays byte-identical (test doubles stub the legacy kwargs).
            _scope_kw: _ScopeKw = (
                {
                    'use_cgroup_scope': True,
                    'scope_tag': _scope_tag_for(config.project_root),
                }
                if config.verify_use_cgroup_scope
                else {}
            )
            # Pass clock_stop only when enabled (mirrors _scope_kw pattern) so the
            # default-off call signature stays byte-identical for existing test doubles.
            _clock_kw: _ClockKw = (
                {
                    'clock_stop': ClockStopConfig(
                        marker_stop=config.verify_clock_stop_marker_stop,
                        marker_heartbeat=config.verify_clock_stop_marker_heartbeat,
                        marker_start=config.verify_clock_stop_marker_start,
                        heartbeat_idle_max=config.verify_clock_stop_heartbeat_idle_max,
                        max_total_secs=config.verify_clock_stop_max_total_secs,
                    ),
                }
                if config.verify_clock_stop_enabled
                else {}
            )
            # Segmented test leg (task 3338 / esc-3062-2). Segment from
            # `config_cmd` — the pre-junitxml, pre-governance capture — so
            # `_with_junitxml_str`'s suppressed-injection INFO log (task 3218)
            # still fires on the WHOLE chain exactly as today. The admission
            # slot is already held ONCE around this whole block, so
            # slot-counting semantics are unchanged no matter how many
            # segments run.
            chain_segments = (
                split_and_chain_segments(config_cmd)
                if segment_chained_test and label == 'test'
                else None
            )
            # Co-occurrence guard (task 3478). These two are mutually
            # exclusive by construction today: junit_path is computed only
            # when role=='merge' and breadth=='full', while segmentation is
            # opted into only as `segment_chained_test=role != 'merge'`. So
            # per-segment junitxml is deliberately UNWIRED — with no writer
            # and no consumer it would be dead code, and node-id attribution
            # on this path comes from _extract_failing_test_ids over the
            # aggregated segment stdout instead.
            #
            # If a future change ever relaxes either gate, the tempting
            # "fix" is to hand every segment the SAME --junitxml path, which
            # is last-writer-wins: N pytest runs, one report, silently
            # describing only the last segment. Say so at the point it
            # becomes observable rather than letting it look like it worked.
            # Storm-safe without extra machinery: at most once per test leg,
            # and only in a configuration that does not exist today.
            if chain_segments is not None and junit_path is not None:
                logger.warning(
                    'Segmented verify (%d segments) co-occurs with a junit report '
                    'path at %s, which will NOT be written: per-segment junitxml is '
                    'deliberately unwired (task 3478). These two were mutually '
                    'exclusive by construction — junit_path requires role==\'merge\' '
                    'with merge_verify_breadth==\'full\', segmentation requires '
                    'role!=\'merge\' — so reaching here means one of those gates '
                    'changed. Do NOT resolve this by injecting one shared '
                    '--junitxml: %d pytest runs writing one path is last-writer-wins, '
                    'a report describing only the final segment. Node-id attribution '
                    'for THIS run remains available via _extract_failing_test_ids '
                    'over the aggregated segment stdout.',
                    len(chain_segments),
                    junit_path,
                    len(chain_segments),
                )
            if chain_segments is not None:
                async def _run_one_segment(
                    segment_cmd: str,
                    segment_cwd: Path,
                    segment_timeout: float,
                    segment_label: str,
                ) -> tuple[int, str, bool]:
                    # The `-n` cap, cpu-governance and the nice prefix are all
                    # per-segment: each segment is its own subprocess, so each
                    # needs its own wrap.
                    #
                    # Per SEGMENT is the right granularity, not N times the
                    # parallelism (task 3478, correcting task 3338's comment):
                    # _run_segmented awaits `run_one` once per loop iteration,
                    # so segments run SEQUENTIALLY. A per-segment `-n N`
                    # therefore caps N workers at any instant — exactly what
                    # `-n N` means for a single pytest command. Without this
                    # the cap was silently dropped on this path, since the
                    # rewrite above lands on `cmd` while segments are built
                    # from `config_cmd`.
                    #
                    # BEFORE _govern_cpu_str — the SAME order as the
                    # unsegmented site above (and as the junitxml injection,
                    # which documents the constraint first): a governed
                    # command is an opaque outer `<exec> -- /bin/bash -c
                    # '...'` string that parse_config_command can no longer
                    # see as pytest. Governance is inert here
                    # (_resolve_governed_exec_path returns None for role !=
                    # 'merge', and segmentation only happens for role !=
                    # 'merge'), so ordering it first is defence in depth
                    # against a future change to that gate rather than a live
                    # dependency — but it is deliberately the same defence the
                    # unsegmented path takes, so widening the governance gate
                    # cannot make one path honour the operator's cap while the
                    # other silently discards it (the asymmetry task 3478
                    # exists to remove).
                    capped = segment_cmd
                    if pytest_n_capped:
                        capped = _with_pytest_numprocesses_str(
                            segment_cmd, config.verify_admission_pytest_n,
                        )
                        assert capped is not None  # None only for a None input
                    governed = _govern_cpu_str(
                        capped, _resolve_governed_exec_path(config, worktree, role),
                    )
                    assert governed is not None  # None only for a None input
                    if admission:
                        seg_prefix = _resolve_nice_prefix(config, role)
                        if seg_prefix:
                            governed = (
                                f'{shlex.join(seg_prefix)} /bin/bash -c {shlex.quote(governed)}'
                            )
                    return await _run_cmd(
                        governed,
                        segment_cwd,
                        segment_timeout,
                        env=verify_env or None,
                        log_path=_stream_log_path(
                            f'{label}.{segment_label}', current_attempt,
                        ),
                        **_scope_kw,
                        **_clock_kw,
                    )

                rc, out, timed_out_flag, segment_dicts = await _run_segmented(
                    chain_segments,
                    run_one=_run_one_segment,
                    worktree=worktree,
                    # The single value run_verification already resolved via
                    # _resolve_verify_timeout governs the WHOLE segmented run,
                    # preserving today's total wall-clock contract exactly.
                    budget_secs=timeout,
                )
            else:
                segment_dicts = None
                rc, out, timed_out_flag = await _run_cmd(
                    cmd,
                    worktree,
                    timeout,
                    env=verify_env or None,
                    log_path=_stream_log_path(label, current_attempt),
                    **_scope_kw,
                    **_clock_kw,
                )
        # Mis-resolved interpreter (task 3367 / esc-3359-1): make the condition
        # LEGIBLE at the point it is observed. Classification alone routes the
        # merge lane correctly (ENV_TRANSIENT -> a loud infra_issue hold) but
        # says nothing about WHY; an operator reading hundreds of phantom
        # "could not be resolved" lines has no way to tell a mis-resolved
        # interpreter from a branch that genuinely dropped its dependencies.
        #
        # Structurally fires at most ONCE per failing check: this is the single
        # post-_run_cmd path, so hundreds of matching output lines collapse to
        # one statement with no de-dup counter. Uses the SAME shared predicate
        # the classifier does — one detection site, never a second copy of the
        # regex. The raw pyright text still streams to the per-leg log file
        # untouched; this line is an interpretation layered ON TOP of it, not a
        # replacement for it (loud-over-silent-degradation).
        if rc != 0 and is_interpreter_missing_workspace_packages(out):
            logger.error(
                'Verification %r check failed against a Python interpreter that '
                'has NONE of the workspace third-party packages: %d distinct '
                'top-level modules are unresolved, including baseline dev '
                'dependencies. This is an ENVIRONMENT mis-resolution, NOT a '
                'branch defect — pyright resolved an interpreter from the '
                'ambient VIRTUAL_ENV/PATH (which verify strips deliberately) '
                'instead of this worktree\'s own .venv. Fix surface: pin '
                '[tool.pyright] venvPath/venv in the checked subproject\'s '
                'pyproject.toml. See task 3367 / esc-3359-1.',
                label,
                len(unresolved_top_level_modules(out)),
            )
        return CheckRun(
            label=label,
            # Deliberately the ORIGINAL full chain even on the segmented path:
            # it feeds _persist_attempt_logs/_build_summary_payload/
            # _summarize_checks, so preserving it keeps every persisted
            # artifact and every failure classification identical. Task 3338
            # changes execution TOPOLOGY and REPORTING, not what any command
            # is. The per-segment `-n` cap below is layered onto the SEGMENT
            # for the same reason the nice prefix and cpu-governance are: an
            # execution detail, not the persisted config command.
            #
            # Task 3478 settled the two dispositions task 3338 recorded here
            # as follow-ups:
            #
            # - junitxml is NOT wired per segment, and not because it is a
            #   harmless no-op: junit_path requires role=='merge' while
            #   segmentation requires role!='merge', so it is unreachable by
            #   construction — dead code with no writer and no consumer
            #   (_extract_failing_test_ids_from_junit runs only `if junit_path
            #   is not None`). Node-id attribution here comes from
            #   _extract_failing_test_ids over the aggregated segment stdout.
            #   The guard above warns if either gate ever relaxes.
            #
            # - apply_pytest_numprocesses IS now applied per segment, inside
            #   _run_one_segment. Its gate's roles ({'task','background'}) are
            #   exactly the segmented-path roles, so it was never exempt —
            #   just silently dropped, the rewrite landing on `cmd` while
            #   segments are built from `config_cmd`. Segments run
            #   sequentially, so a per-segment `-n N` keeps its single-command
            #   meaning (N workers at any instant) rather than multiplying
            #   parallelism by the segment count.
            cmd=config_cmd,
            rc=rc,
            output=out,
            timed_out=timed_out_flag,
            started_at=started_at,
            duration_secs=time.monotonic() - t0,
            segments=segment_dicts,
        )

    # Cold-verify shared-venv pre-provision (task 2997, esc-2913-3): populate
    # the shared .venv ONCE, synchronously, before the concurrent test/lint/type
    # gather below, so the full-repo-scope root LINT/TYPE commands don't race the
    # TEST-driven `uv run pytest` sync and fail spuriously on an empty venv.
    # Gated on is_cold — the race's defining condition (a cold/possibly-empty
    # venv) — so it is a no-op on warm lanes and also covers the post-merge
    # type-only pyright path (test/lint=None → no TEST leg to populate the venv).
    # No-op when the knob is unset (project-agnostic empty default).
    if is_cold:
        await _preprovision_shared_venv(
            worktree, config, verify_env=verify_env, timeout=timeout,
        )

    retries = 0
    while True:
        # attempt_id is the persistence ID handed in by the caller (or None for
        # callers that don't persist).  We use it directly as the streaming
        # attempt index so the streamed log path lines up with the path
        # ``_persist_attempt_logs`` computes below; this loop's local
        # ``retries`` counter is for retry bookkeeping only.
        current_attempt_id = attempt_id if attempt_id is not None else 0
        # `attempt` is assigned unconditionally here, before the loop's only
        # `break` below, so it is always bound whenever the loop exits — no
        # pre-loop sentinel scaffolding needed.
        if concurrent:
            attempt = VerifyAttempt(list(await asyncio.gather(
                _run_or_skip_timed(test_cmd, label='test', current_attempt=current_attempt_id),
                _run_or_skip_timed(lint_cmd, label='lint', current_attempt=current_attempt_id),
                _run_or_skip_timed(type_cmd, label='type', current_attempt=current_attempt_id),
            )))
        elif lint_first:
            # Merge-role fail-fast: run lint first; on a lint failure, short-
            # circuit — record test+type as skipped (vacuous rc=0 passes) and
            # let the attempt fail on the lint leg, so a lint-only-red merge
            # skips the long test phase entirely.
            lint_run = await _run_or_skip_timed(lint_cmd, label='lint', current_attempt=current_attempt_id)
            if lint_run.rc != 0:
                attempt = VerifyAttempt([
                    CheckRun.skipped('test'), lint_run, CheckRun.skipped('type'),
                ])
            else:
                # Lint green: run test+type, then assemble in canonical
                # [test, lint, type] order (not execution order) so the green-
                # path VerifyResult and persisted artifacts stay byte-identical
                # to the plain-sequential branch below.
                test_run = await _run_or_skip_timed(test_cmd, label='test', current_attempt=current_attempt_id)
                type_run = await _run_or_skip_timed(type_cmd, label='type', current_attempt=current_attempt_id)
                attempt = VerifyAttempt([test_run, lint_run, type_run])
        else:
            attempt = VerifyAttempt([
                await _run_or_skip_timed(test_cmd, label='test', current_attempt=current_attempt_id),
                await _run_or_skip_timed(lint_cmd, label='lint', current_attempt=current_attempt_id),
                await _run_or_skip_timed(type_cmd, label='type', current_attempt=current_attempt_id),
            ])

        if attempt.passed or not attempt.pure_timeout_failure or retries >= max_retries:
            break

        retries += 1
        timed_out_names = [c.label for c in attempt.checks if c.timed_out]
        logger.warning(
            'Verification hit timeout on %s; retry %d/%d',
            ','.join(timed_out_names), retries, max_retries,
        )

    # Classify timed_out: true only when the final failure was a pure timeout
    # (no real non-timeout failure mixed in).
    timed_out = (not attempt.passed) and attempt.pure_timeout_failure

    # Build summary/category/cause_hint (shared with the env-recovery retry
    # below via _summarize_checks — see task 2048 code_duplication fix).
    passed, category, cause_hint, summary, failing_leg_categories = _summarize_checks(
        attempt.test.rc, attempt.test.output, attempt.test.timed_out, attempt.test.cmd,
        attempt.lint.rc, attempt.lint.output, attempt.lint.timed_out, attempt.lint.cmd,
        attempt.type.rc, attempt.type.output, attempt.type.timed_out, attempt.type.cmd,
        test_duration=attempt.test.duration_secs,
        lint_duration=attempt.lint.duration_secs,
        type_duration=attempt.type.duration_secs,
    )
    if timed_out:
        summary = f'Verification timed out after {max_retries} retries' if max_retries > 0 else 'Verification timed out'

    # Bounded env-recovery retry: a shared-venv-mutation transient (a
    # concurrent `uv sync` elsewhere vanishing xdist/pip mid-run) is an infra
    # transient, not a code regression. Auto-recover with a single
    # forced-serial retry of the test command — mirrors the pure-timeout
    # retry loop above, but is gated on `category` rather than a timeout
    # flag, and only re-runs the test command since lint/type do not
    # exercise xdist/pip. Recovery passing means the env recovered -> GREEN;
    # recovery still hitting env_transient means it stays environmental
    # (NOT misattributed to test_failure/unknown_test_failure); recovery
    # surfacing a different category means that real signal is reported.
    #
    # This branch can only fire when test_cmd resolves to ToolKind.PYTEST via
    # _tool_for_cmd (see that function's docstring) — a test command wrapped
    # such that the parser can't see a literal `pytest` token never produces
    # ENV_TRANSIENT and so never reaches this retry, even on a genuine
    # shared-venv mutation. True of every production test_cmd today.
    #
    # This PYTEST-only narrowing is broader than just that wrapped-test-cmd
    # case: the pre-δ tool-blind ladder also consulted these env_transient
    # signatures against the LINT and TYPE check outputs (it classified
    # whatever output it was handed, uniformly across all three checks),
    # whereas the RUFF/PYRIGHT tables and the OPAQUE fallback never do now.
    # A conscious tradeoff, not an unnoticed side effect: the signatures are
    # pytest/xdist-specific text ruff/pyright would not emit, and this retry
    # only ever re-runs the test command regardless (lint/type don't exercise
    # xdist/pip — see above).
    #
    # RETIRED COROLLARY (task 3367): this block used to conclude from the above
    # that "a lint or type-check failure cannot classify as env_transient by
    # construction, only the test leg can". That is FALSE as of task 3367.
    # `classify_failure` guard 3 (`_classify_environmental`) is tool-blind and
    # now has three ToolKind-independent ENV_TRANSIENT producers — task 2756's
    # broken `_merge-verify` worktree, task 2831's restart collateral, and task
    # 3367's mis-resolved pyright interpreter (esc-3359-1) — any of which a TYPE
    # or LINT leg can trip. The category therefore no longer implies the TEST
    # leg was the failing one, so the gate below establishes that itself with an
    # explicit `attempt.test.rc != 0`.
    #
    # THE RULE (task 3367): this recovery exists for the TEST leg's shared-venv
    # mutation, and its only action is to re-run the TEST command serially. It
    # therefore fires only when the TEST leg is the failing one. When the test
    # leg already passed there is nothing to recover and the re-run cannot
    # change the verdict — it just spends a full test-suite wall-clock (1320.9s
    # in esc-3359-1, where test rc=0, lint rc=0, type rc=1) before returning the
    # same red, under a "vanished xdist/pip" warning that misdescribes the
    # actual failure. A lint/type env_transient is instead reported directly —
    # loudly (see the per-check ERROR in _run_or_skip_timed) and
    # infra-transient at the merge lane — with no pointless test re-run.
    #
    # The rc check is correct on its own terms, independent of task 3367's new
    # classification: re-running a passing leg can never recover anything.
    if (
        category == FailureCategory.ENV_TRANSIENT
        and attempt.test.cmd is not None
        and attempt.test.rc != 0
    ):
        logger.warning(
            'Verification hit an environmental shared-venv transient '
            '(vanished xdist/pip); retrying test command once, forced serial '
            '(this clears all pyproject addopts, including any marker '
            'filters, for the recovery run — see serial_pytest)'
        )
        recovered_test_cmd = _serial_pytest_str(attempt.test.cmd)
        new_test = await _run_or_skip_timed(
            recovered_test_cmd, label='test', current_attempt=current_attempt_id,
        )

        # Recompute pure-timeout consistency for the recovery run via a fresh
        # VerifyAttempt: lint/type are unchanged from the first pass (only
        # the test leg was re-run), so this reads the SAME
        # VerifyAttempt.pure_timeout_failure property the loop above used —
        # not a second hand-copied formula.  Without this, a recovery run
        # that itself hits the wall-clock timeout would leave the stale
        # timed_out=False from the first pass while category flips to
        # 'infra_timeout' — an inconsistent VerifyResult that both wrongly
        # marks the worktree warm (the "not result.timed_out" check below)
        # and hides the timeout from callers that special-case
        # result.timed_out (merge_queue.py, workflow.py).
        attempt = VerifyAttempt([new_test, attempt.lint, attempt.type])
        timed_out = (not attempt.passed) and attempt.pure_timeout_failure

        passed, category, cause_hint, summary, failing_leg_categories = _summarize_checks(
            attempt.test.rc, attempt.test.output, attempt.test.timed_out, attempt.test.cmd,
            attempt.lint.rc, attempt.lint.output, attempt.lint.timed_out, attempt.lint.cmd,
            attempt.type.rc, attempt.type.output, attempt.type.timed_out, attempt.type.cmd,
            test_duration=attempt.test.duration_secs,
            lint_duration=attempt.lint.duration_secs,
            type_duration=attempt.type.duration_secs,
        )
        if timed_out:
            # Distinct wording from the first-pass timeout summary: this
            # timeout happened on the single bounded env-recovery retry, not
            # the pure-timeout retry loop, so "after {max_retries} retries"
            # would misdescribe it.
            summary = 'Verification timed out during env-recovery retry'

    # Bare pytest-xdist worker-crash retry (task 2365): under host overload a
    # starved xdist worker is os._exit()'d by pytest-timeout's thread method,
    # and --max-worker-restart=0 (task 1907, kept intentionally at 0) turns
    # that into a bare "node down: Not properly terminated" / "worker gwN
    # crashed" failure attributed to whatever test happened to be running —
    # not a real code regression (esc-2286-21). _is_bare_xdist_worker_crash
    # is the conservative discriminator: it returns False (no reclassify)
    # whenever a genuine pytest failure marker is also present, so a real
    # failure is never masked. The gate also requires lint_rc == 0 and
    # type_rc == 0: the discriminator only inspects test_out for pytest
    # failure markers, so without this a genuine, co-occurring lint/type
    # regression would be silently diverted onto the infra-retry path
    # instead of being surfaced as its own test_failure. Requiring both
    # other legs to be clean means the reclassification only fires when the
    # crashed test leg is the ONLY non-zero check. Gated on
    # `not is_merge_verify` — mirrors the _mark_verify_warm precedent below
    # — because the merge path (merge_queue.py) has no VerifyInfraError
    # handler and an uncaught raise there would stall the merge queue.
    # Raising here (rather than returning a failure category) routes
    # through the EXISTING bounded exponential-backoff retry
    # (_run_scoped_verification_with_infra_retry in workflow.py) instead of
    # the debugfix loop's DEBUGGER invocation — the task-path's only
    # auto-retry-without-debugging mechanism.
    #
    # Known accepted tradeoff: a genuine, deterministically-reproducible
    # regression that hangs (e.g. an infinite loop) will, under xdist +
    # pytest-timeout's thread-kill method, also os._exit() the worker and
    # produce this identical bare "node down" signature with no FAILED/E/
    # summary marker — indistinguishable from a host-overload crash by
    # signature alone. Such a hang is routed to the bounded infra retry
    # instead of straight to the debugger; unlike an overload flake it will
    # recur on every retry (a hang doesn't self-heal), so it exhausts the
    # retry window and lands in infra_hold + escalate_to_human rather than
    # being auto-debugged immediately. That is a fail-safe outcome (a human
    # sees it, nothing is silently greened), not a fail-fast one, and is
    # judged acceptable against the status quo of burning debugger
    # iterations on non-reproducible overload flakes.
    if (
        not is_merge_verify
        and attempt.test.rc != 0
        and attempt.lint.rc == 0
        and attempt.type.rc == 0
        and _is_bare_xdist_worker_crash(attempt.test.output)
    ):
        logger.warning(
            'Task %s: bare pytest-xdist worker crash detected (module_prefix=%r) '
            'with no real failure marker in test output — reclassifying as '
            'transient infra (xdist_worker_crash) and raising VerifyInfraError '
            'for the bounded whole-suite retry instead of invoking the debugger',
            task_id, module_prefix,
        )
        raise VerifyInfraError(phase='xdist_worker_crash', errno=None)

    # Name the unrun segments in the one-line verdict (task 3338 / esc-3062-2).
    # `_summarize_checks` already surfaces the NOT RUN block's "unknown is not a
    # pass" wording, so the run is correctly non-green — but non-green is not
    # the same as legible: without the labels a triaging agent still cannot tell
    # WHICH subprojects have no result, which is the human-time cost the whole
    # task is about.
    #
    # Deliberately a small local append rather than a change to
    # `_summarize_checks`: that function's wide positional signature is shared
    # with the env-recovery retry path, and both of its call sites are above, so
    # editing it would be an unrequested blast radius for a purely additive
    # verdict detail. Placed here — after BOTH `_summarize_checks` calls and
    # before `runs`/persistence — so the augmented hint reaches
    # `_persist_attempt_logs`/`_archive_merge_verify_logs` too, not just the
    # returned VerifyResult.
    _not_run_labels = [
        seg['label']
        for seg in (attempt.test.segments or [])
        if seg.get('status') == 'not_run'
    ]
    if _not_run_labels:
        cause_hint = f'{cause_hint} | segments not run: {", ".join(_not_run_labels)}'

    # Hoist runs list so both the merge-path and task-path branches can use it.
    runs = [c.to_dict() for c in attempt.checks]

    worktree_log_paths: list[str] = []
    archive_log_paths: list[str] = []
    if role == 'merge':
        # Merge worktrees have .task/ scrubbed by design (git_ops.py); there
        # are no worktree log files to copy.  Write directly to the durable
        # archive instead.  No deny-list check: on the merge path there is no
        # debugger loop — every failure goes straight to a human/steward —
        # so infra_timeout and test_failure (the exact categories that
        # distinguish timeout-vs-real-failure) must be archived
        # unconditionally.  archive_root is not None is the discriminator
        # that auto-excludes cold-shadow / drift paths (left at None).
        if archive_root is not None and task_id is not None and not passed:
            try:
                arch_paths = _archive_merge_verify_logs(
                    runs, archive_root, task_id, attempt_id or 1,
                    category, cause_hint, module_prefix=module_prefix,
                )
                archive_log_paths = [str(p) for p in arch_paths]
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    'run_verification: merge archival error (non-fatal): %s', exc,
                )
    elif attempt_id is not None and task_id is not None:
        # Task path: persist to worktree/.task/verify/ then optionally copy
        # to the durable archive when category warrants it.
        try:
            wt_paths = _persist_attempt_logs(
                worktree, attempt_id, runs, category, cause_hint,
                module_prefix=module_prefix,
            )
            worktree_log_paths = [str(p) for p in wt_paths]
            arch_paths = _archive_attempt_log(
                wt_paths, archive_root, task_id, attempt_id, category,
            )
            archive_log_paths = [str(p) for p in arch_paths]
        except Exception as exc:  # noqa: BLE001
            logger.warning('run_verification: persistence error (non-fatal): %s', exc)

    # When the three verify commands ran concurrently (asyncio.gather) the
    # true wall-clock cost is the longest single command, not their sum.
    # Serial mode is rare (legacy / explicit opt-out) and sums correctly.
    if concurrent:
        _wall_secs = max(c.duration_secs for c in attempt.checks)
    else:
        _wall_secs = _verify_duration_secs(runs)

    # Task μ: parse the junit report written (if any) by the injection above.
    # None when junit_path was never computed (role != 'merge', breadth !=
    # 'full') or the report is missing/unparseable (nothing ever injected
    # the flag, or the test leg was skipped/crashed before writing it) — the
    # B3 degrade signal. See _extract_failing_test_ids_from_junit's docstring.
    failing_test_ids: list[str] | None = None
    if junit_path is not None:
        failing_test_ids = _extract_failing_test_ids_from_junit(junit_path)

    result = VerifyResult(
        passed=attempt.passed,
        test_output=attempt.test.output,
        lint_output=attempt.lint.output if attempt.lint.rc != 0 else '',
        type_output=attempt.type.output if attempt.type.rc != 0 else '',
        summary=summary,
        timed_out=timed_out,
        cause_hint=cause_hint,
        category=category,
        worktree_log_paths=worktree_log_paths,
        archive_log_paths=archive_log_paths,
        duration_secs=_wall_secs,
        failing_test_ids=failing_test_ids,
        failing_leg_categories=failing_leg_categories,
    )

    # Mark the worktree warm whenever the build completed (no pure timeout),
    # so subsequent verifies use the faster warm timeout.  The marker is
    # per-subproject (keyed by module_prefix) so a concurrent sibling
    # subproject that times out remains cold on the next attempt.
    # Skip the marker for merge-queue verifies: merge worktrees are
    # ephemeral (cleaned up right after), and their `.task/` dir is absent
    # anyway — `_mark_verify_warm` would be a no-op, but the skip keeps the
    # intent explicit.
    if not result.timed_out and not is_merge_verify:
        _mark_verify_warm(worktree, module_prefix)

    if passed:
        logger.info('Verification passed: %s', summary)
    else:
        # Use the richer format when we have a category and a persisted log path —
        # this avoids dumping the raw blob into the orchestrator log.
        if result.category and result.worktree_log_paths:
            hint_part = result.cause_hint or '<no hint>'
            log_ref = result.worktree_log_paths[0]
            log_msg = 'Verification failed: %s — %s (full log: %s)'
            if timed_out:
                logger.warning(log_msg, result.category, hint_part, log_ref)
            else:
                logger.info(log_msg, result.category, hint_part, log_ref)
        else:
            # Legacy format — no log path available (merge-queue, review-checkpoint,
            # or path outside .task/).
            detail_tail = f' — {cause_hint}' if cause_hint else ''
            if timed_out:
                logger.warning('Verification failed: %s%s', summary, detail_tail)
            else:
                logger.info('Verification failed: %s%s', summary, detail_tail)
    return result


def _aggregate_results(results: list[VerifyResult]) -> VerifyResult:
    """Merge per-subproject VerifyResults into one."""
    if len(results) == 1:
        return results[0]

    passed = all(r.passed for r in results)
    test_output = '\n'.join(r.test_output for r in results if r.test_output)
    lint_output = '\n'.join(r.lint_output for r in results if r.lint_output)
    type_output = '\n'.join(r.type_output for r in results if r.type_output)

    # Aggregate timed_out: true only when every failing subproject failed
    # purely due to timeout.  A single real failure poisons the signal.
    failing = [r for r in results if not r.passed]
    timed_out = (not passed) and bool(failing) and all(r.timed_out for r in failing)

    if timed_out:
        summary = 'Verification timed out'
    else:
        parts = []
        if any('tests failed' in r.summary for r in results):
            parts.append('tests failed')
        if any('lint issues' in r.summary for r in results):
            parts.append('lint issues')
        if any('type errors' in r.summary for r in results):
            parts.append('type errors')
        # task 3173: the three literals above are the ONLY fragments this
        # substring scan knows about, so a signal-kill note from
        # `_summarize_checks` matched none of them and a multi-subproject
        # verify silently degraded to a bare 'Failures: ' with no parts at
        # all — erasing the one fact that says the run produced no verdict.
        # Carry every distinct kill note through verbatim, in child order,
        # de-duplicated (two subprojects killed identically must not stutter
        # the same sentence twice).
        for r in results:
            for fragment in r.summary.removeprefix('Failures: ').split(', '):
                if SIGNAL_KILL_SUMMARY_MARKER in fragment and fragment not in parts:
                    parts.append(fragment)
        summary = 'All checks passed' if passed else f'Failures: {", ".join(parts)}'

    # Collect cause_hint from failing child results; join with ' | '.
    cause_hint = ' | '.join(r.cause_hint for r in results if r.cause_hint)

    # Pick the worst child category by priority.
    # Empty string is filtered out to avoid pulling the aggregate to '' when
    # legacy callers (no-persistence path) produced results with no category.
    # 'passed' is intentionally included when present — _worst_category correctly
    # orders failures above 'passed', so a mix of passing and failing children
    # still resolves to the worst failure.  If all children pass the aggregate
    # category will be 'passed', which is the correct result.
    child_categories = [r.category for r in results if r.category]
    category = _worst_category(child_categories) if child_categories else ''

    # Flatten per-child log path lists.
    worktree_log_paths: list[str] = []
    archive_log_paths: list[str] = []
    for r in results:
        worktree_log_paths.extend(r.worktree_log_paths)
        archive_log_paths.extend(r.archive_log_paths)

    # Task μ: aggregate failing_test_ids — None iff EVERY child's is None
    # (nothing collected anywhere: no merge+full module in this run, or
    # every junit report was unreadable), else the sorted, de-duped union of
    # every non-None child list. A None child contributes nothing but does
    # NOT suppress a sibling's collected ids, and a child that collected an
    # empty list ([] — "clean") still makes the aggregate non-None, distinct
    # from a child that never collected at all (None) — see
    # VerifyResult.failing_test_ids's docstring.
    _child_failing_ids = [r.failing_test_ids for r in results if r.failing_test_ids is not None]
    failing_test_ids = (
        sorted({fid for ids in _child_failing_ids for fid in ids}) if _child_failing_ids else None
    )

    # Task 3173 review amendment: union the per-leg categories across FAILING
    # children only, order-preserved and de-duplicated (same deterministic
    # style as the kill-note fragment merge above).  A PASSING child has no
    # failing legs to report, so its value is irrelevant and never consulted.
    #
    # Deliberately UNLIKE failing_test_ids immediately above, which treats a
    # None child as "contributed nothing": here a FAILING child whose value is
    # None POISONS the aggregate to None.  The consumer (merge_queue's veto
    # gate) may only decline to veto when EVERY failing leg is verdict-less, so
    # an unrecorded failing child means the aggregate cannot make that claim —
    # fail CLOSED rather than silently answering for legs it never saw.
    failing_leg_categories: list[str] | None = []
    for r in failing:
        if r.failing_leg_categories is None:
            failing_leg_categories = None
            break
        for leg_category in r.failing_leg_categories:
            if leg_category not in failing_leg_categories:
                failing_leg_categories.append(leg_category)

    return VerifyResult(
        passed=passed,
        test_output=test_output,
        lint_output=lint_output,
        type_output=type_output,
        summary=summary,
        timed_out=timed_out,
        cause_hint=cause_hint,
        category=category,
        worktree_log_paths=worktree_log_paths,
        archive_log_paths=archive_log_paths,
        # Wall-clock approximation: modules run concurrently via asyncio.gather
        # so the slowest module dominates the total elapsed time.  Single-module
        # tasks hit the len==1 fast path above and carry the exact value.
        duration_secs=max((r.duration_secs for r in results), default=0.0),
        failing_test_ids=failing_test_ids,
        failing_leg_categories=failing_leg_categories,
    )


async def run_full_verification(
    project_root: Path,
    config: OrchestratorConfig,
    *,
    force_rediscover: bool = False,
    role: Literal['merge', 'task', 'background'] = 'task',
) -> VerifyResult:
    """Run verification for ALL subprojects against the project root.

    Unlike run_scoped_verification, this runs full (unscoped) test suites
    for every subproject that has an orchestrator.yaml. Used by review
    checkpoints to check integration health across the whole codebase.

    *role* is threaded to every internal :func:`run_verification` call
    (both the per-subproject fan-out and the no-subproject global-fallback
    branch).  Defaults to ``'task'``, which keeps the primary production
    caller ``review_checkpoint.py`` (and every other existing caller)
    byte-identical.  ``run_main_tip_sweep`` passes ``role='background'`` so
    the sweep's fan-out acquires the background admission slot and the
    nice-19/ionice-idle tier instead.

    Discovery reuse: ``config._module_configs`` uses a sentinel of ``None`` to
    mean "discovery never ran".  When it holds any dict (including ``{}``,
    meaning discovery ran and found no subprojects) and *project_root* resolves
    to the same absolute path as ``config.project_root``, the pre-discovered
    dict is reused directly and no additional filesystem walk is performed.  A
    fresh walk via ``_discover_module_configs`` is retained as a fallback for
    two cases: (1) *project_root* differs from ``config.project_root``; or (2)
    the config was constructed without going through ``load_config`` (e.g.
    direct instantiation in tests) so ``_module_configs`` is still ``None``.

    **Staleness note**: the reuse path reads a load_config-time snapshot, so a
    subproject whose ``orchestrator.yaml`` is added or merged *after* startup
    is absent from full verification for the remainder of the run.  This is an
    intentional trade-off: the snapshot eliminates the redundant filesystem walk
    on the hot production review-checkpoint path, where the vast majority of
    runs see a stable module set.  A mid-run addition is rare; documenting the
    trade-off and providing an explicit escape hatch is the right balance.

    To force fresh discovery (e.g. after a merge that introduces a new module),
    pass ``force_rediscover=True``.  The primary production caller
    ``review_checkpoint.py`` does **not** pass this flag — it keeps the fast
    snapshot path.  The ``force_rediscover`` parameter is keyword-only so call
    sites that opt in must state the intent explicitly.
    """
    from orchestrator.config import _discover_module_configs

    resolved = project_root.resolve()
    if not force_rediscover and config._module_configs is not None and resolved == config.project_root:
        module_configs = config._module_configs
    else:
        module_configs = _discover_module_configs(project_root)
    if not module_configs:
        logger.info('Full verification: no subproject configs — using global')
        return await run_verification(project_root, config, role=role)

    logger.info(
        'Full verification: running %d subprojects in parallel',
        len(module_configs),
    )
    results = await asyncio.gather(
        *(run_verification(project_root, config, mc, role=role) for mc in module_configs.values())
    )
    return _aggregate_results(list(results))


def _worktree_reader(
    worktree: Path | None,
    cache: dict[str, str | None] | None = None,
) -> Callable[[str], str | None]:
    """Build a ``verify_plan`` ``worktree_reader`` bound to *worktree*.

    Mirrors the ``is_file()`` + ``read_text(errors='replace')`` pattern used
    inline by ``scope_module_config``'s and ``_build_fallback_config``'s own
    structural-file content reads, so ``derive_verify_plan``'s STRUCTURAL
    detection (task γ, D2) sees exactly the same file content those
    functions do. Always answers ``None`` when *worktree* is ``None``
    (mirrors those functions' own ``worktree is not None`` guards) — no
    STRUCTURAL PlannedRun can be derived without a worktree to read from.

    *cache*, when given, memoizes each path's content (including a ``None``
    miss) for the lifetime of the returned reader. A caller that also passes
    the SAME dict to ``scope_module_config``/``_build_fallback_config``'s own
    ``content_cache`` parameter gets a touched file's content read from disk
    at most once — shared between the execution-scoping structural probe and
    this reader's use by ``derive_verify_plan`` — instead of once per
    consumer (task γ amendment, addressing a duplicate-I/O finding).
    ``run_scoped_verification``'s module_configs branch wires this; its
    fallback branch currently does not (see the "NOTE (task γ amendment)"
    comment at that call site).
    """
    def _read(path: str) -> str | None:
        if cache is not None and path in cache:
            return cache[path]
        if worktree is None:
            value = None
        else:
            full = worktree / path
            if not full.is_file():
                value = None
            else:
                try:
                    value = full.read_text(encoding='utf-8', errors='replace')
                except OSError:
                    value = None
        if cache is not None:
            cache[path] = value
        return value
    return _read


# Maps a PlannedRun's reason-string tool prefix to the ModuleConfig attribute
# it renders into. Used by _executed_module_configs_from_plan to recover
# which of test_command/lint_command/type_check_command a PlannedRun belongs
# to. Keyed off `reason` (verify_plan._derive_module_runs's docstring: "Each
# per-tool run's reason is prefixed with the tool name ... so a caller can
# recover tool identity even for a SKIPPED slot") rather than `run.cmd.tool`,
# because an OPAQUE-parsed command — an unrecognised head, e.g. a synthetic
# test-fixture command or dark_factory's real multi-clause fleet lint/type
# `&&`-chains (config.yaml) — always resolves to the SAME `ToolKind.OPAQUE`
# regardless of which slot produced it, so `cmd.tool` alone cannot tell an
# OPAQUE lint command apart from an OPAQUE test command. A SKIPPED run's cmd
# is None and needs no attribution, since the corresponding ModuleConfig
# field is simply left unset (None) there.
_MC_ATTR_BY_REASON_PREFIX: tuple[tuple[str, str], ...] = (
    ('lint:', 'lint_command'),
    ('pyright:', 'type_check_command'),
    ('pytest:', 'test_command'),
)


def _mc_attr_for_run_or_none(run: verify_plan.PlannedRun) -> str | None:
    """Resolve which ModuleConfig attribute *run* renders into, from its reason prefix.

    Returns ``None`` (rather than raising) when *run*.reason carries no
    recognised tool prefix — the shape ``_derive_fallback_runs`` emits for its
    single "no .py files touched" SKIPPED run (e.g. a fallback diff whose only
    source files are ``.rs``), which is not keyed to any of the three
    ``ModuleConfig`` tool slots. :func:`_mc_attr_for_run` is the raising
    variant for callers that have already excluded that shape.
    """
    for prefix, attr in _MC_ATTR_BY_REASON_PREFIX:
        if run.reason.startswith(prefix):
            return attr
    return None


def _mc_attr_for_run(run: verify_plan.PlannedRun) -> str:
    """Resolve which ModuleConfig attribute *run* renders into, from its reason prefix."""
    attr = _mc_attr_for_run_or_none(run)
    if attr is None:
        raise ValueError(f'PlannedRun.reason has no recognised tool prefix: {run.reason!r}')
    return attr


def _executed_module_configs_from_plan(
    module_configs: list[ModuleConfig],
    plan: verify_plan.VerifyPlan,
) -> list[ModuleConfig]:
    """Build the ModuleConfig(s) *plan* prescribes running, for the module-config branch.

    The plan->execution bridge (task κ, verify-scope-inversion-prd.md):
    groups *plan*.runs by module_prefix and, for each *mc* in
    *module_configs*, rebuilds its three commands from the corresponding
    PlannedRun's scope_kind:

    - SKIPPED -> ``None`` (that check does not run).
    - FULL_SUITE -> *mc*'s OWN verbatim command (preserves e.g.
      ``--directory``, matching ``scope_module_config``'s has_conftest/
      has_test_data/has_structural branches, which reuse
      ``mc.test_command``/``mc.type_check_command`` unmodified).
    - FILE_SCOPED -> ``render(run.cmd)`` (identical to ``_scope_to_keyword``'s
      output, which itself renders a VerifyCmd — see ``verify_cmd.render``).

    A module whose ONLY run is the "no files under prefix" SKIPPED emitted by
    ``verify_plan._derive_module_runs`` (a single-element runs list — the
    normal 3-run-per-module shape never collapses to one) is dropped from the
    returned list entirely, mirroring ``scope_module_config`` returning
    ``None`` for a subproject with zero matching files: the caller must skip
    that subproject rather than run its full unscoped suite.

    Uses ``dataclasses.replace`` (imported as ``replace``) rather than a
    hand-listed ``ModuleConfig(...)`` reconstruction, so every other
    ModuleConfig field — lock_depth, max_per_module, module_overrides,
    verify_command_timeout_secs, verify_cold_command_timeout_secs,
    concurrent_verify, verify_env, scope_cargo — survives onto the executed
    config unchanged (the same pattern :func:`_apply_cargo_scope` already
    uses), so ``run_verification``'s resolvers see identical per-module
    overrides to what they'd have seen from *mc* directly.

    Guard: a TRIVIAL *plan* (``derive_verify_plan``'s top-level "no
    .py/.rs file at all" short-circuit — a single run with
    ``module_prefix=''``, which matches no real ``mc.prefix``) means every
    module has zero matching ``.py`` files by construction, so this returns
    ``[]`` immediately rather than mis-reading the absent per-module lookup
    as "no commands configured" and emitting a vacuous ModuleConfig per
    module (``scope_module_config`` returns ``None`` — excluded — for every
    *mc* in this case too).
    """
    if len(plan.runs) == 1 and plan.runs[0].scope_kind is verify_plan.ScopeKind.TRIVIAL:
        return []

    runs_by_prefix: dict[str, list[verify_plan.PlannedRun]] = {}
    for run in plan.runs:
        runs_by_prefix.setdefault(run.module_prefix, []).append(run)

    executed: list[ModuleConfig] = []
    for mc in module_configs:
        runs = runs_by_prefix.get(mc.prefix, [])
        only_run = runs[0] if len(runs) == 1 else None
        if (
            only_run is not None
            and only_run.cmd is None
            and only_run.scope_kind is verify_plan.ScopeKind.SKIPPED
        ):
            continue

        commands: dict[str, str | None] = {}
        for run in runs:
            if run.cmd is None:
                continue  # SKIPPED slot — the matching field stays unset (None) below
            attr = _mc_attr_for_run(run)
            commands[attr] = (
                getattr(mc, attr)
                if run.scope_kind is verify_plan.ScopeKind.FULL_SUITE
                else render(run.cmd)
            )

        executed.append(replace(
            mc,
            test_command=commands.get('test_command'),
            lint_command=commands.get('lint_command'),
            type_check_command=commands.get('type_check_command'),
        ))
    return executed


# Maps a fallback PlannedRun's ModuleConfig attribute (as resolved by
# _mc_attr_for_run, reused here) to the ToolKind its raw-wrapped executed
# VerifyCmd should carry in _executed_fallback_plan. Fixed rather than
# derived from the DECISION run's own `cmd.tool` because a SKIPPED decision
# run's `cmd` is None — there is no tool to read off it — yet the tool
# identity is still needed to wrap an execution-time value that turned out
# non-None (e.g. the decision skipped pytest for lack of a real suite, but a
# subproject rescoping might yet resolve one).
_FALLBACK_TOOLKIND_BY_ATTR: dict[str, ToolKind] = {
    'test_command': ToolKind.PYTEST,
    'lint_command': ToolKind.RUFF,
    'type_check_command': ToolKind.PYRIGHT,
}


def _executed_fallback_plan(
    plan: verify_plan.VerifyPlan,
    fallback: ModuleConfig,
) -> verify_plan.VerifyPlan:
    """Rebuild *plan* (``derive_verify_plan``'s fallback-branch DECISION record)
    into the EXECUTED plan, using *fallback* — the ``ModuleConfig``
    :func:`_build_fallback_config` (plus :func:`_apply_cargo_scope`) actually
    produced — as the ground truth for WHAT ran.

    The plan->execution bridge for the fallback branch (task κ,
    verify-scope-inversion-prd.md), mirroring
    :func:`_executed_module_configs_from_plan`'s role for the module-config
    branch but in the OPPOSITE direction: rather than building an executed
    ``ModuleConfig`` FROM the plan, this rebuilds the EXECUTED PLAN from the
    already-executed ``ModuleConfig``, since ``_build_fallback_config``'s
    filesystem-dependent rendering (subproject cd-scoping, mixed
    root+subproject scoping, TYPE/LINT uv-context rescoping, OPAQUE
    fleet-chain first-clause scoping — see its own docstring) is retained
    unchanged as the fallback's execution-layer renderer rather than being
    reimplemented here (or threaded a `plan` parameter to consume directly:
    :func:`_build_fallback_config`'s call site must keep its exact current
    arguments — ``TestRunScopedVerificationForwardsWorktreeToFallback``
    replaces it with a fixed ``(task_files, config=None, worktree=None)``
    fake with no ``**kwargs`` catch-all, so any new call-site keyword breaks
    it, the same constraint the "NOTE (task γ amendment)" comment at that
    call site already documents for ``content_cache``).

    Each of *plan*'s runs (keyed by the same tool-prefixed *reason*
    convention :func:`_mc_attr_for_run` relies on) is replaced by a new
    ``PlannedRun`` whose ``module_prefix`` is *fallback*.prefix (the ACTUAL
    subproject/fallback identity that ran — e.g. ``'cockpit'``, not the
    idealized flat ``'__fallback__'``) and whose ``cmd`` is a raw
    pass-through ``VerifyCmd`` wrapping *fallback*'s corresponding field
    verbatim (``None`` when that field is ``None`` — an execution-time skip,
    which may not match the decision's own ``scope_kind``, e.g. a subproject
    diff with no derivable pytest target). ``scope_kind``/``reason`` carry
    over from the decision unchanged: they answer WHY a tool slot was scoped
    a given way, not WHERE/HOW it actually ran.

    A run with no recognised tool prefix — ``_derive_fallback_runs``'s single
    "no .py files touched" SKIPPED run, returned when *plan* was derived from
    an ``existing_files`` list with zero ``.py`` files (e.g. a diff touching
    only ``.rs`` sources) — is passed through unchanged rather than resolved
    against :func:`_mc_attr_for_run`: it is not keyed to any of *fallback*'s
    three tool slots, so there is nothing to reconcile it against. (In
    practice ``run_scoped_verification`` only calls this when
    ``_build_fallback_config`` returned non-``None``, which requires at least
    one ``.py`` file — the same precondition under which
    ``_derive_fallback_runs`` never emits this shape — so this guard is
    defense-in-depth against that precondition drifting rather than a
    presently-reachable path.)
    """
    executed_runs = []
    for run in plan.runs:
        attr = _mc_attr_for_run_or_none(run)
        if attr is None:
            executed_runs.append(run)
            continue
        executed_cmd_str = getattr(fallback, attr)
        cmd = (
            VerifyCmd(tool=_FALLBACK_TOOLKIND_BY_ATTR[attr], raw=executed_cmd_str)
            if executed_cmd_str is not None
            else None
        )
        executed_runs.append(replace(run, module_prefix=fallback.prefix, cmd=cmd))
    return replace(plan, runs=tuple(executed_runs))


def _safe_derive_verify_plan_dict(
    existing_files: list[str],
    module_configs: list[ModuleConfig],
    config: OrchestratorConfig,
    worktree_reader: Callable[[str], str | None],
    *,
    role: Literal['merge', 'task'],
    executed_fallback: ModuleConfig | None = None,
) -> dict | None:
    """Best-effort executed-plan dict for ``VerifyResult.plan``.

    Used by the fallback (no-``module_configs``) branch of
    :func:`run_scoped_verification` — the module-config branch derives its
    plan directly and executes it (see
    :func:`_executed_module_configs_from_plan`), so this helper's job is
    giving the fallback branch the same ``VerifyResult.plan`` treatment: it
    best-effort-builds the EXECUTED plan record without re-deriving it
    inline. When *executed_fallback* is given (the fallback branch's own
    already-executed ``ModuleConfig``), the derived DECISION plan is folded
    through :func:`_executed_fallback_plan` first, so the returned dict
    records what actually ran (module_prefix + rendered command) rather than
    the idealized flat ``'__fallback__'`` decision alone. A bug in the
    decision layer or this reconciliation — an unforeseen ``VerifyCmd``/
    dataclass edge, or a future change to ``_verify_cmd_to_dict``/``to_dict``
    — must never fail an otherwise-passing verify attempt just because its
    plan record couldn't be built. Catches broadly and logs a warning,
    returning ``None`` (``VerifyResult.plan``'s own default) on any failure
    instead of propagating and failing the gate.
    """
    try:
        plan = verify_plan.derive_verify_plan(
            existing_files, module_configs, config, worktree_reader, role=role,
        )
        if executed_fallback is not None:
            plan = _executed_fallback_plan(plan, executed_fallback)
        return plan.to_dict()
    except Exception as exc:  # noqa: BLE001 — best-effort; must never fail the verify gate
        logger.warning(
            'derive_verify_plan failed — omitting VerifyResult.plan for this attempt: %s',
            exc, exc_info=True,
        )
        return None


def _reverse_dependency_module_configs(
    changed_files: list[str],
    config: OrchestratorConfig,
    worktree: Path,
    already_scoped: set[str],
    content_cache: dict[str, str | None] | None = None,
) -> list[ModuleConfig]:
    """Widen scoped merge-verify to a depended-upon package's reverse-dependents (task 2607).

    Closes the blind spot where a task's diff scoped to orchestrator/ SOURCE
    never runs escalation/'s coupled cross-package tests — module_configs is
    resolved from the TASK's own touched modules only, so escalation's
    ModuleConfig is never a candidate for an orchestrator-only diff. This
    recurred 3x as a RED-main fix-forward (1736->1761, 2173->2038,
    2435->2604) before being structurally closed here.

    *changed_files* should be the RAW (pre-existence-filter) task file list
    — NOT one narrowed to files that still exist on disk (amendment, review
    suggestion 2). :func:`verify_plan.reverse_dependent_test_targets`'s
    trigger gate matches purely on path shape (``<pkg>/src/**.py``), so a
    deleted or renamed-away source path still correctly counts as a
    trigger — precisely the kind of change most likely to break a
    dependent's ``from <pkg> import ...`` contract. An existence-filtered
    list would silently under-trigger whenever the only surviving
    (still-existing) file under the depended-upon package's own prefix is a
    non-source file, e.g. a test edit made alongside the source deletion.
    Note this only helps when the caller reaches this function at all:
    :func:`run_scoped_verification`'s ``if not scoped:`` early-exit means a
    diff with NO surviving file under the task's own module_configs never
    calls this helper in the first place, regardless of *changed_files* —
    that residual gap is architectural (a property of the surrounding
    scoping branch), not something this helper can close on its own.

    Thin impure wrapper around the pure
    :func:`verify_plan.reverse_dependent_test_targets`: builds a
    ``list_pkg_tests`` closure (rglobs ``worktree/<pkg>/tests`` for ``.py``
    files, mapped to worktree-relative POSIX paths, filtered to genuinely
    pytest-collectable files via ``verify_plan._is_collectable_test_file`` —
    mirrors how ``derive_verify_plan`` selects collectable tests) and a
    ``read_content`` reader via :func:`_worktree_reader` (threaded with the
    caller's *content_cache*, so a file already read elsewhere this attempt
    is not read from disk twice).

    For each ``(dependent, coupled_files)`` the pure function returns: skips
    it when *dependent* is already in *already_scoped* (already covered by
    the task's own module_configs, or a prior widening — no double-add);
    looks up its BASE ``ModuleConfig`` via ``config.module_configs_or_empty``
    and skips it when absent or when it has no ``test_command`` configured
    (nothing to scope); else narrows ``test_command`` to *coupled_files* via
    :func:`_scope_to_keyword` (the same scoping :func:`_derive_module_runs`
    uses for an in-diff FILE_SCOPED pytest run). When that scoping is a
    no-op — :func:`_scope_to_keyword` returns *cmd* UNCHANGED when its
    ``'pytest'`` keyword isn't literally present or the prefix doesn't parse
    into a structured command, per its own documented contract — widening
    with the untouched command would silently run the dependent's FULL
    suite instead of just the coupled files, contradicting the
    cost/flake-bounding goal (design decision: import-scoped, not
    whole-suite). So that dependent is skipped (with a warning logged
    rather than silently dropped — amendment, review suggestion 3) instead
    of being widened unscoped. Otherwise appends a ``dataclasses.replace``
    of the base config with the scoped ``test_command`` and
    ``lint_command``/``type_check_command`` forced to ``None`` — the
    widening is pytest-only (design decision: the failure class that
    recurred is a runtime import/attribute break, and reverse-dependency
    pyright is already covered by hooks/project-checks, task 2551). Every
    other ``ModuleConfig`` field (lock_depth, verify_env, timeouts, ...)
    survives via ``replace`` so the widened run uses the dependent's normal
    per-module overrides.
    """
    def _list_pkg_tests(pkg: str) -> list[str]:
        tests_dir = worktree / pkg / 'tests'
        if not tests_dir.is_dir():
            return []
        rel_paths = (p.relative_to(worktree).as_posix() for p in tests_dir.rglob('*.py'))
        return [p for p in rel_paths if verify_plan._is_collectable_test_file(p)]

    read_content = _worktree_reader(worktree, cache=content_cache)

    triggered = verify_plan.reverse_dependent_test_targets(
        changed_files, verify_plan._REVERSE_TEST_DEPENDENTS, _list_pkg_tests, read_content,
    )

    widened: list[ModuleConfig] = []
    for dependent, coupled in triggered:
        if dependent in already_scoped:
            continue
        base = config.module_configs_or_empty.get(dependent)
        if base is None or not base.test_command:
            continue
        scoped_test_command = _scope_to_keyword(base.test_command, 'pytest', coupled)
        if scoped_test_command == base.test_command:
            # _scope_to_keyword no-op (no literal 'pytest' keyword match, or
            # an unparseable command prefix) — widening this dependent
            # would silently fan out to its FULL, un-narrowed suite.  Skip
            # rather than run unscoped; log so a mis-shaped test_command on
            # a mapped dependent is visible instead of silently missing
            # from merge-verify coverage.
            logger.warning(
                "Reverse-dependency widening: %s's test_command could not be "
                "scoped to %d coupled file(s) via 'pytest' keyword-matching "
                '(no literal match, or an unparseable command prefix) — '
                'skipping widening for this dependent rather than running '
                'its full suite unscoped',
                dependent, len(coupled),
            )
            continue
        widened.append(replace(
            base,
            test_command=scoped_test_command,
            lint_command=None,
            type_check_command=None,
        ))
    return widened


async def run_scoped_verification(
    worktree: Path,
    config: OrchestratorConfig,
    module_configs: list[ModuleConfig],
    task_files: list[str] | None = None,
    *,
    max_retries: int | None = None,
    is_merge_verify: bool = False,
    attempt_id: int | None = None,
    task_id: str | None = None,
    archive_root: Path | None = None,
    force_workspace: bool = False,
    role: Literal['merge', 'task'] = 'task',
    event_store: 'EventStore | None' = None,
) -> VerifyResult:
    """Run verification scoped to specific subprojects and optionally to task files.

    Scoping modes (in priority order):

    1. **File-scoped within subprojects** — when *module_configs* is non-empty
       and *task_files* is provided, each ModuleConfig's commands are narrowed
       to the specific files via :func:`scope_module_config`.  Subprojects
       with zero matching files are skipped entirely.
    2. **Fallback-scoped** — when *module_configs* is empty and *task_files* is
       provided, a synthetic ModuleConfig is built via
       :func:`_build_fallback_config`, bypassing the global commands entirely.
    3. **Global** — when *task_files* is ``None`` (or falsy) with no
       module_configs, or when fallback returns ``None`` (no .py files).
    4. **Workspace (train-member override)** — when *force_workspace* is
       ``True``, all scoping is bypassed and the project-wide commands from
       *config* (e.g. ``cargo test --workspace``) run against the worktree
       branch tip.  Mirrors the *is_merge_verify* flag-threading pattern
       (PRD §9.5, γ₁).  Non-train callers leave this at its default ``False``
       for byte-identical existing behaviour.

    *max_retries* overrides ``config.verify_timeout_retries`` for this call;
    pass ``0`` from the merge-queue path so a deterministic hang doesn't
    triple the stall.

    *is_merge_verify* is forwarded unchanged to every :func:`run_verification`
    call.  Merge worktrees are freshly created (no warm cargo cache) but lack
    ``.task/``, which would otherwise misclassify them as warm via
    :func:`_is_verify_cold`.  Set this to ``True`` from the merge-queue
    call sites so post-merge verifies get the cold timeout.
    """
    scope_cargo_enabled = config.scope_cargo

    # Bound the per-subproject fan-out so a large (or accidentally polluted)
    # module set can never launch an unbounded number of full builds into one
    # worktree at once.  Created per call; a no-op when only one module runs.
    # (Root cause of the 226-way merge-verify storm: a polluted module set
    # turned the no-match fan-out into 226 concurrent `cargo` pipelines in one
    # `_merge-*` worktree.)
    #
    # The cap is role-aware (task 2393, T5): merge-role pytests bypass the T2
    # counting admission slot (`_admission_slot` no-ops for role='merge' — the
    # anti-livelock/C-merge-priority guarantee), so merge's internal fan-out
    # needs its OWN bound (`merge_verify_max_concurrent_modules`), orthogonal
    # to `verify_admission_task_slots`. The 'task' role (this function's only
    # other role — see the `Literal['merge', 'task']` signature above) keeps
    # the general `max_concurrent_module_verifies` — its pytests are
    # additionally bounded by the admission slot, so the general knob mostly
    # just caps burst concurrency for it.
    _fanout_cap = (
        config.merge_verify_max_concurrent_modules
        if role == 'merge'
        else config.max_concurrent_module_verifies
    )
    _fanout_sem = asyncio.Semaphore(max(1, _fanout_cap))

    async def _verify_module(mc: ModuleConfig) -> 'VerifyResult':
        async with _fanout_sem:
            return await run_verification(
                worktree, config, mc,
                max_retries=max_retries,
                is_merge_verify=is_merge_verify,
                attempt_id=attempt_id, task_id=task_id, archive_root=archive_root,
                role=role,
            )

    # When the plan didn't provide a file list, try to derive one from git —
    # but only when we're not bypassing scoping entirely (force_workspace=True
    # goes straight to the global run_verification call without needing a file
    # list).
    if task_files is None and not force_workspace:
        task_files = await _derive_task_files_from_git(worktree, config)

    # _prune_archive runs exactly once in the finally block regardless of which
    # branch returns, preventing concurrent per-module prune races and removing
    # the repetitive guard at every return site.
    try:
        # Train-member workspace override: bypass ALL scoping and run the
        # project-wide workspace command verbatim (config.test_command/…).
        # Cargo --workspace→-p rewrites only fire when task_files are passed,
        # so this path runs `cargo test --workspace` (or the configured command)
        # unchanged against the worktree branch tip.
        if force_workspace:
            # λ (task 2589, T1): merge_verify_breadth forks WHAT this
            # bypassed-scoping path executes, not WHETHER it executes.
            # role=='merge' + breadth=='full' replaces the single OPAQUE
            # global command below with a per-module full-suite fan-out
            # across every REGISTERED module (config.module_configs_or_empty),
            # reusing the SAME _derive_full_suite_runs /
            # _executed_module_configs_from_plan bridge the module_configs-
            # branch merge+full expansion above uses (PRD Resolved decision
            # 2: the broad gate must be per-module parseable commands, never
            # the opaque chain). breadth=='scoped' (the shipped default)
            # falls through to the legacy single global call below,
            # byte-identical (R4 rollback golden). This is the
            # merge_verify_workspace=True routing
            # verify_runner.LocalRunner._run threads role='merge' into; the
            # role=='task' train-member override (workflow.py,
            # _run_scoped_verification_with_infra_retry) never takes this
            # fork — breadth is merge-role-gated only, so that call stays on
            # the legacy path unconditionally regardless of the knob.
            if role == 'merge' and verify_plan._merge_breadth_is_full(config):
                registered_modules = list(config.module_configs_or_empty.values()) or module_configs
                if registered_modules:
                    plan = verify_plan.VerifyPlan(runs=tuple(
                        run
                        for mc in registered_modules
                        for run in verify_plan._derive_full_suite_runs(mc, role)
                    ))
                    scoped = _executed_module_configs_from_plan(registered_modules, plan)
                    # Guard this block's own stated intent ("degrade ...
                    # rather than silently verifying nothing"): a registered
                    # module always survives into `scoped` here —
                    # _derive_full_suite_runs always emits 3 runs per module
                    # (never the single-SKIPPED shape
                    # _executed_module_configs_from_plan drops) — but if
                    # EVERY module has zero configured commands (all
                    # lint/pyright/test None), every surviving ModuleConfig
                    # is itself all-None, and gathering over them would be a
                    # vacuous per-module pass that aggregates to an overall
                    # passed=True with nothing actually executed. Only take
                    # this path when at least one module contributes at
                    # least one real command; otherwise fall through to the
                    # legacy global call below (loud-over-silent-degradation).
                    if any(
                        mc.test_command or mc.lint_command or mc.type_check_command
                        for mc in scoped
                    ):
                        logger.info(
                            'Verification mode: workspace (merge_verify_breadth=full — '
                            'per-module full suite across %d registered module(s))',
                            len(registered_modules),
                        )
                        results = await asyncio.gather(*(_verify_module(mc) for mc in scoped))
                        aggregated = _aggregate_results(list(results))
                        aggregated.plan = plan.to_dict()
                        return aggregated
                    logger.warning(
                        'Verification mode: workspace (merge_verify_breadth=full) found '
                        'no configured commands across %d registered module(s) — '
                        'falling back to the legacy global command rather than '
                        'silently verifying nothing',
                        len(registered_modules),
                    )
                # No registered modules AND nothing passed to fall back to —
                # degrade to the legacy global call below rather than
                # silently verifying nothing.
            logger.info(
                'Verification mode: workspace (train member — file-scoping bypassed)'
            )
            return await run_verification(
                worktree, config,
                max_retries=max_retries,
                is_merge_verify=is_merge_verify,
                attempt_id=attempt_id,
                task_id=task_id,
                archive_root=archive_root,
                role=role,
            )
        if module_configs:
            # λ (task 2589, R1): the broad merge gate. role=='merge' +
            # merge_verify_breadth=='full' expands module_configs from the
            # FULL registry (config.module_configs_or_empty) — not just the
            # task's/train's OWN modules passed in — so every REGISTERED
            # module is covered, not only the ones this diff happens to
            # touch (the gap the task-role pytest floor, R3, deliberately
            # leaves open — see verify_plan._derive_full_suite_runs'
            # docstring). Placed ahead of BOTH the file-scoped and
            # unscoped-fan-out sub-branches below, so it broadens either
            # execution path uniformly. An empty registry (e.g. a
            # direct-instantiated config in most unit tests) falls back to
            # the passed module_configs unchanged — degrades safely rather
            # than silently verifying nothing.
            if role == 'merge' and verify_plan._merge_breadth_is_full(config):
                module_configs = list(config.module_configs_or_empty.values()) or module_configs
            # Apply file-level scoping within each subproject when task_files given
            if task_files:
                # Filter to files that still exist — tasks may delete files as part of their work
                existing_files = [f for f in task_files if (worktree / f).exists()]
                # Shared structural-content cache (task γ amendment): threaded
                # into derive_verify_plan's worktree_reader below, so a
                # touched file is read from disk at most once per attempt
                # instead of once per (module, observability) consumer.
                _content_cache: dict[str, str | None] = {}
                # Plan-authoritative execution (task κ,
                # verify-scope-inversion-prd.md): derive_verify_plan is the
                # SOLE decision tree for module scope now — classify_file
                # runs exactly once per touched file here, not a second time
                # in a hand-mirrored scope_module_config tree.
                # _executed_module_configs_from_plan renders each module's
                # PlannedRuns into the ModuleConfig run_verification actually
                # executes; a module whose only run is "no files under
                # prefix" is dropped entirely (mirrors scope_module_config's
                # `return None` "caller must skip this subproject" contract).
                plan = verify_plan.derive_verify_plan(
                    existing_files, module_configs, config,
                    _worktree_reader(worktree, cache=_content_cache), role=role,
                )
                scoped = _executed_module_configs_from_plan(module_configs, plan)
                scoped_prefixes = {mc.prefix for mc in scoped}
                skipped = [mc.prefix for mc in module_configs if mc.prefix not in scoped_prefixes]
                if skipped:
                    logger.info(
                        'Verification scope: skipping %d subproject(s) with no matching files: %s',
                        len(skipped), ', '.join(skipped),
                    )
                if not scoped:
                    # No subproject has matching files. Two sub-cases:
                    #   (a) Diff has no .py/.rs at all (docs, YAML, JSON …) —
                    #       every existing scope branch would no-op anyway; the
                    #       previous global-pytest fall-through was unsafe in
                    #       this layout. Trivially pass — UNLESS the merge-role
                    #       verify-pipeline-guard says a full gate is required
                    #       (e.g. diff touches verify.sh which shifts plan-line
                    #       counts — the drift-ambush class).
                    #   (b) Source files exist but don't fit any prefix
                    #       (e.g. root-level conftest.py + skills/*.md). Fan
                    #       out per-subproject so each runs in its own venv
                    #       with its own pyproject options.
                    if not _has_source_files(existing_files):
                        if role == 'merge' and is_merge_verify:
                            # INV-1 (task 2883): the ADOPTABLE merge verdict must
                            # never trivially pass a no-evidence resolution. Any
                            # 'nothing to run' outcome (no source files, or an
                            # empty existing_files set from an ENOENT-clobbered
                            # worktree) escalates to the full per-subproject gate
                            # when at least one module carries a real command;
                            # otherwise it FAILs loud rather than vouching for a
                            # tree no gate ever ran on.
                            reason = _trivial_pass_reason(existing_files)
                            if any(
                                mc.test_command or mc.lint_command
                                or mc.type_check_command
                                for mc in module_configs
                            ):
                                _emit_trivial_pass_escalated(
                                    event_store, task_id,
                                    reason=reason, resolution='full_gate',
                                )
                                logger.info(
                                    'INV-1: merge gate escalating would-be trivial'
                                    ' pass (%s) to the per-subproject full gate',
                                    reason,
                                )
                                # Fall through to per-subproject fan-out below.
                            else:
                                _emit_trivial_pass_escalated(
                                    event_store, task_id,
                                    reason=reason, resolution='loud_fail',
                                )
                                logger.warning(
                                    'INV-1: merge gate has no full-gate command'
                                    ' (%s) — failing loud (merge_no_evidence)',
                                    reason,
                                )
                                return _merge_no_evidence_fail(reason)
                        else:
                            # Cheap deterministic backstop (task 2838) OR'd first so
                            # it short-circuits the verify-pipeline-guard.sh
                            # subprocess on the merge hot path; empty globs (default)
                            # → False → expression byte-identical to guard-only.
                            # The backstop matches the FULL changed set (task_files)
                            # — including DELETED manifest-relevant paths, which are
                            # absent from existing_files — because removing a file a
                            # manifest enumerates shifts the manifest just as adding
                            # one does (reviewer amendment, task 2838). The reify
                            # consult keeps its existing on-disk existing_files
                            # contract.
                            should_override = role == 'merge' and (
                                _merge_config_only_diff_forces_full_gate(config, task_files)
                                or await _verify_pipeline_guard_requires_full_gate(
                                    worktree, existing_files,
                                )
                            )
                            if should_override:
                                logger.info(
                                    'config-only fast-path overridden by manifest-drift'
                                    ' backstop or verify-pipeline-guard'
                                    ' — running full gate (module_configs merge path)',
                                )
                                # Fall through to per-subproject fan-out below.
                            else:
                                logger.info(
                                    'Verification mode: trivial pass (no source files in diff)',
                                )
                                return _trivial_pass(
                                    'No source files changed — verify trivially passes',
                                )
                    logger.info(
                        'Verification mode: per-subproject fan-out (%d subprojects)',
                        len(module_configs),
                    )
                    results = await asyncio.gather(*(
                        _verify_module(mc) for mc in module_configs
                    ))
                    return _aggregate_results(list(results))
                # Rewrite cargo --workspace → cargo -p <crate> when all task files
                # are .rs and map to known workspace crates.
                scoped = [
                    _apply_cargo_scope(mc, existing_files, worktree, scope_cargo_enabled)
                    for mc in scoped
                ]
                n_files = len(existing_files)
                n_mods = len(scoped)
                logger.info('Verification mode: file-scoped (%d files across %d subprojects)', n_files, n_mods)
                # `plan` (derived once, above) IS the execution driver for
                # this branch — `scoped` was built from it by
                # _executed_module_configs_from_plan. Reuse the same object
                # for VerifyResult.plan rather than deriving a second time.
                plan_dict = plan.to_dict()
                logger.info('Verify plan: %s', plan_dict)
                # Reverse-dependency test widening (task 2607): the plan
                # above is authoritative for file-classification scope (task
                # κ, verify-scope-inversion-prd.md) over the task's OWN
                # module_configs — but a diff scoped to orchestrator/ SOURCE
                # alone never puts escalation's ModuleConfig in
                # module_configs, so escalation's coupled cross-package
                # merge_queue tests never ran. That blind spot caused
                # RED-main fix-forward 3x (1761/2038/2604), each patched
                # reactively. Append any reverse-dependent test targets to
                # the EXECUTED `scoped` list rather than folding them into
                # `plan`/`plan_dict` — derive_verify_plan would SKIP
                # escalation (no escalation files changed; the widening
                # keys off the orchestrator SOURCE change, orthogonal to
                # file-classification scope) — see design decision 6.
                # Mirrors the pyright-only reverse-dep precedent in
                # hooks/project-checks (task 2551), scoped to pytest/import
                # coupling instead of whole-package lint fan-out.
                #
                # `scoped` (surviving) union `skipped` (no matching files) is
                # every mc.prefix in `module_configs` — the task's OWN
                # modules — by construction (skipped is defined as
                # module_configs minus scoped_prefixes above). So when the
                # task's own diff already touches escalation, 'escalation' is
                # already in already_scoped and the helper's own dedup
                # (`if dependent in already_scoped: continue`) skips it — no
                # double-add. Pinned by TestRunScopedVerificationReverse-
                # DependencyGuards (test_verify_reverse_dep.py, task 2607
                # step-9): orchestrator-test-only diffs don't trigger (the
                # verify_plan `<pkg>/src/` gate), an escalation-in-diff task
                # isn't double-widened (this union), and a no-map-entry
                # package (e.g. dashboard) widens to nothing.
                #
                # Merge-only (amendment, review suggestion 4): every design
                # decision above is framed as "merge-verify path" / "merge-
                # path flake/cost surface" — the 3x recurrence this closes
                # was always a merge-time escape, never a task-role dev-loop
                # gap. Gating on role=='merge' leaves per-task verify
                # latency for every orchestrator-source task unchanged; a
                # task-role diff still gets its normal coverage plus the
                # whole-tree main-tip sweep and hooks/project-checks'
                # pyright reverse-dep backstops.
                #
                # `task_files` — the RAW, pre-existence-filter list, NOT
                # `existing_files` — drives the trigger gate (amendment,
                # review suggestion 2): a deleted/renamed orchestrator
                # SOURCE path still counts as a trigger, since
                # reverse_dependent_test_targets matches on path shape
                # alone, never on-disk existence. See
                # _reverse_dependency_module_configs' docstring for the
                # residual gap this narrows but does not fully close (a
                # diff with no surviving file under the task's own
                # module_configs never reaches this call at all).
                already_scoped = {mc.prefix for mc in scoped} | set(skipped)
                widened = (
                    _reverse_dependency_module_configs(
                        task_files, config, worktree, already_scoped,
                        content_cache=_content_cache,
                    )
                    if role == 'merge'
                    else []
                )
                if widened:
                    logger.info(
                        'Verification scope: widening to %d reverse-dependent subproject(s): %s',
                        len(widened), ', '.join(mc.prefix for mc in widened),
                    )
                    scoped = scoped + widened
            else:
                scoped = module_configs
                plan_dict = None
                logger.info('Verification mode: subproject-scoped (%d subprojects)', len(module_configs))
            results = await asyncio.gather(
                *(_verify_module(mc) for mc in scoped)
            )
            aggregated = _aggregate_results(list(results))
            if plan_dict is not None:
                aggregated.plan = plan_dict
            return aggregated

        # No module_configs — try fallback or global
        if task_files:
            # Filter to files that still exist — tasks may delete files as part of their work
            existing_files = [f for f in task_files if (worktree / f).exists()]
            # Mirror the same docs-only short-circuit as the module_configs
            # branch: with no .py/.rs files _build_fallback_config would
            # return None and we'd fall through to the unsafe global pytest.
            if not _has_source_files(existing_files):
                if role == 'merge' and is_merge_verify:
                    # INV-1 (task 2883): the ADOPTABLE merge verdict must never
                    # trivially pass a no-evidence resolution. Escalate to the
                    # global full gate when a global command exists (fall through:
                    # _build_fallback_config→None for a no-source diff, cargo-scope
                    # is skipped, and control reaches the global run_verification
                    # tail which runs config.test_command/lint/type). If NO global
                    # command exists, FAIL loud INLINE so the global-tail backstop
                    # never double-emits.
                    reason = _trivial_pass_reason(existing_files)
                    if (
                        config.test_command
                        or config.lint_command
                        or config.type_check_command
                    ):
                        _emit_trivial_pass_escalated(
                            event_store, task_id,
                            reason=reason, resolution='full_gate',
                        )
                        logger.info(
                            'INV-1: merge gate escalating would-be trivial pass'
                            ' (%s) to the global full gate',
                            reason,
                        )
                        # Fall through to the global run_verification tail below.
                    else:
                        _emit_trivial_pass_escalated(
                            event_store, task_id,
                            reason=reason, resolution='loud_fail',
                        )
                        logger.warning(
                            'INV-1: merge gate has no global full-gate command'
                            ' (%s) — failing loud (merge_no_evidence)',
                            reason,
                        )
                        return _merge_no_evidence_fail(reason)
                else:
                    # Cheap deterministic backstop (task 2838) OR'd first so it
                    # short-circuits the verify-pipeline-guard.sh subprocess on the
                    # merge hot path; empty globs (default) → False → expression
                    # byte-identical to the guard-only behaviour.
                    # The backstop matches the FULL changed set (task_files) —
                    # including DELETED manifest-relevant paths, which are absent
                    # from existing_files — because removing a file a manifest
                    # enumerates shifts the manifest just as adding one does
                    # (reviewer amendment, task 2838). The reify consult keeps its
                    # existing on-disk existing_files contract.
                    should_override = role == 'merge' and (
                        _merge_config_only_diff_forces_full_gate(config, task_files)
                        or await _verify_pipeline_guard_requires_full_gate(
                            worktree, existing_files,
                        )
                    )
                    if should_override:
                        logger.info(
                            'config-only fast-path overridden by manifest-drift'
                            ' backstop or verify-pipeline-guard'
                            ' — running full gate (no-module_configs merge path)',
                        )
                        # Fall through to the existing global run_verification path.
                    else:
                        logger.info(
                            'Verification mode: trivial pass (no source files, no module configs)',
                        )
                        return _trivial_pass(
                            'No source files changed — verify trivially passes',
                        )
            # NOTE (task γ amendment): unlike the module_configs branch below,
            # this call site deliberately does NOT thread a shared
            # content_cache into _build_fallback_config — doing so would add
            # a keyword argument that TestRunScopedVerificationForwardsWorktreeToFallback
            # .test_worktree_forwarded_to_build_fallback_config (test_verify.py,
            # outside this task's locked modules) cannot accept: it replaces
            # _build_fallback_config with a fixed `(task_files, config=None,
            # worktree=None)` fake with no **kwargs catch-all, so any new
            # call-site keyword breaks it. _build_fallback_config still
            # accepts content_cache for direct/future callers; only this
            # branch's dedup between it and derive_verify_plan's reader below
            # is left unwired pending a follow-up that updates the test double.
            fallback = _build_fallback_config(existing_files, config, worktree=worktree)
            if fallback is not None:
                fallback = _apply_cargo_scope(
                    fallback, existing_files, worktree, scope_cargo_enabled,
                )
                logger.info('Verification mode: fallback-scoped (%d files)', len(existing_files))
                # Plan-authoritative execution (task κ,
                # verify-scope-inversion-prd.md): derive_verify_plan's
                # fallback branch supplies the D1/D2 scope_kind DECISION (why
                # a tool slot is FULL_SUITE/FILE_SCOPED/SKIPPED);
                # _build_fallback_config (above) still owns the
                # filesystem-dependent RENDERING (subproject cd-scoping,
                # mixed root+subproject scoping, TYPE/LINT uv-context
                # rescoping, OPAQUE fleet-chain first-clause scoping — see
                # its own docstring) exactly as before this task. Passing
                # `fallback` (the already-executed, already-cargo-scoped
                # ModuleConfig) as `executed_fallback` below folds that
                # rendering back onto the plan via _executed_fallback_plan,
                # so VerifyResult.plan records the EXECUTED command
                # (module_prefix + cmd) rather than an idealized flat
                # '__fallback__' guess that ignores subproject rescoping. Its
                # structural-content read is still NOT deduped against
                # _build_fallback_config's own read above (see the NOTE at
                # the _build_fallback_config call site) — unchanged from
                # before this task.
                plan_dict = _safe_derive_verify_plan_dict(
                    existing_files, module_configs, config, _worktree_reader(worktree), role=role,
                    executed_fallback=fallback,
                )
                if plan_dict is not None:
                    logger.info('Verify plan: %s', plan_dict)
                fallback_result = await run_verification(
                    worktree, config, fallback, max_retries=max_retries,
                    is_merge_verify=is_merge_verify,
                    attempt_id=attempt_id, task_id=task_id, archive_root=archive_root,
                    role=role,
                    # Task 3338 / esc-3062-2: the fallback runs the fleet-wide
                    # `&&` chain, where the shell's short-circuit means an
                    # unrelated earlier subproject's red skips the clause a
                    # task's OWN assigned files live in — and one rc cannot say
                    # so. This is the ONLY call site that opts in.
                    #
                    # ...and NOT for role='merge' (amendment). Removing the
                    # short-circuit is a cost/benefit trade that inverts on the
                    # merge lane. Benefit: the per-segment diagnostic exists so
                    # a task AGENT can read its own result instead of proving
                    # an unrelated red unrelated — but a merge failure goes
                    # straight to a human, who has the whole chain anyway. Cost:
                    # a merge verify whose first subproject goes red would now
                    # run the remaining seven suites, up to the full resolved
                    # budget, with the queue blocked behind it — on the one path
                    # this module already treats as latency-critical (see the
                    # `-n`-cap comment in _run_or_skip_timed: 'merge' is never
                    # -n-capped for exactly this reason). Budget exhaustion is
                    # strictly MORE likely once every segment always runs; the
                    # yaml's measured table already has five of seven segments
                    # costing 1838.60s.
                    segment_chained_test=role != 'merge',
                )
                fallback_result.plan = plan_dict
                return fallback_result

            # For Rust projects with no module_configs and no Python fallback
            # (Reify's layout), try to scope the global commands.
            if existing_files and scope_cargo_enabled:
                synthetic = ModuleConfig(
                    prefix='__cargo_scoped__',
                    test_command=config.test_command,
                    lint_command=config.lint_command,
                    type_check_command=config.type_check_command,
                )
                rewritten = _apply_cargo_scope(
                    synthetic, existing_files, worktree, scope_cargo_enabled,
                )
                if rewritten is not synthetic:
                    logger.info(
                        'Verification mode: cargo-scoped (%d .rs files)',
                        len(existing_files),
                    )
                    return await run_verification(
                        worktree, config, rewritten, max_retries=max_retries,
                        is_merge_verify=is_merge_verify,
                        attempt_id=attempt_id, task_id=task_id, archive_root=archive_root,
                        role=role,
                    )

        # INV-1 (task 2883) GLOBAL-tail backstop: the merge gate must never
        # reach the final global run_verification with an empty command set —
        # a config whose test/lint/type commands are all None/'' would make
        # _summarize_checks return a vacuous 0==0==0 PASS, vouching for a tree
        # no gate ever ran on. This path is reached only when Site 2 was NOT
        # entered (task_files empty/None-derived-to-empty) — Site 2 already
        # returns inline for the no-source case — so there is no double emission.
        if role == 'merge' and is_merge_verify and not (
            config.test_command or config.lint_command or config.type_check_command
        ):
            _emit_trivial_pass_escalated(
                event_store, task_id,
                reason='empty_command_set', resolution='loud_fail',
            )
            logger.warning(
                'INV-1: merge gate reached the global tail with no command to run'
                ' — failing loud (merge_no_evidence) rather than a vacuous pass',
            )
            return _merge_no_evidence_fail('empty_command_set')

        logger.info('Verification mode: global (no scope info)')
        return await run_verification(
            worktree, config, max_retries=max_retries,
            is_merge_verify=is_merge_verify,
            attempt_id=attempt_id, task_id=task_id, archive_root=archive_root,
            role=role,
        )
    finally:
        _maybe_prune_archive(archive_root)


async def verify_failure_is_preexisting_on_main(
    worktree: Path,
    config: 'OrchestratorConfig',
    module_configs: 'list[ModuleConfig]',
    task_files: 'list[str] | None',
    failing_result: VerifyResult,
    git_ops: object,
) -> tuple[bool, str]:
    """Detect whether *failing_result* is inherited from the current main HEAD.

    LEASE-SAFETY & HOST-AFFINITY (task 2565): this is a leaseless,
    local-only detective classifier.  It NEVER acquires or holds a merge
    :class:`~orchestrator.verify_runner.HostLease` or host slot — it
    verifies via a local ``_mainprobe-`` ephemeral worktree (see below)
    and ``run_scoped_verification`` directly, never routing through
    :class:`~orchestrator.verify_runner.HostAllocator` or
    :class:`~orchestrator.verify_runner.RemoteRunner`.  It also NEVER runs
    remote-affine (probing on the failing verify's origin host, e.g. a
    laptop lease) even when the triggering post-merge verify ran remote —
    it always runs on the dispatching/trust-anchor host.  This is
    deliberate, not incidental: the verdict drives a queue-halting
    born-at-L2 escalation and a "fix main" auto-heal, a
    correctness-critical GLOBAL decision that must answer "is main
    actually broken on the trust anchor?" — reproducing a HOST-SPECIFIC
    toolchain/env/flock/disk quirk from a remote host onto that same
    host's main would falsely conclude main is red (a false queue halt
    plus a false fix-main task), and a remote-affine probe would
    re-introduce the remote-lease / orphaned-remote-build hazards task
    1757's ``_abort_remote_verify`` teardown exists to avoid.  The
    (category, normalised cause_hint) signature comparison below is
    environment-robust for genuine main breaks; a host-specific failure
    that doesn't reproduce locally correctly falls through to
    ``(False, '')`` (task-fault, fail-safe).  Callers thread an
    ``origin_is_local`` signal (derived from the triggering verify's
    ``runner``) through ``_run_deferred_main_health_probe`` down to the
    ``origin_host``/``probe_host`` pair recorded on the
    ``main_health_red`` merge-attempt telemetry purely for OBSERVABILITY
    of the placement decision — it never changes where or how this
    function itself probes.

    Returns:
        ``(True, main_sha)`` iff the same (category, normalised cause_hint) signature
        reproduces on main — the break is preexisting.  The caller can reuse
        *main_sha* for fingerprint composition without a second ``get_main_sha`` call.

        ``(False, '')`` (fail-safe) for any of:
          - Main probe passes (break is task-own).
          - Different signature (break is task-own or different inherited break).
          - ``git_ops.get_main_sha()`` fails or returns empty.
          - ``git worktree add --detach`` fails after retries.
          - Any unexpected exception during probing.
          - CONFIRM GATE (task 3597): the signature matched, but
            ``_main_probe_failure_is_isolated_flake`` positively confirmed
            every named failing test passes in isolation on the main
            probe — a CPU-starvation load flake, not a real break.  See
            that function's docstring for the full contract (what it
            re-runs, the precondition that guards against a co-occurring
            genuine lint/type break, the one-way-ratchet guarantee, and
            why the downgrade is deliberately not cached).

    CONFIRM GATE (task 3597): a signature match alone is not trusted
    blindly — see ``_main_probe_failure_is_isolated_flake``'s docstring for
    the full contract.  This gate is scoped to the signature-comparison
    path only — the task-μ baseline-diff fork above
    (``failing_result.failing_test_ids is not None``) returns before this
    point and is untouched.

    A process-wide TTL cache (keyed by (main_sha, category, normalised cause_hint))
    avoids redundant probes from the same or sibling tasks against an unchanged main.

    The task worktree is NEVER mutated — all git ops target *config.project_root*
    (for worktree-level commands) and the detached probe path (for probe verify).
    Cleanup (worktree remove --force + shutil.rmtree of the probe dir) always
    runs in a ``finally`` block.  No broad ``git worktree prune`` is issued so
    concurrently-active sibling probes are not disturbed.

    The probe worktree is created under *git_ops.worktree_base* with a
    ``_mainprobe-<id>`` prefix (mirroring ``_create_merge_worktree``'s
    ``_merge-<id>`` scheme).  Placement under worktree_base ensures environment
    parity: upward directory traversal resolves node_modules / repo-root shared
    installs exactly as task worktrees do.  The ``_mainprobe-`` prefix is
    distinct from ``_merge-`` so the disk-pressure prune
    (``prune_stale_merge_worktrees``, targeting ``_merge-*`` only) never
    reclaims the probe mid-run.
    """
    from orchestrator.git_ops import EphemeralWorktreeError, WorktreeKind

    # Lazy-import normalisation helper from workflow to avoid the
    # verify<->workflow import cycle (workflow imports verify at module level).
    # Using the same normaliser as the loop-guard ensures the comparison is
    # apples-to-apples.
    def _normalize(hint: str | None) -> str:
        try:
            from orchestrator.workflow import _normalize_cause_hint
            return _normalize_cause_hint(hint)
        except Exception:
            return (hint or '').strip().lower()

    try:
        # Resolve the current main SHA.
        try:
            main_sha: str = await git_ops.get_main_sha()  # type: ignore[union-attr]
        except Exception:
            logger.debug('verify_failure_is_preexisting_on_main: get_main_sha failed', exc_info=True)
            return False, ''
        if not main_sha:
            return False, ''

        # Task μ (verify-scope-inversion-prd.md): when the failing result
        # carries junit-derived failing_test_ids (merge+full breadth — see
        # run_verification), decide via a per-main-SHA failing-test-id
        # baseline diff instead of the (category, cause_hint) signature
        # comparison below (B1). A baseline of None (B3 degrade — OPAQUE/
        # non-pytest command, or the baseline probe itself failed) falls
        # through to the existing signature-comparison path unchanged, and
        # failing_test_ids=None (today's callers, e.g. task-verify at
        # workflow.py) always takes that legacy path too.
        #
        # Cost note (reviewer_comprehensive finding 2, task 2590): on a cold
        # cache, main_baseline_failing_ids below pays for a FULL-SUITE
        # merge-role probe (task_files=None) rather than the cheaper scoped
        # role='task' probe further down this function — this applies to
        # every caller that reaches here with a non-None failing_test_ids,
        # sync (train/merge_gates/solo-reverify, via _classify_main_health_red)
        # and deferred alike. This is confirmed acceptable, not an oversight:
        # (1) it is opt-in — failing_test_ids is only ever non-None under
        # merge_verify_breadth='full' (default remains 'scoped', so every
        # caller that hasn't opted in pays exactly zero extra cost, byte-
        # identical to pre-μ behaviour); (2) it is required for correctness
        # — a full-suite branch id-set is only meaningfully diffable against
        # an equally full-suite baseline id-set, a scoped signature
        # comparison would not be apples-to-apples here; and (3) steady-state
        # cost is amortized to a cache read by the pass-path seeding (B2,
        # see seed_main_baseline) — a cold probe only happens on the first
        # gate run against a given main tip, or after a TTL expiry / restart.
        if failing_result.failing_test_ids is not None:
            baseline = await main_baseline_failing_ids(
                config, module_configs, git_ops, main_sha,
            )
            if baseline is not None:
                branch_ids = frozenset(failing_result.failing_test_ids)
                wholly = is_wholly_preexisting(branch_ids, baseline)
                return wholly, (main_sha if wholly else '')
            # baseline is None (B3) — fall through to the legacy probe below.

        # Check the process-wide probe cache before paying the worktree-add cost.
        _norm_hint = _normalize(failing_result.cause_hint)
        _cache_key = (main_sha, failing_result.category or '', _norm_hint)
        _now = time.monotonic()
        if _cache_key in _PROBE_CACHE:
            _cached_at, _cached = _PROBE_CACHE[_cache_key]
            if _now - _cached_at < _PROBE_CACHE_TTL:
                logger.debug(
                    'verify_failure_is_preexisting_on_main: cache hit '
                    '(main_sha=%.8s, preexisting=%s)', main_sha, _cached,
                )
                return _cached, (main_sha if _cached else '')

        # Probe worktree lifecycle (mint under worktree_base with the
        # '_mainprobe-' band, retry-on-lock-contention add, GUARANTEED scoped
        # cleanup on exit — never a broad 'git worktree prune', DD5) is owned
        # by GitOps.ephemeral_worktree; see its docstring for the full
        # contract.
        #
        # warm_seed=True (task 2567): CoW-seeds the probe's target/ from the
        # shared warm-lane base (reusing GitOps._seed_warm_lane) so the probe
        # starts from a pre-built main and verifies in the warm timeout tier
        # instead of a cold ~30-45min recompile. Fails soft to cold when no
        # warm base exists — see ephemeral_worktree's warm_seed docstring.
        async with git_ops.ephemeral_worktree(  # type: ignore[union-attr]
            WorktreeKind.MAIN_PROBE, main_sha, warm_seed=True,
        ) as tmp_path:
            # Probe main with the same scoped commands, no retries.
            try:
                main_result = await run_scoped_verification(
                    tmp_path, config, module_configs,
                    task_files=task_files,
                    max_retries=0,
                    role='task',
                )
            except Exception:
                logger.debug(
                    'verify_failure_is_preexisting_on_main: probe verify raised', exc_info=True,
                )
                return False, ''

            if main_result.passed:
                # Main is clean — the break is task-own.
                _PROBE_CACHE[_cache_key] = (time.monotonic(), False)
                return False, ''

            # Compare (category, normalised cause_hint).
            branch_sig = (failing_result.category or '', _norm_hint)
            main_sig = (main_result.category or '', _normalize(main_result.cause_hint))
            is_preexisting = branch_sig == main_sig

            if is_preexisting:
                # CONFIRM GATE (task 3597): a signature match alone is not
                # trusted blindly — see _main_probe_failure_is_isolated_flake's
                # docstring for the full contract.
                flake_ids = await _main_probe_failure_is_isolated_flake(
                    tmp_path, config, module_configs, main_result,
                )
                if flake_ids is not None:
                    # Repeat-count observability (reviewer_comprehensive
                    # finding 4, task 3597 amendment): the downgrade below is
                    # deliberately not cached (see comment further down), so
                    # a sustained load-flake storm re-pays the full probe +
                    # isolated-rerun cost on every sibling task that hits
                    # this exact signature. That no-cache decision is
                    # unchanged — caching would reintroduce the false-
                    # negative risk it exists to avoid — but the repetition
                    # itself is now named in the WARNING and recorded on the
                    # audit entry instead of recurring silently.
                    _repeat_count = _MAIN_PROBE_DOWNGRADE_REPEAT_COUNTS.get(_cache_key, 0) + 1
                    _MAIN_PROBE_DOWNGRADE_REPEAT_COUNTS[_cache_key] = _repeat_count
                    logger.warning(
                        'verify_failure_is_preexisting_on_main: %s passed on '
                        'isolated re-run on the main probe (sha=%.8s) — '
                        'downgrading from preexisting-main-break to '
                        'task-own (main is not red for these tests)%s',
                        flake_ids, main_sha,
                        (
                            f' [repeat #{_repeat_count} for this exact '
                            f'signature — the verdict is deliberately not '
                            f'cached, so a persistent load-flake storm '
                            f're-probes and re-downgrades every time]'
                        ) if _repeat_count > 1 else '',
                    )
                    _suppressed_flake_records.append({
                        'sha': main_sha,
                        'node_ids': flake_ids,
                        'first_pass_category': main_result.category,
                        'first_pass_cause_hint': main_result.cause_hint,
                        'suppressed_via': 'main_probe_isolated_rerun',
                        'repeat_count': _repeat_count,
                    })
                    # Deliberately NOT cached: a load-flake verdict is a
                    # statement about transient host state, not the main
                    # tip, so caching it would mis-attribute a LATER genuine
                    # red main (same signature) to every sibling task that
                    # hits this cache key within the TTL.
                    return False, ''

            _PROBE_CACHE[_cache_key] = (time.monotonic(), is_preexisting)
            return is_preexisting, (main_sha if is_preexisting else '')

    except EphemeralWorktreeError as e:
        logger.warning(
            'verify_failure_is_preexisting_on_main: %s — contagion guard '
            'disabled for this attempt', e,
        )
        return False, ''
    except Exception:
        logger.debug('verify_failure_is_preexisting_on_main: unexpected error', exc_info=True)
        return False, ''


async def run_main_tip_sweep(
    config: 'OrchestratorConfig',
    git_ops: object,
    *,
    main_sha: str | None = None,
) -> 'tuple[str, VerifyResult] | None':
    """Run a full unscoped verification sweep against the current main-tip SHA.

    Creates a throwaway detached worktree pinned at *main_sha* under
    ``git_ops.worktree_base`` (``_mainsweep-<hex>`` prefix — distinct from
    ``_merge-``/``_mainprobe-`` so the disk-pressure prune never reclaims it
    mid-run).  Runs ``run_full_verification`` (all subprojects: test + lint +
    typecheck in parallel) and returns ``(main_sha, result)``.

    Args:
        config: Orchestrator configuration.
        git_ops: GitOps instance (needs ``get_main_sha``, ``worktree_base``).
        main_sha: Optional pre-resolved SHA.  When provided the internal
            ``git_ops.get_main_sha()`` call is skipped, eliminating a redundant
            subprocess and closing the TOCTOU window between the harness
            SHA-dedup gate and the worktree pin.  Callers that already resolved
            the SHA (e.g. ``_run_main_tip_sweep`` in harness.py) should pass it.

    Returns:
        ``(main_sha, VerifyResult)`` on success (result.passed may be False).
        ``None`` (fail-safe) when:
          - ``get_main_sha()`` raises or returns an empty string.
          - ``git worktree add --detach`` fails after retries.
          - Any unexpected exception during sweep setup.
          - ``run_full_verification`` returns ``category`` in
            ``{'pytest_internalerror', 'env_transient'}`` on either the first
            pass or the retry (infra crash / shared-venv-mutation transient,
            not drift).
        The harness treats ``None`` as "no signal — retry next tick" and does
        NOT mark the SHA as swept, so the same tip is retried on the next interval.

    Isolated PRE-FILTER (task 3095, gated by
    ``config.main_tip_sweep_isolated_prefilter_enabled``, default on): when the
    first pass fails, ``_sweep_failure_reproduces_in_isolation`` re-runs JUST
    the named failing node-ids — scoped, forced-serial, generous-timeout — in
    this same pinned worktree BEFORE the full retry is paid for.  Only a
    deterministic reproduction (``True``) short-circuits: the full retry is
    skipped and the FIRST-PASS failing result is returned.  ``False``
    (did-not-reproduce) and ``None`` (unconfirmable) both fall through to the
    unchanged full retry below.  Setting the flag False restores
    byte-identical pre-3095 behavior.

    This is a COST gate, never a verdict.  The pre-filter can shorten the path
    to a failing return but can never produce a passing one, so the sweep
    NEVER returns a ``passed=True`` result derived from subset-only evidence —
    which is what keeps the harness's self-heal precondition intact
    (``_close_superseded_main_sweep_escalations`` fires on a passing sweep
    result, and that must mean a genuine full-verify PASS; an isolated subset
    re-run is weaker evidence than that).  Both error directions are safe:
    a residue-induced false ``True`` only hands a FAILING result to the
    harness, which still re-runs the node-ids in a FRESH worktree via
    ``confirm_main_tip_failure_is_real`` — the sole suppression authority —
    before filing anything; a false ``False`` merely pays for the retry as
    before.

    Retry-on-flake: when the first ``run_full_verification`` call fails (and its
    category is NOT one of the infra sentinels above, and the pre-filter did not
    short-circuit), the function re-runs it ONCE in the same pinned worktree
    (idempotent; no second ``git worktree add``).  **The retry reuses first-pass
    worktree state by design** — no cleanup of temp files, partially-written
    DBs, or caches is performed before the re-run.  This is intentional: the
    purpose is a fast flake-vs-drift heuristic, not a hermetic isolation
    guarantee.  A first run that fails partway may leave residue that makes the
    retry non-representative in either direction; the single-retry bound and the
    two-failure-escalates rule limit the blast radius.

    - Retry PASSES → emit a WARNING, append a record to
      ``verify._suppressed_flake_records`` (durable in-process audit trail), and
      return ``(main_sha, retry_result)`` so the harness files no drift
      escalation.  NOTE: this suppresses the flake but **MAY MASK a real
      intermittent regression** introduced by a merge.  Since task 3095 that
      masking window is NARROWER but not closed: it is now reachable only when
      the pre-filter did NOT see the named tests reproduce, i.e. the failure
      already looks load-induced — and a genuine FULL green is still required
      to reach it.  A deterministic failure no longer gets a second lottery
      ticket at being masked.
    - Retry FAILS → return ``(main_sha, retry_result)`` so deterministic drift
      still escalates.
    - Retry hits pytest INTERNALERROR or env_transient → return ``None``
      (infra, retry next tick).

    Cleanup: scoped ``git worktree remove --force <tmp_path>`` + ``shutil.rmtree``
    always runs in a ``finally`` block.  NO broad ``git worktree prune`` (DD5
    guarantee: a broad prune would deregister concurrently-active sibling
    probe/merge worktrees).
    """
    from orchestrator.git_ops import (  # noqa: PLC0415 — lazy, mirrors verify_failure_is_preexisting_on_main
        EphemeralWorktreeError,
        WorktreeKind,
    )

    try:
        # Resolve the current main SHA unless the caller pre-resolved it.
        # Accepting a pre-resolved value eliminates a redundant git rev-parse
        # subprocess and closes the TOCTOU window between the harness SHA-dedup
        # gate and the worktree pin (both now use the same resolved value).
        if main_sha is None:
            try:
                main_sha = await git_ops.get_main_sha()  # type: ignore[union-attr]
            except Exception:
                logger.debug('run_main_tip_sweep: get_main_sha failed', exc_info=True)
                return None
        if not main_sha:
            return None

        # Sweep worktree lifecycle (mint under worktree_base with the
        # '_mainsweep-' band, retry-on-lock-contention add, GUARANTEED scoped
        # cleanup on exit — never a broad 'git worktree prune', DD5) is owned
        # by GitOps.ephemeral_worktree; see its docstring for the full
        # contract.
        async with git_ops.ephemeral_worktree(  # type: ignore[union-attr]
            WorktreeKind.MAIN_SWEEP, main_sha,
        ) as tmp_path:
            def _enoent_on_self(r: 'VerifyResult') -> bool:
                """SECONDARY backstop (task 2507): True iff *r* is a
                not-passed result whose cause_hint names an ENOENT ('No
                such file or directory' / '[Errno 2]') referencing THIS
                sweep's own tmp_path — i.e. the worktree vanished mid-run.

                Narrowly scoped to this sweep's own path — deliberately
                NOT a blanket 'unknown_test_failure' addition to
                INFRA_TRANSIENT_CATEGORIES, which would silently swallow
                genuine main-tip drift under that broad category. This is
                defense-in-depth behind the PRIMARY fix (
                GitOps.ephemeral_worktree's flock-liveness guard, which
                stops the warm-lane-GC race that used to cause exactly
                this ENOENT); it backstops residual worktree-vanish causes
                (disk fault, a manual ``rm``) after the flock closes the
                GC race.
                """
                if r.passed or not r.cause_hint:
                    return False
                hint_lower = r.cause_hint.lower()
                return (
                    ('no such file or directory' in hint_lower or '[errno 2]' in hint_lower)
                    and str(tmp_path) in r.cause_hint
                )

            # Run full (unscoped) verification — all discovered subprojects, no scope filter.
            # role='background' (lowest nice tier — task 2391/PRD T3): the sweep is a
            # background asyncio.Task with no dispatch/merge/deploy path awaiting it, so
            # its fan-out should never contend with real task/merge lane verifies.
            result = await run_full_verification(tmp_path, config, role='background')  # type: ignore[arg-type]

            # pytest INTERNALERROR means the test infrastructure itself crashed (e.g. an
            # xdist worker was killed by os._exit).  env_transient means a concurrent
            # `uv sync` elsewhere transiently mutated the shared venv mid-run (vanished
            # xdist/pip).  Both are infra failures, not drift — return the None sentinel
            # so the harness retries next tick and files no false-positive drift L1.
            # The CM's guaranteed cleanup still runs on the way out.
            if result.category in INFRA_TRANSIENT_CATEGORIES:
                logger.warning(
                    'run_main_tip_sweep: %s in first-pass sweep — '
                    'treating as infra crash, not drift (retrying next tick); '
                    'cause_hint=%r',
                    'pytest INTERNALERROR' if result.category == 'pytest_internalerror'
                    else 'environmental shared-venv transient (env_transient)',
                    result.cause_hint,
                )
                return None

            if _enoent_on_self(result):
                logger.warning(
                    'run_main_tip_sweep: sweep worktree %s vanished mid-run '
                    '(cause_hint=%r) — treating as infra transient, not '
                    'drift (retrying next tick)',
                    tmp_path, result.cause_hint,
                )
                return None

            if not result.passed:
                # First pass failed (not an INTERNALERROR).  Re-run once in the same
                # pinned worktree to distinguish a transient load-sensitive flake from
                # deterministic drift.  A second worktree add is NOT needed — the
                # worktree is already pinned at main_sha, so re-running is idempotent.
                # NOTE: worktree state (temp files, partially-written DBs, caches) from
                # the first run is NOT reset before the retry — this is intentional (fast
                # heuristic, not hermetic isolation; see docstring for tradeoff discussion).
                _sha_prefix = main_sha[:12] if main_sha else '?'
                logger.warning(
                    'run_main_tip_sweep: first-pass verification failed at %s '
                    '(category=%r, cause_hint=%r) — retrying once in the same '
                    'worktree to distinguish transient flake from deterministic drift',
                    _sha_prefix, result.category, result.cause_hint,
                )

                # COST pre-filter (task 3095): before paying for a whole
                # second full-suite run, re-run just the named failing
                # node-ids in isolation.  A deterministic reproduction means
                # the full retry is near-certainly wasted work AND that the
                # sweep would otherwise keep adding minutes of background load
                # during a red-main investigation.  Only True short-circuits;
                # False/None both fall through to the unchanged full retry, so
                # a passing sweep result still requires a genuine FULL green.
                #
                # Defense-in-depth: the helper already wraps its own body, so
                # in practice it cannot raise — but a raise escaping HERE would
                # hit run_main_tip_sweep's outer `except Exception` and
                # silently collapse a REAL red-main signal into the None
                # "no signal" sentinel, dropping the drift on the floor. That
                # failure mode is severe and silent enough to be worth pinning
                # against a future edit to the helper, so the call is caught
                # locally and degraded to "did not short-circuit".
                _reproduced: bool | None = None
                if config.main_tip_sweep_isolated_prefilter_enabled:
                    try:
                        _reproduced = await _sweep_failure_reproduces_in_isolation(
                            tmp_path, config, result,
                        )
                    except Exception:
                        logger.warning(
                            'run_main_tip_sweep: isolated pre-filter raised at '
                            '%s — falling through to the full-suite retry',
                            _sha_prefix, exc_info=True,
                        )
                if _reproduced is True:
                    logger.warning(
                        'run_main_tip_sweep: first-pass failure at %s '
                        'reproduced deterministically in isolation '
                        '(category=%r, cause_hint=%r) — skipping the '
                        'full-suite retry and returning the first-pass '
                        'failure; the harness confirm gate still adjudicates '
                        'before any escalation is filed',
                        _sha_prefix, result.category, result.cause_hint,
                    )
                    return (main_sha, result)

                retry = await run_full_verification(tmp_path, config, role='background')  # type: ignore[arg-type]

                if retry.category in INFRA_TRANSIENT_CATEGORIES:
                    logger.warning(
                        'run_main_tip_sweep: retry at %s hit %s — '
                        'treating as infra crash, not drift (retrying next tick); '
                        'cause_hint=%r',
                        _sha_prefix,
                        'pytest INTERNALERROR' if retry.category == 'pytest_internalerror'
                        else 'an environmental shared-venv transient (env_transient)',
                        retry.cause_hint,
                    )
                    return None

                if _enoent_on_self(retry):
                    logger.warning(
                        'run_main_tip_sweep: sweep worktree %s vanished '
                        'mid-run on retry (cause_hint=%r) — treating as '
                        'infra transient, not drift (retrying next tick)',
                        tmp_path, retry.cause_hint,
                    )
                    return None

                if retry.passed:
                    logger.warning(
                        'run_main_tip_sweep: first-pass failure at %s did NOT '
                        'reproduce on retry (first-pass category=%r, '
                        'cause_hint=%r) — treating as transient flake and '
                        'suppressing drift escalation. '
                        'NOTE: retry-on-flake MAY MASK a real intermittent '
                        'regression introduced by a recent merge.',
                        _sha_prefix, result.category, result.cause_hint,
                    )
                    # Append to the in-process audit registry so suppressed flakes
                    # remain observable beyond the log stream (tests can inspect
                    # verify._suppressed_flake_records; operators can too via the
                    # live object graph).
                    _suppressed_flake_records.append({
                        'sha': main_sha,
                        'first_pass_category': result.category,
                        'first_pass_cause_hint': result.cause_hint,
                    })

                # Return the retry result: passing (flake suppressed) or failing
                # (deterministic drift — harness files L1 escalation as usual).
                return (main_sha, retry)

            return (main_sha, result)

    except EphemeralWorktreeError as e:
        logger.warning(
            'run_main_tip_sweep: %s — sweep skipped for this tick', e,
        )
        return None
    except Exception:
        logger.debug('run_main_tip_sweep: unexpected error', exc_info=True)
        return None


# Bound on isolated-rerun attempts per confirm-gate subproject group (task
# 2370). A PASS on ANY attempt within this bound is treated as a confirmed
# flake for that group — mirrors run_main_tip_sweep's single-retry heuristic,
# widened slightly since this re-run is already scoped to just the named
# tests (cheap) and serial/addopts-cleared (task 2045's proven xdist-
# contention recovery). No config flag: the gate is a strict fail-safe
# improvement over the status quo (a bare, unconfirmed alarm), so it is
# always-on.
_SWEEP_CONFIRM_MAX_ATTEMPTS = 2

#: Generous per-test timeout (seconds) injected into the main-tip-sweep
#: CONFIRM gate's isolated re-run command via ``_with_pytest_timeout_str``.
#: Same rationale as ``_MERGE_FLAKE_CONFIRM_TIMEOUT_SECS`` /
#: ``_SWEEP_PREFILTER_TIMEOUT_SECS``: the serial recovery's ``-o addopts=``
#: clears pyproject ``addopts`` but NOT the
#: ``[tool.pytest.ini_options] timeout=60`` default, so without this
#: explicit override a still-loaded host can starve the isolated confirm
#: run into a false "still fails" verdict — and unlike the merge gate
#: (which only holds a merge), a false verdict HERE files a red-main L1.
#: Kept as a SEPARATE constant from the pre-filter's and the merge gate's
#: so the three are retuned on their own signals.
_SWEEP_CONFIRM_TIMEOUT_SECS: int = 300


class _RerunOutcome(StrEnum):
    """What a bounded isolated re-run actually observed — verify's INTERNAL
    engine vocabulary, deliberately distinct from the wire-facing
    :class:`orchestrator.flake_ledger.FlakeVerdict`.

    Exists because ``bool`` cannot say the third thing. ``_run_isolated_confirm_group``'s
    own log line has always said ``'unconfirmable, not counted as a pass'`` for an
    infra-sentinel attempt, while its return type collapsed that into the same
    ``False`` a genuine red produces — the fact lived in a variable and died in a
    log (INV-2, structured-facts-at-failure). ``confirm_isolated_rerun_verdict``
    needs the distinction to map an infra-sentinel re-run to
    ``FlakeVerdict.unconfirmable`` rather than mislabelling it as a real red.

    PRECEDENCE, when attempts disagree: ``passed`` (a pass anywhere is proof the
    group can run green) > ``failed`` (a genuine red observed anywhere is real
    evidence) > ``unconfirmable`` (we never actually got a verdict). So
    ``unconfirmable`` means EVERY attempt was uninformative, never "some were".
    """

    passed = 'passed'  # some attempt ran and PASSED — a confirmed flake for this group
    failed = 'failed'  # some attempt ran and genuinely FAILED, none passed
    unconfirmable = 'unconfirmable'  # no attempt produced a usable verdict at all


async def _run_isolated_confirm_group_outcome(
    worktree: Path,
    config: 'OrchestratorConfig',
    module_config: ModuleConfig,
) -> _RerunOutcome:
    """Run *module_config* (already scoped + forced-serial) up to
    ``_SWEEP_CONFIRM_MAX_ATTEMPTS`` times against *worktree*, preserving WHICH
    of the three outcomes was observed.

    The outcome-preserving source of truth behind ``_run_isolated_confirm_group``
    (which is a ``outcome is passed`` bool shim over this).

    Returns ``passed`` as soon as any attempt PASSES (that group is a confirmed
    flake). Otherwise returns ``failed`` if any attempt produced a genuine red —
    a real failure, or a timeout (``VerifyResult.timed_out`` with
    ``passed=False``) — and ``unconfirmable`` only when EVERY attempt was
    uninformative: an infra-sentinel category
    (``pytest_internalerror``/``env_transient`` — never trusted as confirmation
    either way) or a raised exception (caught here so a transient error on one
    attempt doesn't abort the remaining attempts). Never raises.
    """
    saw_genuine_failure = False
    for attempt in range(_SWEEP_CONFIRM_MAX_ATTEMPTS):
        try:
            result = await run_verification(worktree, config, module_config, max_retries=0)
        except Exception:
            logger.debug(
                'confirm_main_tip_failure_is_real: isolated re-run raised '
                '(attempt %d/%d) for %r',
                attempt + 1, _SWEEP_CONFIRM_MAX_ATTEMPTS, module_config.test_command,
                exc_info=True,
            )
            continue
        # An infra-sentinel category (pytest_internalerror/env_transient) is
        # never trusted as confirmation, even in the (normally impossible)
        # case it were paired with passed=True — mirrors run_main_tip_sweep's
        # own category-first check, which is deliberately independent of the
        # passed flag (see its INFRA_TRANSIENT_CATEGORIES branch).
        if result.category in INFRA_TRANSIENT_CATEGORIES:
            logger.debug(
                'confirm_main_tip_failure_is_real: isolated re-run hit %s '
                '(attempt %d/%d) for %r — unconfirmable, not counted as a pass',
                result.category, attempt + 1, _SWEEP_CONFIRM_MAX_ATTEMPTS,
                module_config.test_command,
            )
            continue
        if result.passed:
            return _RerunOutcome.passed
        # A completed, non-sentinel, non-passing attempt IS evidence of a real
        # red — the one thing an exhausted loop must not report as "we could
        # not tell".
        saw_genuine_failure = True
    return _RerunOutcome.failed if saw_genuine_failure else _RerunOutcome.unconfirmable


async def _run_isolated_confirm_group(
    worktree: Path,
    config: 'OrchestratorConfig',
    module_config: ModuleConfig,
) -> bool:
    """Run *module_config* (already scoped + forced-serial) up to
    ``_SWEEP_CONFIRM_MAX_ATTEMPTS`` times against *worktree*.

    Returns ``True`` as soon as any attempt PASSES (that group is a confirmed
    flake). Returns ``False`` if every attempt exhausts without a pass —
    covers a genuine failure, a timeout (``VerifyResult.timed_out`` with
    ``passed=False``), an infra-sentinel category
    (``pytest_internalerror``/``env_transient`` — never trusted as
    confirmation either way), and a raised exception (caught here so a
    transient error on one attempt doesn't abort the remaining attempts).
    Never raises.

    A lossy shim over ``_run_isolated_confirm_group_outcome``, which is the
    outcome-preserving source of truth: both ``failed`` and ``unconfirmable``
    flatten to ``False`` here, deliberately, so this function's three existing
    callers observe no behaviour change at all.
    """
    return await _run_isolated_confirm_group_outcome(
        worktree, config, module_config,
    ) is _RerunOutcome.passed


def _group_node_ids_by_subproject(
    worktree: Path,
    module_configs: dict[str, ModuleConfig],
    node_ids: list[str],
    *,
    log_label: str,
) -> dict[str, list[str]] | None:
    """Map each pytest node-id in *node_ids* to its owning subproject prefix.

    PURE (no I/O beyond ``Path.exists`` probes against the on-disk
    *worktree*), sync, never raises on ordinary input. Extracted from
    ``confirm_main_tip_failure_is_real`` so the main-tip-sweep isolated
    pre-filter reuses it rather than the tree gaining a THIRD copy of this
    block (task 3095). All THREE call sites now share this one implementation
    — ``confirm_main_tip_failure_is_real``,
    ``_sweep_failure_reproduces_in_isolation``, and the merge gate
    ``confirm_merge_verify_flake_suppressible`` (task 3290, which folded in
    the last inline copy; it passes ``{mc.prefix: mc for mc in
    module_configs}`` built from its list-shaped parameter).

    For each node-id, the file component (``node_id.split('::', 1)[0]``) is
    probed against EVERY discovered subproject — not just the first match —
    so a bare relative path that happens to exist under more than one
    subproject can be flagged rather than silently mis-attributed to
    whichever prefix iterates first:

    * ``<worktree>/<prefix>/<relpath>`` exists → subproject-relative node-id;
      the qualified id ``<prefix>/<node_id>`` is recorded.
    * ``<relpath>`` already starts with ``<prefix>/`` and
      ``<worktree>/<relpath>`` exists → already worktree-root-relative; the
      node-id is recorded verbatim.

    Returns:
        ``dict[prefix, list[qualified_node_id]]`` — node-ids owned by the same
        subproject grouped together, INPUT ORDER preserved within each group,
        so a caller can build one scoped re-run command per subproject.
        An empty *node_ids* yields ``{}`` (NOT the None sentinel — "nothing to
        map" is distinct from "unmappable"; callers early-out on empty input
        themselves).

        ``None`` when ANY node-id maps to no discovered subproject (logged at
        INFO with *log_label*). This is the callers' fail-safe signal — one
        unmappable node-id poisons the whole batch rather than the helper
        guessing which subproject it belongs to, or silently running a
        partial subset.

    An ambiguous node-id (relpath present under >1 subproject) resolves
    deterministically to the FIRST candidate by *module_configs* iteration
    order and logs a WARNING naming every candidate prefix and *log_label* —
    a low-likelihood, non-fatal ambiguity, not a fail-safe path.

    Args:
        worktree: Tree the existence probes run against.
        module_configs: Prefix -> ModuleConfig, freshly discovered on
            *worktree* by the caller (never a snapshot for a different tree).
        node_ids: Extracted failing pytest node-ids, in output order.
        log_label: Caller name, embedded in every log line so an operator can
            attribute the message to the right call site.
    """
    groups: dict[str, list[str]] = {}
    for node_id in node_ids:
        file_part = node_id.split('::', 1)[0]
        candidates: list[tuple[str, str]] = []
        for prefix, _mc in module_configs.items():
            if (worktree / prefix / file_part).exists():
                candidates.append((prefix, f'{prefix}/{node_id}'))
            elif file_part.startswith(f'{prefix}/') and (worktree / file_part).exists():
                candidates.append((prefix, node_id))
        if not candidates:
            logger.info(
                '%s: node-id %r did not map to any discovered subproject in '
                '%s — unconfirmable',
                log_label, node_id, worktree,
            )
            return None
        if len(candidates) > 1:
            logger.warning(
                '%s: node-id %r matched %d discovered subprojects (%s) in %s '
                '— using %r; a relative path shared across subprojects can '
                'mis-attribute the isolated re-run to the wrong ModuleConfig',
                log_label, node_id, len(candidates), [c[0] for c in candidates],
                worktree, candidates[0][0],
            )
        matched_prefix, matched_node_id = candidates[0]
        groups.setdefault(matched_prefix, []).append(matched_node_id)
    return groups


#: Generous per-test timeout (seconds) injected into the main-tip-sweep
#: isolated PRE-FILTER's re-run command via ``_with_pytest_timeout_str``.
#: Same rationale as ``_MERGE_FLAKE_CONFIRM_TIMEOUT_SECS``: the serial
#: recovery's ``-o addopts=`` clears pyproject ``addopts`` but NOT the
#: ``[tool.pytest.ini_options] timeout=60`` default, so without this explicit
#: override a still-loaded host could starve the isolated run into a false
#: "reproduces" verdict. Kept as a SEPARATE constant from the merge gate's so
#: sweep tuning is not coupled to merge-gate tuning (they are retuned on
#: different signals).
_SWEEP_PREFILTER_TIMEOUT_SECS: int = 300


async def _sweep_failure_reproduces_in_isolation(
    worktree: Path,
    config: 'OrchestratorConfig',
    failing_result: VerifyResult,
) -> bool | None:
    """Does *failing_result*'s named failing test set reproduce in ISOLATION?

    A COST pre-filter for ``run_main_tip_sweep``'s expensive full-suite retry
    — **never a suppression verdict** (task 3095). The sole suppression
    authority remains the harness's fresh-worktree
    ``confirm_main_tip_failure_is_real`` gate; this helper only decides
    whether paying for a second full ``run_full_verification`` is worthwhile.

    Re-runs just the originally-failing node-ids, scoped + forced-serial +
    generous-timeout, in the sweep's OWN already-pinned *worktree* (no second
    ``git worktree add``). First-pass residue in that reused tree is
    deliberately accepted: both error directions are safe by construction (see
    Returns), so hermetic isolation would buy no precision here.

    Returns:
        ``True`` — REPRODUCES: at least one named test still fails in
        isolation, so the failure is deterministic and the full retry is
        near-certainly wasted work. A spurious True (residue-induced) only
        skips the retry and hands a FAILING result to the harness, which still
        re-runs the node-ids in a FRESH worktree before filing — so this
        direction can never manufacture a false alarm.

        ``False`` — DOES NOT REPRODUCE: every named test passed in isolation,
        i.e. a suspected contention flake. The caller must run the full retry
        as before, so a genuine FULL green is still required for the harness's
        self-heal precondition.

        ``None`` — UNCONFIRMABLE: no recoverable node-id (a lint/type failure
        or an unparseable crash notice), a node-id owned by no discovered
        subproject, an infra-sentinel re-run category
        (``INFRA_TRANSIENT_CATEGORIES`` — never trusted as evidence either
        way, independent of the ``passed`` flag), or ANY raised exception. The
        caller falls through to byte-identical pre-3095 behavior.

    Never raises: the whole body is wrapped, and the handler logs at WARNING
    (not debug) — a pre-filter that keeps silently degrading to the expensive
    path must be visible in the log stream.
    """
    try:
        # Cheap early-out: nothing named means nothing to re-run — pay no
        # subprocess at all.
        node_ids = _extract_failing_test_ids(failing_result.test_output)
        if not node_ids:
            return None

        # Discover module configs on the SWEEP worktree (never config's
        # snapshot — that is for a different worktree/SHA). Lazy import
        # mirrors confirm_main_tip_failure_is_real.
        from orchestrator.config import _discover_module_configs  # noqa: PLC0415

        module_configs = _discover_module_configs(worktree)

        groups = _group_node_ids_by_subproject(
            worktree, module_configs, node_ids,
            log_label='run_main_tip_sweep prefilter',
        )
        # None = an unmapped node-id. An empty dict is unreachable here (the
        # empty-node_ids early-out above already returned), but `not groups`
        # keeps a hypothetical {} from being mistaken for "all groups clean"
        # -> False, which would assert a not-reproduced verdict on zero
        # evidence.
        if not groups:
            return None

        for prefix, group_node_ids in groups.items():
            mc = module_configs[prefix]
            scoped_cmd = _with_pytest_timeout_str(
                _serial_pytest_str(
                    _scope_to_keyword(mc.test_command, 'pytest', group_node_ids),
                ),
                _SWEEP_PREFILTER_TIMEOUT_SECS,
            )
            scoped_mc = replace(
                mc, test_command=scoped_cmd, lint_command=None, type_check_command=None,
            )
            result = await run_verification(
                worktree, config, scoped_mc, max_retries=0, role='background',
            )
            # Category-first, independent of the passed flag (mirrors
            # run_main_tip_sweep / _run_isolated_confirm_group): an infra
            # sentinel is evidence of nothing, so it maps to UNCONFIRMABLE
            # rather than to either verdict.
            if result.category in INFRA_TRANSIENT_CATEGORIES:
                logger.info(
                    'run_main_tip_sweep prefilter: isolated re-run for %s hit '
                    '%s — unconfirmable, falling through to the full retry',
                    prefix, result.category,
                )
                return None
            if not result.passed:
                logger.info(
                    'run_main_tip_sweep prefilter: %s still failed in '
                    'isolation (category=%r) — deterministic reproduction',
                    group_node_ids, result.category,
                )
                return True

        return False

    except Exception:
        logger.warning(
            'run_main_tip_sweep prefilter: unexpected error — unconfirmable, '
            'falling through to the full-suite retry',
            exc_info=True,
        )
        return None


async def confirm_main_tip_failure_is_real(
    config: 'OrchestratorConfig',
    git_ops: object,
    failing_result: VerifyResult,
    *,
    main_sha: str,
) -> bool:
    """Confirm a main-tip-sweep failure is real before the harness files an alarm.

    ``run_main_tip_sweep``'s own full-suite retry runs in the SAME contended
    worktree, so a load-induced xdist flake reliably fails twice and still
    reaches the harness as "drift" — the false-positive source behind
    esc-main-sweep-ea2bd3c95e33-2 and the 2026-07-09 park_stop/symlink-loop
    incidents. This function is the harness's confirm-before-alarm gate: it
    extracts the named failing pytest node-ids from *failing_result*, and
    re-runs JUST those tests, in ISOLATION (scoped + forced-serial + addopts
    cleared — the exact task-2045 recovery — plus an explicit generous
    ``--timeout``), in a FRESH probe worktree pinned at *main_sha* — never
    the sweep's own contended worktree.

    The ``--timeout`` (``_SWEEP_CONFIRM_TIMEOUT_SECS``, task 3290) is not
    cosmetic: ``-o addopts=`` clears pyproject's ``addopts`` but NOT its
    ``[tool.pytest.ini_options] timeout=60`` default, so without the
    override a still-loaded host could starve this confirmation into a
    false "still fails" verdict — which here means filing a red-main L1
    escalation for a flake, the exact false positive this gate exists to
    prevent.

    Returns:
        ``False`` (suppress the alarm) ONLY when every named failing test
        demonstrably PASSES on isolated re-run (within
        ``_SWEEP_CONFIRM_MAX_ATTEMPTS`` attempts per owning subproject). On
        suppress, logs a non-blocking INFO note and appends an entry to
        ``_suppressed_flake_records`` (``'suppressed_via': 'isolated_rerun'``)
        so the flake stays observable for de-flaking.

        ``True`` (file the alarm) for every other path — the hard "never
        mask a REAL red" constraint:
          - *failing_result* has no recoverable node-id (a non-test failure
            such as a lint/type error, or an unparseable worker-crash notice).
          - A node-id doesn't map to any subproject discovered in the probe
            worktree.
          - ``git worktree add --detach`` fails after retries.
          - Module discovery, the isolated re-run, or any other step raises,
            times out, or comes back with an infra-sentinel category
            (``pytest_internalerror``/``env_transient``) on every attempt.
          - Any unexpected exception during confirmation.

    The probe worktree is created under ``git_ops.worktree_base`` with a
    ``_mainsweepconfirm-<hex>`` prefix — distinct from ``_mainsweep-``/
    ``_mainprobe-``/``_merge-`` so the disk-pressure prune never reclaims it
    mid-run, and distinct from the sweep's own worktree so this confirmation
    is never contended by the same load that produced the original flake.
    Cleanup (scoped ``git worktree remove --force`` + ``shutil.rmtree``)
    always runs in a ``finally`` block; no broad ``git worktree prune`` (DD5
    guarantee — mirrors ``run_main_tip_sweep``/
    ``verify_failure_is_preexisting_on_main``).

    Node-id -> subproject mapping: module configs are freshly re-discovered
    on the probe worktree (never reused from *config* — that snapshot is for
    a different worktree/SHA). Each node-id's file component is checked for
    existence as ``<worktree>/<mc.prefix>/<relpath>`` (subproject-relative
    node-id, the common case for the sweep's aggregated per-subproject
    output) or ``<worktree>/<relpath>`` (already worktree-root-relative /
    prefix-qualified). Node-ids owned by the same subproject are grouped into
    one scoped+serial re-run. If a node-id's relative path happens to exist
    under more than one discovered subproject, the first (by module-config
    discovery order) is used and a WARNING is logged — this is a low-
    likelihood, non-fatal ambiguity, not a fail-safe-to-alarm path.
    """
    import uuid  # noqa: PLC0415, I001 — lazy, mirrors run_main_tip_sweep/verify_failure_is_preexisting_on_main
    from orchestrator.config import _discover_module_configs  # noqa: PLC0415
    from orchestrator.git_ops import _run  # noqa: PLC0415

    _sha_prefix = main_sha[:12] if main_sha else '?'

    # Cheap early-out: no recoverable node-id means nothing to confirm — pay
    # no worktree-add cost at all.
    node_ids = _extract_failing_test_ids(failing_result.test_output)
    if not node_ids:
        logger.info(
            'confirm_main_tip_failure_is_real: no recoverable node-id in '
            'failure output at %s (category=%r) — unconfirmable, filing alarm',
            _sha_prefix, failing_result.category,
        )
        return True

    tmp_path: Path | None = None
    worktree_added: bool = False
    try:
        base: Path = git_ops.worktree_base  # type: ignore[union-attr]
        base.mkdir(parents=True, exist_ok=True)
        tmp_path = base / f'_mainsweepconfirm-{uuid.uuid4().hex[:8]}'

        # Retry worktree add on transient git lock contention, mirroring
        # run_main_tip_sweep / verify_failure_is_preexisting_on_main.
        _MAX_ADD_RETRIES = 3
        rc, _, err = 1, '', 'not attempted'
        for _attempt in range(_MAX_ADD_RETRIES):
            rc, _, err = await _run(
                ['git', 'worktree', 'add', '--detach', str(tmp_path), main_sha],
                cwd=config.project_root,  # type: ignore[union-attr]
            )
            if rc == 0:
                worktree_added = True
                break
            if _attempt < _MAX_ADD_RETRIES - 1:
                await asyncio.sleep(0.5 * (_attempt + 1))
        if not worktree_added:
            logger.warning(
                'confirm_main_tip_failure_is_real: worktree add failed after '
                '%d retries (rc=%d): %s — cannot confirm, filing alarm',
                _MAX_ADD_RETRIES, rc, err,
            )
            return True

        try:
            module_configs = _discover_module_configs(tmp_path)
        except Exception:
            logger.debug(
                'confirm_main_tip_failure_is_real: module discovery raised',
                exc_info=True,
            )
            return True

        # Map each node-id to its owning subproject via the shared helper
        # (see its docstring for the probe rules). Any unmapped node-id
        # returns None and fails safe to alarm — no guessing which subproject
        # a node-id belongs to.
        groups = _group_node_ids_by_subproject(
            tmp_path, module_configs, node_ids,
            log_label='confirm_main_tip_failure_is_real',
        )
        if groups is None:
            logger.info(
                'confirm_main_tip_failure_is_real: an extracted node-id did '
                'not map to a discovered subproject at %s — unconfirmable, '
                'filing alarm',
                _sha_prefix,
            )
            return True

        # Each subproject group gets its own scoped + forced-serial isolated
        # re-run. ALL groups must confirm green to suppress.
        for prefix, group_node_ids in groups.items():
            mc = module_configs[prefix]
            scoped_cmd = _with_pytest_timeout_str(
                _serial_pytest_str(
                    _scope_to_keyword(mc.test_command, 'pytest', group_node_ids),
                ),
                _SWEEP_CONFIRM_TIMEOUT_SECS,
            )
            scoped_mc = replace(
                mc, test_command=scoped_cmd, lint_command=None, type_check_command=None,
            )
            if not await _run_isolated_confirm_group(tmp_path, config, scoped_mc):
                return True

        logger.info(
            'confirm_main_tip_failure_is_real: sweep flake suppressed: %s '
            'failed under load, passed on isolated re-run at %s',
            node_ids, _sha_prefix,
        )
        _suppressed_flake_records.append({
            'sha': main_sha,
            'node_ids': node_ids,
            'first_pass_category': failing_result.category,
            'first_pass_cause_hint': failing_result.cause_hint,
            'suppressed_via': 'isolated_rerun',
        })
        return False

    except Exception:
        logger.debug('confirm_main_tip_failure_is_real: unexpected error', exc_info=True)
        return True
    finally:
        # Scoped cleanup: remove only this specific probe worktree.
        # INTENTIONALLY NO 'git worktree prune' (DD5 guarantee).
        if worktree_added and tmp_path is not None:
            try:
                await _run(
                    ['git', 'worktree', 'remove', '--force', str(tmp_path)],
                    cwd=config.project_root,  # type: ignore[union-attr]
                )
            except Exception:
                logger.debug(
                    'confirm_main_tip_failure_is_real: worktree remove failed',
                    exc_info=True,
                )
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                shutil.rmtree(tmp_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Merge-gate single flake-retry (PRD task α, cpu-load-robust-verify-prd.md)
# ---------------------------------------------------------------------------

#: Generous per-test timeout (seconds) injected into the α confirm gate's
#: isolated re-run command via ``_with_pytest_timeout_str``. Must comfortably
#: exceed any legitimate single-test wall time: the serial recovery's
#: ``-o addopts=`` clears pyproject ``addopts`` but NOT the
#: ``[tool.pytest.ini_options] timeout=60`` default, so without this explicit
#: override the isolated confirm re-run could itself starve under residual load
#: into a false non-suppression. A tunable (PRD §9).
_MERGE_FLAKE_CONFIRM_TIMEOUT_SECS = 300


async def confirm_merge_verify_flake_suppressible(
    config: 'OrchestratorConfig',
    failing_result: VerifyResult,
    *,
    worktree: Path,
    module_configs: list[ModuleConfig],
) -> list[str] | None:
    """PURE gate: is *failing_result* a suppressible CPU-starvation flake?

    The merge-path analogue of ``confirm_main_tip_failure_is_real``, with three
    deliberate differences (PRD task α):

    * SAME-TREE (INV-3): re-runs the named failing tests in the GIVEN merge
      *worktree* at the merge SHA — the exact tree being gated — rather than
      minting a fresh probe worktree for a different SHA. No ``git worktree
      add``/``remove``, no cleanup ``finally``.
    * Returns a VERDICT (``list[str]`` of confirmed-flake node-ids, or
      ``None``) and NEVER raises: the merge path (merge_queue.py) has no
      ``VerifyInfraError`` handler, so an uncaught raise there stalls the merge
      queue. The whole body is defensively wrapped — any unexpected exception
      fails CLOSED to ``None`` (merge stays red).
    * Single-shot per node-id group (PRD §5.1), not the sweep's 2-attempt loop.

    Returns the extracted node-id list ONLY when every named failing test
    demonstrably PASSES on a scoped + forced-serial + generous-timeout isolated
    re-run. Returns ``None`` (fail-closed to red — never mask a REAL red) for:
    no recoverable node-id (opaque/lint/type failure), any node-id that maps to
    no given subproject, a re-run that still fails / errors / times out, or an
    infra-sentinel re-run category (``INFRA_TRANSIENT_CATEGORIES`` — never
    trusted as confirmation).

    Node-id -> subproject mapping DELEGATES to ``_group_node_ids_by_subproject``
    over ``{mc.prefix: mc for mc in module_configs}`` — the same shared helper
    ``confirm_main_tip_failure_is_real`` and the sweep pre-filter use (task
    3290 retired this call site's inline copy). The list -> dict conversion is
    order-preserving on a prefix-deduped list, so candidate iteration and
    first-wins ambiguity resolution are unchanged. The mapping runs over the
    GIVEN *module_configs* + *worktree* (the merge tree already on disk), never
    re-discovered. Empty *module_configs* / files-not-on-disk (unit-test fakes)
    naturally map nothing -> ``None``, which keeps existing
    ``LocalRunner.run_merge_verify`` tests byte-identical.
    """
    try:
        node_ids = _extract_failing_test_ids(failing_result.test_output)
        if not node_ids:
            return None

        # Map each node-id to its owning subproject over the given
        # module_configs + the on-disk merge worktree, via the SHARED helper
        # (see its docstring for the probe rules and the ambiguity WARNING).
        # mc_by_prefix is the single source for both the helper argument and
        # the per-group lookup below.
        mc_by_prefix: dict[str, ModuleConfig] = {mc.prefix: mc for mc in module_configs}
        groups = _group_node_ids_by_subproject(
            worktree, mc_by_prefix, node_ids,
            log_label='confirm_merge_verify_flake_suppressible',
        )
        # `not groups`, NOT `is None`: both sentinels must fail CLOSED. Falling
        # into the groups.items() loop with an empty dict would exit having run
        # ZERO isolated re-runs and then `return node_ids` — a full suppression
        # verdict on zero evidence, letting a genuinely red merge land. Mirrors
        # the same defensive guard in _sweep_failure_reproduces_in_isolation.
        if not groups:
            # Name the offending node-ids HERE so this ONE merge-lane line
            # answers both "which tests failed to map?" and "what did the gate
            # decide?" without correlating a second line: the shared helper's
            # own INFO names the first unmappable node-id but only knows the
            # neutral 'unconfirmable', while the verdict vocabulary ('not
            # suppressing') is the caller's alone. Rendering `groups` also
            # separates the reachable None case from the defensive-only {}.
            # The preview is bounded — a mass failure can extract hundreds of
            # node-ids, and a log line is not a report.
            shown = node_ids[:10]
            extra = len(node_ids) - len(shown)
            logger.info(
                'confirm_merge_verify_flake_suppressible: node-id -> subproject '
                'mapping yielded nothing usable (%r) for %s%s in %s — '
                'unconfirmable, not suppressing',
                groups, shown, f' (+{extra} more)' if extra else '', worktree,
            )
            return None

        # Each subproject group gets its own scoped + forced-serial +
        # generous-timeout isolated re-run in the SAME merge worktree. ALL
        # groups must confirm green to suppress.
        for prefix, group_node_ids in groups.items():
            mc = mc_by_prefix[prefix]
            scoped_cmd = _with_pytest_timeout_str(
                _serial_pytest_str(
                    _scope_to_keyword(mc.test_command, 'pytest', group_node_ids),
                ),
                _MERGE_FLAKE_CONFIRM_TIMEOUT_SECS,
            )
            scoped_mc = replace(
                mc, test_command=scoped_cmd, lint_command=None, type_check_command=None,
            )
            result = await run_verification(
                worktree, config, scoped_mc,
                max_retries=0, is_merge_verify=True, role='merge',
            )
            # An infra-sentinel category is never trusted as confirmation, even
            # paired with passed=True (mirrors _run_isolated_confirm_group).
            if result.category in INFRA_TRANSIENT_CATEGORIES or not result.passed:
                logger.info(
                    'confirm_merge_verify_flake_suppressible: isolated re-run '
                    'for %s did not confirm green (category=%r, passed=%s) — '
                    'not suppressing',
                    prefix, result.category, result.passed,
                )
                return None

        logger.info(
            'confirm_merge_verify_flake_suppressible: merge-verify flake '
            'confirmed suppressible: %s failed under load, passed on isolated '
            're-run in %s',
            node_ids, worktree,
        )
        return node_ids

    except Exception:
        logger.warning(
            'confirm_merge_verify_flake_suppressible: unexpected error — '
            'failing closed to red',
            exc_info=True,
        )
        return None


# ---------------------------------------------------------------------------
# Main-probe isolated-flake confirm gate (task 3597)
# ---------------------------------------------------------------------------

#: Generous per-test timeout (seconds) injected into the main-probe confirm
#: gate's isolated re-run command via ``_with_pytest_timeout_str``. A
#: SEPARATE constant from ``_SWEEP_CONFIRM_TIMEOUT_SECS`` /
#: ``_MERGE_FLAKE_CONFIRM_TIMEOUT_SECS`` so the three are retuned on their own
#: signals. Same rationale as both: ``-o addopts=`` clears pyproject
#: ``addopts`` but NOT the ``[tool.pytest.ini_options] timeout=60`` default,
#: so without this explicit override a still-loaded host can starve the
#: isolated confirm run into a false "still fails" verdict — and here that
#: false verdict would mean a false red-main verdict (BLOCKED awaiting a
#: hotfix for a main that is actually green) survives untouched.
_MAIN_PROBE_CONFIRM_TIMEOUT_SECS: int = 300

#: Per-signature repeat counter for the main-probe isolated-flake downgrade
#: (reviewer_comprehensive finding 4, task 3597 amendment pass). The
#: downgraded verdict is deliberately NOT written to ``_PROBE_CACHE`` (a
#: load-flake verdict is a statement about transient host state, not a fact
#: about the main tip — see the call site), so a sustained CPU-starvation
#: storm makes every sibling task that hits the same (main_sha, category,
#: hint) signature repeat the full probe + isolated-rerun cost. Caching the
#: downgrade instead would reintroduce exactly the false-negative risk the
#: no-cache decision exists to avoid, so this dict only makes the repetition
#: OBSERVABLE (loud-over-silent-degradation norm) without changing the
#: caching decision. Grows one entry per distinct downgraded signature for
#: the life of the process — unpruned, mirroring ``_PROBE_CACHE``'s own
#: unbounded-growth precedent (neither is actively pruned; both are small,
#: bounded by the number of distinct signatures ever seen).
_MAIN_PROBE_DOWNGRADE_REPEAT_COUNTS: dict[tuple[str, str, str], int] = {}


async def _main_probe_failure_is_isolated_flake(
    probe_worktree: Path,
    config: 'OrchestratorConfig',
    module_configs: 'list[ModuleConfig]',
    main_result: VerifyResult,
) -> list[str] | None:
    """CONFIRM GATE: did *main_result*'s named failures pass on isolated re-run?

    Called by ``verify_failure_is_preexisting_on_main`` immediately before it
    would return ``(True, main_sha)`` — i.e. after the main probe's
    (category, normalised cause_hint) signature has already matched the
    branch's. A CPU-starvation load flake can starve the SAME
    timing-sensitive test on both the task branch and the main probe,
    matching signatures on a main that is actually green (ground truth:
    esc-3514-2 / task 3514 — 12 named failures on an aborted xdist run, all
    passing in isolation, main was not broken). This gate re-runs JUST the
    node-ids named in *main_result*'s own failing output — scoped,
    forced-serial (``-p no:xdist -o addopts=``), generous-timeout — inside
    the ALREADY-OPEN *probe_worktree* pinned at main (SAME-TREE, no second
    ``git worktree add``; the caller owns the worktree's lifecycle).

    PRECONDITION — test-leg-only (reviewer_comprehensive finding 1):
    ``run_verification``'s ``_summarize_checks``/``_worst_category`` picks
    ONE ``category`` across up to three legs (test/lint/type), so a matched
    (category, cause_hint) signature does NOT by itself guarantee the
    lint/type legs were clean on main — a genuine, co-occurring lint/type
    break can lose the "worst" contest to the test leg's category (e.g. it
    classifies as the lower-priority ``unknown_test_failure``) and still be
    real. Re-running just the named TEST node-ids would then wrongly
    downgrade a genuinely red main. ``VerifyResult.lint_output``/
    ``type_output`` are populated ONLY when that leg's return code is
    non-zero (see ``run_verification``'s result construction), so a
    non-empty value here is a precise, free signal that another leg is not
    clean — this bails to ``None`` before doing any work.

    Returns:
        ``list[str]`` — every named failing test on THIS main tip
        demonstrably PASSED on isolated re-run. The caller MAY downgrade its
        verdict to ``(False, '')``, which is ``verify_failure_is_preexisting_on_main``'s
        own documented fail-safe return.

        ``None`` — every other (fail-safe) path; the caller keeps today's
        ``(True, main_sha)`` verdict unchanged. Covers: a co-occurring
        lint/type break on main (see PRECONDITION above); no recoverable
        node-id in ``main_result.test_output`` (an opaque/lint/type-only
        failure); a node-id that maps to no discovered subproject (or
        *module_configs* is empty); an isolated re-run that still fails,
        times out, or hits an infra-sentinel category
        (``INFRA_TRANSIENT_CATEGORIES`` — never trusted as confirmation)
        after ``_SWEEP_CONFIRM_MAX_ATTEMPTS`` attempts; and any unexpected
        exception (never raises).

    ONE-WAY RATCHET: this gate can only downgrade a verdict that an isolated
    re-run has POSITIVELY shown green. Every degraded path preserves today's
    verdict, so nothing about a genuinely red main changes.

    Node-id -> subproject mapping delegates to
    ``_group_node_ids_by_subproject`` over ``{mc.prefix: mc for mc in
    module_configs}`` — the FOURTH call site (after
    ``confirm_main_tip_failure_is_real``,
    ``_sweep_failure_reproduces_in_isolation``, and
    ``confirm_merge_verify_flake_suppressible``). The isolated re-run engine
    is ``_run_isolated_confirm_group`` (the bounded
    ``_SWEEP_CONFIRM_MAX_ATTEMPTS``-attempt loop, ``role='task'`` by
    default via ``run_verification``'s own default), unchanged.

    *module_configs* SOURCE (reviewer_comprehensive finding 5): this is the
    CALLER's snapshot — discovered on the task branch's worktree by
    ``verify_failure_is_preexisting_on_main``'s own caller — reused as-is
    rather than re-discovered on *probe_worktree*, unlike
    ``confirm_main_tip_failure_is_real`` (which re-runs
    ``_discover_module_configs`` against its own fresh probe tree). This is
    consistent with, not a new departure from,
    ``verify_failure_is_preexisting_on_main``'s own PRE-EXISTING behaviour:
    it already runs ``run_scoped_verification(tmp_path, config,
    module_configs, ...)`` — the same branch-side *module_configs* — against
    this same main-pinned *probe_worktree* to produce *main_result* in the
    first place, so reusing it again here for node-id mapping introduces no
    tree/config mismatch beyond what that earlier call already accepts. A
    task that renames/adds a subproject or changes a ``test_command``
    between branch and main degrades fail-safe, not silently wrong: a stale
    prefix yields either no recoverable node-id (early-out above), an
    unmapped node-id (``_group_node_ids_by_subproject`` returns ``None``,
    below), or a scoped command that errors/still-fails on the probe tree
    (``_run_isolated_confirm_group`` returns ``False``) — every one of those
    keeps today's verdict rather than producing a wrong downgrade.
    """
    try:
        # PRECONDITION (finding 1 above): bail before any work when another
        # leg is known non-clean on main.
        if main_result.lint_output or main_result.type_output:
            return None

        node_ids = _extract_failing_test_ids(main_result.test_output)
        if not node_ids:
            return None

        mc_by_prefix: dict[str, ModuleConfig] = {mc.prefix: mc for mc in module_configs}
        groups = _group_node_ids_by_subproject(
            probe_worktree, mc_by_prefix, node_ids,
            log_label='verify_failure_is_preexisting_on_main confirm gate',
        )
        # `not groups`, NOT `is None`: both sentinels must keep today's
        # verdict. Falling into the groups.items() loop with an empty dict
        # would run ZERO isolated re-runs and then `return node_ids` — a
        # full downgrade on zero evidence. Mirrors the same defensive guard
        # in confirm_merge_verify_flake_suppressible /
        # _sweep_failure_reproduces_in_isolation.
        if not groups:
            shown = node_ids[:10]
            extra = len(node_ids) - len(shown)
            logger.info(
                'verify_failure_is_preexisting_on_main confirm gate: node-id '
                '-> subproject mapping yielded nothing usable (%r) for %s%s '
                'in %s — unconfirmable, keeping the preexisting verdict',
                groups, shown, f' (+{extra} more)' if extra else '', probe_worktree,
            )
            return None

        for prefix, group_node_ids in groups.items():
            mc = mc_by_prefix[prefix]
            scoped_cmd = _with_pytest_timeout_str(
                _serial_pytest_str(
                    _scope_to_keyword(mc.test_command, 'pytest', group_node_ids),
                ),
                _MAIN_PROBE_CONFIRM_TIMEOUT_SECS,
            )
            scoped_mc = replace(
                mc, test_command=scoped_cmd, lint_command=None, type_check_command=None,
            )
            if not await _run_isolated_confirm_group(probe_worktree, config, scoped_mc):
                return None

        logger.warning(
            'verify_failure_is_preexisting_on_main confirm gate: %s passed on '
            'isolated re-run on the main probe — main is not red for these '
            'tests',
            node_ids,
        )
        return node_ids

    except Exception:
        logger.warning(
            'verify_failure_is_preexisting_on_main confirm gate: unexpected '
            'error — keeping the preexisting verdict',
            exc_info=True,
        )
        return None


def _merge_flake_suppressed_pass(
    failing_result: VerifyResult, node_ids: list[str],
) -> VerifyResult:
    """Turn a confirmed-flake *failing_result* into a PASSED VerifyResult.

    Reuses ``dataclasses.replace`` so the original log paths / plan / duration
    are preserved (the durable evidence of the suppressed red survives),
    flipping only ``passed``/``timed_out``/``category``/``summary``. The
    ``merge_flake_suppressed`` category lets the merge proceed into the unscoped
    typecheck gate (``LocalRunner.run_merge_verify`` falls through on a passed
    scoped result) while staying greppable in logs/archives.
    """
    joined = ', '.join(node_ids)
    return replace(
        failing_result,
        passed=True,
        timed_out=False,
        category='merge_flake_suppressed',
        summary=f'merge-verify flake suppressed (isolated re-run passed): {joined}',
    )


def _emit_merge_flake_suppressed(
    event_store: 'EventStore | None',
    task_id: str | None,
    merge_sha: str,
    node_ids: list[str],
) -> None:
    """Emit the INV-2 structured suppression fact. None-safe (skips on None).

    ``EventType`` is imported lazily to avoid any import-order coupling on this
    central module (event_store.py has no reverse dependency on verify.py, but
    the lazy import keeps it that way by construction).
    """
    if event_store is None:
        return
    from orchestrator.event_store import EventType  # noqa: PLC0415 — lazy, avoid cycle

    event_store.emit(
        EventType.merge_flake_suppressed,
        task_id=task_id,
        data={
            'node_ids': node_ids,
            'merge_sha': merge_sha,
            'measured_at': datetime.now(UTC).isoformat(),
        },
    )


def _emit_trivial_pass_escalated(
    event_store: 'EventStore | None',
    task_id: str | None,
    *,
    reason: str,
    resolution: str,
) -> None:
    """Emit the INV-1 structured escalation fact (task 2883). None-safe.

    Emitted by :func:`run_scoped_verification` when the merge gate (role='merge'
    AND is_merge_verify) escalates a would-be trivial pass to the full gate
    (``resolution='full_gate'``) or FAILs loud (``resolution='loud_fail'``).
    ``reason`` ∈ {no_source_files, empty_existing_files, empty_command_set}.

    ``EventType`` is imported lazily to avoid any import-order coupling on this
    central module (mirrors :func:`_emit_merge_flake_suppressed`).  The remote
    in-worktree LocalRunner leaves *event_store* None (it cannot reach the
    dispatching store), so only the dispatch-side event is local — the
    correctness fix still applies remotely.
    """
    if event_store is None:
        return
    from orchestrator.event_store import EventType  # noqa: PLC0415 — lazy, avoid cycle

    event_store.emit(
        EventType.trivial_pass_escalated,
        task_id=task_id,
        role='merge',
        data={
            'reason': reason,
            'resolution': resolution,
            'measured_at': datetime.now(UTC).isoformat(),
        },
    )


#: Module-global suppression counter (INV-4 storm detector). Bumped ONLY on a
#: suppression; reset to 0 only once the window (threshold) is reached and the
#: storm escalation decision is made. A clean, non-suppressed merge-verify does
#: NOT reset it, so this is a CUMULATIVE count of suppressions since the last
#: reset — NOT a count of back-to-back (consecutive) merges. A count-window
#: detector; time-windowing is a sanctioned PRD §9 follow-up.
_merge_flake_suppression_streak = 0

#: Suppressions per window before the born-at-L2 storm escalation fires. A
#: tunable (PRD §9): chronic suppression means α is repeatedly masking reds —
#: a fleet-health "someone must look now" condition.
_MERGE_FLAKE_SUPPRESSION_STREAK_THRESHOLD = 5

#: Fixed dedup sentinel task_id for the storm escalation — the signal is a
#: global fleet-health condition, not tied to any one merge task.
_MERGE_FLAKE_SUPPRESSION_STORM_SENTINEL = 'merge-flake-suppression-storm'


def _bump_suppression_streak_and_maybe_escalate(
    escalation_queue: Any, task_id: str | None, merge_sha: str,
) -> None:
    """Advance the suppression streak; file a born-at-L2 storm escalation at
    the threshold, then reset the counter (INV-4).

    Modeled on ``merge_queue._alarm_verify_worktree_contention``: a born-at-L2
    escalation (``severity='critical'``, ``level=2``,
    ``agent_role='orchestrator-merge-flake-monitor'`` — the ``orchestrator-``
    prefix marks it a harness sentinel so the escalation server never downgrades
    the critical severity) that routes straight to a human, bypassing the
    auto-watcher. Deduped on a fixed open-L2 sentinel task_id so a persistent
    storm files at most one open critical per window.

    The window resets to 0 whenever the threshold is reached — on submit, on a
    dedup-skip, AND on a ``None`` queue — so the counter can never grow
    unbounded and each fresh window makes an independent escalation decision.
    None-safe: with no queue there is nothing to file into, so it resets and
    returns (the CLI / remote paths that pass ``escalation_queue=None`` are not
    the CPU-starvation target this gate addresses — see the α scope fence).
    """
    global _merge_flake_suppression_streak
    _merge_flake_suppression_streak += 1
    if _merge_flake_suppression_streak < _MERGE_FLAKE_SUPPRESSION_STREAK_THRESHOLD:
        return

    # Window reached: make the escalation decision once, then reset regardless.
    _merge_flake_suppression_streak = 0
    if escalation_queue is None:
        return

    from escalation.models import Escalation  # noqa: PLC0415 — local, escalation optional dep

    sentinel = _MERGE_FLAKE_SUPPRESSION_STORM_SENTINEL
    # Dedup: don't re-alarm while an open L2 already exists for the storm
    # sentinel (has_open_l1 is hardcoded to level=1, so get_by_task is used).
    if escalation_queue.get_by_task(sentinel, status='pending', level=2):
        return

    summary = (
        'Merge-verify flake-suppression storm: the isolated-rerun-confirm gate '
        f'has suppressed {_MERGE_FLAKE_SUPPRESSION_STREAK_THRESHOLD} merge-verify '
        'reds since the last reset'
    )
    detail = (
        f'The role=merge isolated-rerun-confirm gate (verify.'
        f'apply_merge_flake_suppression) has suppressed '
        f'{_MERGE_FLAKE_SUPPRESSION_STREAK_THRESHOLD} merge-verify failures as '
        f'CPU-starvation flakes since the counter was last reset — a CUMULATIVE '
        f'count, NOT necessarily back-to-back merges (a clean merge-verify does '
        f'not reset the counter). Most recent merge SHA: {merge_sha}, task_id: '
        f'{task_id}. Each suppression means a merge-verify red passed on isolated '
        're-run — but a sustained rate of suppressions indicates either chronic '
        'host CPU starvation or a genuinely flaky test that is being repeatedly '
        'masked. Investigate before the gate hides a real regression.'
    )
    esc = Escalation(
        id=escalation_queue.make_id(sentinel),
        task_id=sentinel,
        agent_role='orchestrator-merge-flake-monitor',
        severity='critical',
        level=2,
        category='merge_flake_suppression_storm',
        summary=summary,
        detail=detail,
        suggested_action=(
            'Inspect merge-flake-suppressed events (EventType.merge_flake_suppressed) '
            'and host CPU load. Confirm the suppressed tests are load flakes, not a '
            'masked regression; if a specific test is chronically flaky, de-flake or '
            'quarantine it.'
        ),
    )
    escalation_queue.submit(esc)


async def apply_merge_flake_suppression(
    failing_result: VerifyResult,
    *,
    worktree: Path,
    config: 'OrchestratorConfig',
    module_configs: list[ModuleConfig],
    merge_sha: str,
    event_store: 'EventStore | None' = None,
    escalation_queue: Any = None,
    task_id: str | None = None,
    _confirm=confirm_merge_verify_flake_suppressible,
) -> VerifyResult:
    """Merge-verify result handler: suppress a confirmed CPU-starvation flake.

    THE hook ``LocalRunner.run_merge_verify`` calls on its ``not scoped.passed``
    branch (PRD task α). Runs the pure gate *_confirm*; on a confirmed flake it
    emits the INV-2 fact, bumps the INV-4 storm streak, and returns a PASSED
    VerifyResult (category ``merge_flake_suppressed``) so the merge proceeds
    into the unscoped typecheck gate. On a non-confirmation it returns
    *failing_result* UNCHANGED (merge stays red; no fact, streak untouched).

    Never raises: the pure gate is itself fail-closed and non-raising, and the
    fact/streak side-effects are None-safe — an uncaught raise here would stall
    the merge queue (merge_queue.py has no VerifyInfraError handler). *_confirm*
    is injectable for testing.
    """
    ids = await _confirm(
        config, failing_result, worktree=worktree, module_configs=module_configs,
    )
    if not ids:
        return failing_result
    _emit_merge_flake_suppressed(event_store, task_id, merge_sha, ids)
    _bump_suppression_streak_and_maybe_escalate(escalation_queue, task_id, merge_sha)
    return _merge_flake_suppressed_pass(failing_result, ids)
