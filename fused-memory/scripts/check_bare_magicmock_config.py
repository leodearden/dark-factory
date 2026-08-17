#!/usr/bin/env python3
"""Lint checks: mock-spec discipline in test files.

This script carries TWO INDEPENDENT RULES.  They share only the AST predicates
(``_is_magicmock_call`` / ``_is_specced``), the exemption-comment contract and the
output format; they have separate detection pipelines, separate message vocabularies
and separate ``# noqa`` codes.  Every statement in the "Rule A" section below is
scoped to Rule A and says nothing about Rule B.

  Rule A — ``bare-magicmock`` (tasks 1339/1372)
      Config-NAMED bindings only, ``ast.Assign``/``ast.AnnAssign`` positions only.
      Remedies are pydantic-specific (``mock_orch_config``, ``pydantic_spec``).

  Rule B — ``bare-dataclass-double`` (task 4016)
      Registry-driven and POSITION-BLIND: any ``ast.Call`` anywhere — including
      ``return MagicMock(...)``, an argument, a comprehension body — whose literal
      kwargs match a registered stdlib-dataclass shape (``_DATACLASS_SHAPES``;
      ``VerifyResult`` today).  The binding name is never consulted.  Remedies are
      dataclass-specific (``_fake_verify_result``, ``MagicMock(spec=VerifyResult)``).
      Carries a shrink-only per-file debt baseline (``_DATACLASS_DOUBLE_DEBT``).

WIDEN-NOT-SIBLING RULING (task 4016): Rule B was added here rather than as a sibling
script.  A sibling would have cost nine wiring edits (seven package ``orchestrator.yaml``
lint_commands, ``dark-factory-orchestrator.yaml``, ``hooks/project-checks``), a second
``python3`` process per lint run and a second fleet-lint-coverage entry — all to run the
same ``ast.parse`` over the same files.  Widening cost zero wiring edits.  Rule A's
stated non-goals below guard against inflating the CONFIG-NAME set, which would blur one
rule's boundary; adding a second rule with its own name, vocabulary and noqa code is the
opposite of that scope creep.  The FILENAME is therefore a deliberately retained legacy
name — it no longer describes the whole file, and renaming it would touch those same nine
call sites and break every in-flight branch, for a cosmetic gain.

---------------------------------------------------------------------------
Rule A — ``bare-magicmock``
---------------------------------------------------------------------------

Rule: Any assignment of the form ``<config_name> = MagicMock()`` where the call has
no ``spec``, no ``spec_set`` keyword argument, and no positional argument (MagicMock's
first positional IS spec) is a violation — unless the immediately-preceding non-blank
source line is a structured exemption comment:

    # noqa: bare-magicmock — <reason>   (em-dash or ASCII hyphen; reason must be non-empty)

Inline trailing exemption NOT honored: a ``# noqa: bare-magicmock`` comment placed on
the *same* line as the assignment (e.g. ``config = MagicMock()  # noqa: bare-magicmock — x``)
is intentionally ignored.  Only the nearest preceding non-blank source line is consulted.
Placing the exemption on the same line as the violating code is a common footgun with
ruff-style suppressions; keeping the contract to a dedicated preceding line makes
exemptions both auditable and deliberate.

Config-name set: exact ``config`` and ``cfg``, plus any name ending with ``_config`` or
``_cfg`` (e.g. ``orch_config``, ``mock_cfg``).  Generic names like ``mcp``, ``mock``, ``m``
are intentionally excluded — this rule targets config objects only (non-goal: no scope creep).

Attribute target exclusion: only ``ast.Name`` assignment targets are detected (i.e.,
module-level and local ``config = MagicMock()``).  Attribute assignments such as
``self.config = MagicMock()`` or ``obj.cfg = MagicMock()`` are intentionally excluded —
resolving attribute targets requires class/instance context that is not available during
AST inspection.  This exclusion is an intentional non-goal; it is documented here so the
scope limit is discoverable.

Tuple/list-unpacking non-goal: ``a, b = MagicMock()`` uses an ``ast.Tuple`` target, not
an ``ast.Name``, so it is excluded by the same ast.Name-only rule.  Inspecting unpacked
targets would require data-flow analysis to identify which element ends up in a
config-named binding; this is intentionally out of scope.

Chained/multi-target assignment: ``mock = config = MagicMock()`` is an ``ast.Assign``
with ``len(node.targets) == 2``.  Each ``ast.Name`` target is evaluated independently
against the shared RHS value (a single MagicMock call).  One Violation is emitted per
config-named ``ast.Name`` target:
  • ``mock = config = MagicMock()``  → 1 violation  (``config`` only)
  • ``config = cfg = MagicMock()``   → 2 violations (both config-named)
  • ``mock = other = MagicMock()``   → 0 violations (no config-named targets)
Spec and exemption checks apply once per target (shared value/lineno; per-target
``col_offset``).

Preferred alternatives named in the rejection message:
  • ``mock_orch_config`` fixture (orchestrator/tests/conftest.py:91)
  • ``MagicMock(spec_set=pydantic_spec(...))`` (orchestrator/tests/_orch_helpers.py:19)

Origin: Task 1339 (migrate existing bare configs), task 1313/1064 (spec discipline).
This guard is implemented in task 1372 to prevent regressions after the migration.

---------------------------------------------------------------------------
Rule B — ``bare-dataclass-double``
---------------------------------------------------------------------------

Rule: any ``MagicMock(...)`` call, IN ANY POSITION, with no spec/spec_set and no
positional argument, whose literal keyword names match a shape registered in
``_DATACLASS_SHAPES`` — unless the preceding non-blank line carries
``# noqa: bare-dataclass-double — <reason>``, or the file is on the debt baseline.

Matching is ANCHOR + OVERLAP, deliberately NOT "kwargs are a subset of the fields":
every anchor must be present AND at least ``min_field_matches`` fields must match.
A kwarg that is not a field is *drift evidence* named in the message, never an
exemption — a bare MagicMock accepts any keyword silently, so an unrecognised one is
the strongest signal the double has drifted from the type it impersonates.  A subset
rule would have missed all ten sites behind task 3980, every one of which passed
``verify_skipped=`` (a MergeOutcome field, merge_types.py:945).

Why this rule exists: Rule A provably cannot see the shape, for three independent
reasons, any one of them fatal — it inspects only ``ast.Assign``/``ast.AnnAssign``
while all ten task-3980 sites were ``return MagicMock(...)``; ``_is_config_name``
matches only config/cfg/*_config/*_cfg targets; and its remedies read pydantic
``model_fields`` while ``VerifyResult`` is a stdlib dataclass.

Preferred alternatives named in the rejection message:
  • ``_fake_verify_result(...)`` (orchestrator/tests/test_merge_queue_concurrent_verify.py:484)
  • ``MagicMock(spec=VerifyResult)`` seeded from ``dataclasses.fields(VerifyResult)``

Origin: task 3477 (built the factory), task 3980 (migrated ten sites and added a
file-local guard), task 4016 (this shared, repo-wide guard).

---------------------------------------------------------------------------

This script is intentionally stdlib-only (ast, argparse, pathlib, re, sys, typing) so
hooks/project-checks can invoke it via plain python3 without uv env-resolution overhead.
Adding a third-party dependency here would break that fast path.  This is why
``_DATACLASS_SHAPES`` hardcodes field names instead of importing the dataclasses it
describes: ``import orchestrator.verify`` would need pydantic and break every caller.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path, PurePosixPath
from typing import NamedTuple


class Violation(NamedTuple):
    """A lint violation found by the checker."""

    filename: str
    lineno: int
    col_offset: int
    message: str


# Names that the checker considers "config objects".
# Exact matches plus suffix rules — generic names are excluded by design.
_CONFIG_EXACT: frozenset[str] = frozenset({'config', 'cfg'})
_CONFIG_SUFFIXES: tuple[str, ...] = ('_config', '_cfg')


class _DataclassShape(NamedTuple):
    """A dataclass whose *shape* Rule B recognises in unspecced MagicMock kwargs.

    Matching is ANCHOR + OVERLAP, never "kwargs are a subset of fields":
      - every name in ``anchors`` must appear among the call's literal kwarg names, AND
      - at least ``min_field_matches`` of ``fields`` must be matched.

    A kwarg that is NOT a field does not exempt the call — it is *additional drift
    evidence* and gets named in the violation message.  This is load-bearing: all ten
    sites behind task 3980 passed ``verify_skipped=``, a MergeOutcome field
    (merge_types.py:945) that VerifyResult does not have, so a subset rule would have
    missed precisely the defect this rule exists to catch.
    """

    name: str
    module: str
    fields: frozenset[str]
    anchors: frozenset[str]
    min_field_matches: int
    factory: str


# Registered shapes.  The field lists are LITERALS, not imports: this script is
# stdlib-only by hard contract (see module docstring) so hooks/project-checks and
# all seven package lint_commands can run it under bare ``python3`` with no venv
# resolution.  ``import orchestrator.verify`` would need pydantic and break every caller.
#
# VerifyResult's field list mirrors orchestrator/src/orchestrator/verify.py:3216.
# Drift is absorbed structurally rather than by keeping this list exhaustive:
# matching keys on the ``passed`` anchor plus a 2-field overlap floor, so adding,
# renaming or removing a peripheral field cannot silently disable detection.  Only
# removing ``passed`` itself could, and that is a VerifyResult refactor that would
# break the orchestrator far more loudly first.
_DATACLASS_SHAPES: tuple[_DataclassShape, ...] = (
    _DataclassShape(
        name='VerifyResult',
        module='orchestrator.verify',
        fields=frozenset({
            'passed',
            'test_output',
            'lint_output',
            'type_output',
            'summary',
            'timed_out',
            'cause_hint',
            'category',
            'worktree_log_paths',
            'archive_log_paths',
            'contention',
            'plan',
            'failing_test_ids',
            'failing_leg_categories',
            'trivial',
            'duration_secs',
        }),
        anchors=frozenset({'passed'}),
        min_field_matches=2,
        factory='_fake_verify_result',
    ),
)


def _is_config_name(name: str) -> bool:
    """Return True if *name* identifies a config-named variable."""
    if name in _CONFIG_EXACT:
        return True
    return any(name.endswith(suffix) for suffix in _CONFIG_SUFFIXES)


def _is_magicmock_call(node: ast.expr) -> bool:
    """Return True if *node* is a call whose func resolves to ``MagicMock``.

    Accepts:
      - ``MagicMock(...)``                  → ast.Name(id='MagicMock')
      - ``mock.MagicMock(...)``             → ast.Attribute(attr='MagicMock')
      - ``unittest.mock.MagicMock(...)``    → ast.Attribute(attr='MagicMock')
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == 'MagicMock'
    if isinstance(func, ast.Attribute):
        return func.attr == 'MagicMock'
    return False


def _is_specced(call: ast.Call) -> bool:
    """Return True if *call* provides a real spec via positional arg or spec/spec_set kwarg.

    Edge cases handled explicitly:
    - ``MagicMock(*args)`` (ast.Starred positional): treated as NOT specced.
      The spread is opaque at AST-inspection time, so we cannot guarantee a spec
      is present; flagging is safer than a false negative.
    - ``MagicMock(**kwargs)`` (double-starred keyword spread): treated as NOT specced.
      Like ``*args``, the spread is opaque — even if the dict contains a ``spec`` key
      we cannot verify it statically; conservative flagging mirrors the *args stance.
    - ``MagicMock(spec=None)`` / ``MagicMock(spec_set=None)``: treated as NOT specced.
      Passing None is semantically equivalent to omitting spec altogether and defeats
      the rule's intent.
    """
    # Concrete (non-Starred) positional args only — *args spread cannot be inspected.
    concrete_args = [a for a in call.args if not isinstance(a, ast.Starred)]
    if concrete_args:
        # First positional parameter of MagicMock IS spec.
        return True
    # spec/spec_set keyword args — but only when the value is not the literal None.
    for kw in call.keywords:
        if kw.arg in ('spec', 'spec_set') and not (
            isinstance(kw.value, ast.Constant) and kw.value.value is None
        ):
            return True
    return False


# Exemption comment regexes, one per rule code.
# Matches: ``# noqa: <code> — <non-empty-reason>``
# Accepts em-dash (—) or ASCII hyphen (-) as separator.
# Requires at least one non-space character after the separator.
#
# Each rule gets its OWN code so a suppression written for one can never silently
# exempt the other: the remedies are unrelated (mock_orch_config/pydantic_spec vs
# _fake_verify_result/spec=VerifyResult), so a pragma for one is not informed
# consent for the other.  Both are built from the same template, so the
# em-dash/ASCII-hyphen and mandatory-reason contract is identical across rules.
_EXEMPT_TEMPLATE = r'#\s*noqa:\s*{code}\s*[—\-]+\s*\S.*'

_RULE_A_CODE = 'bare-magicmock'
_RULE_B_CODE = 'bare-dataclass-double'

# Kept at its historical value so Rule A's behaviour is bit-identical.
_EXEMPT_RE = re.compile(_EXEMPT_TEMPLATE.format(code=re.escape(_RULE_A_CODE)))

_EXEMPT_RES: dict[str, re.Pattern[str]] = {
    _RULE_A_CODE: _EXEMPT_RE,
    _RULE_B_CODE: re.compile(_EXEMPT_TEMPLATE.format(code=re.escape(_RULE_B_CODE))),
}

_VIOLATION_MSG = (
    'bare MagicMock() assigned to a config variable with no spec/spec_set.'
    ' Use the mock_orch_config fixture or MagicMock(spec_set=pydantic_spec(...))'
    ' instead (see task 1339 regression guard; tasks 1313/1064).'
    ' To suppress: add # noqa: bare-magicmock — <reason> on the preceding non-blank line.'
)


def _dataclass_violation_msg(shape: _DataclassShape, kwargs: set[str]) -> str:
    """Build Rule B's rejection message for *shape* matched by *kwargs*.

    Deliberately shares NO vocabulary with ``_VIOLATION_MSG``: Rule A's remedies
    (``mock_orch_config`` / ``pydantic_spec``) read ``model_fields`` and require a
    pydantic BaseModel, so they are unusable for a stdlib dataclass.  Offering them
    here would send the reader down a dead end.

    Any kwarg that is not a field of *shape* is reported as drift evidence: a bare
    MagicMock accepts any keyword without objection, so an unrecognised one is the
    single strongest signal that the double has drifted from the type it impersonates.
    """
    drift = sorted(kwargs - shape.fields)
    drift_clause = ''
    if drift:
        names = ', '.join(drift)
        drift_clause = (
            f' It also passes {names} — not {"a field" if len(drift) == 1 else "fields"}'
            f' of {shape.name}, which a bare MagicMock accepts silently.'
        )
    return (
        f'unspecced MagicMock shaped like {shape.module}.{shape.name}'
        f' (matched {shape.name} fields: {", ".join(sorted(kwargs & shape.fields))}).'
        ' Reading an absent attribute on it auto-vivifies a truthy child Mock instead'
        f' of raising AttributeError.{drift_clause}'
        f' Use the {shape.factory}(...) helper, or MagicMock(spec={shape.name}) seeded'
        f' from dataclasses.fields({shape.name}), so unknown-attribute reads raise'
        ' (tasks 3477/3980 built the factory; task 4016 added this guard).'
        ' To suppress: add # noqa: bare-dataclass-double — <reason> on the preceding'
        ' non-blank line.'
    )


# ---------------------------------------------------------------------------
# Rule B debt baseline — SHRINK-ONLY.
#
# These 11 files carried 95 pre-existing dataclass-double sites when Rule B
# landed (AST census over all seven scanned tests/ directories, task 4016).
# Shipping the rule hot with no transition would have turned orchestrator/tests'
# lint_command red on day one and stalled the merge lane repo-wide, so they are
# grandfathered — Rule B ONLY; Rule A still applies in full to every file here.
#
# DO NOT ADD ENTRIES.  The list may only shrink, as files are migrated onto
# _fake_verify_result / MagicMock(spec=VerifyResult); that migration is filed as
# a follow-up task. A NEW file with a bare dataclass double must fail the gate —
# that is the entire reason the baseline is opt-OUT rather than opt-in.
#
# orchestrator/tests/test_merge_speculation.py is deliberately NOT here: task
# 3980 just cleaned that module, and a blanket entry would silently un-cover it.
# Its one deliberate double carries a per-site pragma instead.
_DATACLASS_DOUBLE_DEBT: frozenset[str] = frozenset({
    'orchestrator/tests/test_merge_queue.py',  # 63 sites
    'orchestrator/tests/test_concurrent_verify_boundary.py',  # 9 sites
    'orchestrator/tests/test_merge_queue_permit_conservation.py',  # 7 sites
    'orchestrator/tests/test_merge_queue_resolve_release.py',  # 7 sites
    'orchestrator/tests/test_merge_queue_request_liveness.py',  # 3 sites
    'orchestrator/tests/test_coalesce_integration_gate.py',  # 1 site
    'orchestrator/tests/test_merge_item_union.py',  # 1 site
    'orchestrator/tests/test_merge_queue_equivalence.py',  # 1 site
    'orchestrator/tests/test_merge_queue_lifecycle_registry.py',  # 1 site
    'orchestrator/tests/test_merge_queue_metrics.py',  # 1 site
    'orchestrator/tests/test_merge_queue_single_writer_asserts.py',  # 1 site
})


def _is_debt_path(filename: str) -> bool:
    """Return True if *filename* is a Rule B debt-baseline file.

    Compares TRAILING PATH COMPONENTS, not substrings: ``a/b/c.py`` matches an
    entry ``b/c.py`` because its last two components are exactly ``b`` and ``c.py``.
    Component-awareness is what stops ``orchestrator/tests/not_test_merge_queue.py``
    — which merely contains a debt filename as a substring — from being grandfathered.

    Trailing-component matching (rather than an exact string compare) is required
    because the nine call sites pass repo-relative paths while pytest passes
    absolutes; both must reach the same verdict.
    """
    parts = PurePosixPath(filename.replace('\\', '/')).parts
    for entry in _DATACLASS_DOUBLE_DEBT:
        entry_parts = PurePosixPath(entry).parts
        if len(parts) >= len(entry_parts) and parts[-len(entry_parts) :] == entry_parts:
            return True
    return False


def _literal_kwarg_names(call: ast.Call) -> set[str]:
    """Return *call*'s literal keyword-argument names.

    The ``kw.arg is not None`` guard is what makes a ``**spread`` a non-match: CPython
    represents ``MagicMock(**kw)`` as a keyword whose ``arg`` is None.  A spread exposes
    no literal name, so it can never satisfy an anchor gate.
    """
    return {kw.arg for kw in call.keywords if kw.arg is not None}


def _matching_shape(call: ast.Call) -> tuple[_DataclassShape, set[str]] | None:
    """Return the first registered shape *call* matches (with its kwarg names), else None.

    ANCHOR + OVERLAP, both required:
      1. every name in ``shape.anchors`` appears among the literal kwarg names, AND
      2. at least ``shape.min_field_matches`` of ``shape.fields`` are matched.

    Gate 1 alone would flag a stray ``MagicMock(passed=True)`` on an unrelated object;
    gate 2 alone would flag ``MagicMock(summary=..., timed_out=...)`` with no anchor.
    Neither is sufficient by itself.

    Kwargs that are NOT fields neither block nor weaken the match — they are drift
    evidence surfaced in the message.  A subset rule (``kwargs <= fields``) would have
    missed all ten task-3980 sites, every one of which carried ``verify_skipped=``.

    Only the FIRST matching shape is returned: a call must never produce one violation
    per registered shape.
    """
    kwargs = _literal_kwarg_names(call)
    for shape in _DATACLASS_SHAPES:
        if not shape.anchors <= kwargs:
            continue
        if len(kwargs & shape.fields) < shape.min_field_matches:
            continue
        return shape, kwargs
    return None


def _find_dataclass_double_violations(
    tree: ast.AST, lines: list[str], filename: str
) -> list[Violation]:
    """Rule B: flag unspecced MagicMocks shaped like a registered dataclass, in ANY position.

    Deliberately POSITION-BLIND — a plain ``ast.walk`` over every ``ast.Call`` rather
    than Rule A's ``ast.Assign``/``ast.AnnAssign`` pipeline.  All ten sites behind task
    3980 were ``return MagicMock(...)``, which Rule A cannot see; the binding name is
    not consulted either, so Rule A's config-name gate does not apply here.

    Reuses ``_is_magicmock_call`` and ``_is_specced`` unchanged, so the two rules can
    never disagree about what "a MagicMock" or "specced" means.

    Violations carry ``call.col_offset`` (the ``MagicMock(`` token), so a node that
    trips both rules yields two deterministically-ordered entries rather than a collision.
    """
    violations: list[Violation] = []
    # Short-circuit before the walk: a debt-baseline file is grandfathered for
    # Rule B wholesale (Rule A still applies — it runs in a separate pass).
    if _is_debt_path(filename):
        return violations
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _is_magicmock_call(node):
            continue
        if _is_specced(node):
            continue
        match = _matching_shape(node)
        if match is None:
            continue
        shape, kwargs = match
        # Computed lazily — only after a shape match — so the upward line walk keeps
        # the cost profile it has under Rule A rather than running on every call node.
        if _is_exempted(lines, node.lineno, _RULE_B_CODE):
            continue
        violations.append(
            Violation(
                filename=filename,
                lineno=node.lineno,
                col_offset=node.col_offset,
                message=_dataclass_violation_msg(shape, kwargs),
            )
        )
    return violations


def _is_exempted(lines: list[str], lineno: int, code: str) -> bool:
    """Return True if the node at *lineno* (1-based) carries a valid ``code`` exemption.

    Walks upward from ``lineno - 1`` over blank lines to the nearest non-blank line.
    If that line matches ``code``'s exemption regex the node is exempt.
    Any intervening non-blank, non-matching line breaks the exemption.

    The *code* parameter keeps the two rules' suppressions strictly separate: a
    ``# noqa: bare-magicmock`` pragma does not exempt a ``bare-dataclass-double``
    violation, or vice versa.  Only the regex differs — the walk, the blank-line
    tolerance and the mandatory-non-empty-reason contract are shared verbatim.

    Inline trailing exemption NOT honored: only the nearest *preceding* non-blank line
    is inspected.  A ``# noqa: ...`` comment on the same line as the node (inline
    trailing) is intentionally ignored.  This is by design — see module-level docstring.
    """
    exempt_re = _EXEMPT_RES[code]
    # lineno is 1-based; convert to 0-based index of the line ABOVE the node.
    idx = lineno - 2  # the line immediately above
    while idx >= 0:
        line = lines[idx]
        stripped = line.strip()
        if stripped == '':
            idx -= 1
            continue
        # Nearest non-blank line found — must match the exemption regex.
        return bool(exempt_re.match(stripped))
    return False


def find_violations(source: str, filename: str) -> list[Violation]:
    """Parse *source* and return violations for bare MagicMock() config assignments.

    A violation is emitted for each ``<config_name> = MagicMock()`` call that:
      - has at least one ast.Name target whose id matches the config-name set,
      - calls MagicMock (by name or attribute) with no spec/spec_set,
      - is NOT preceded (on the nearest non-blank source line) by a valid exemption comment.

    For chained assignments (``mock = config = MagicMock()``), each ast.Name target
    is evaluated independently and may produce a separate Violation.

    SyntaxError in *source* → returns an empty list.

    Returned violations are sorted ascending by (lineno, col_offset) for
    deterministic source-order output (ast.walk yields BFS order, not source order).
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return []

    lines = source.splitlines()
    violations: list[Violation] = []

    for node in ast.walk(tree):
        # Normalise both ast.Assign (possibly multi-target) and ast.AnnAssign
        # (single annotated target) into a uniform (targets, value, lineno) triple
        # so the evaluation pipeline below can be written once.
        if isinstance(node, ast.Assign):
            # All ast.Name targets share the RHS value and the assignment lineno.
            # Non-Name targets (ast.Tuple for unpacking, ast.Attribute for
            # self.config, etc.) are intentional non-goals skipped by the
            # isinstance guard in the loop below.
            targets: list[ast.expr] = node.targets
            value = node.value
            assignment_lineno = node.lineno
        elif isinstance(node, ast.AnnAssign):
            if node.value is None:
                continue
            targets = [node.target]
            value = node.value
            assignment_lineno = node.lineno
        else:
            continue

        # Shared upfront checks: both branches reject non-MagicMock or specced calls
        # before iterating targets, so the (typically cheap) per-target name check
        # does not run for irrelevant RHS expressions.
        if not _is_magicmock_call(value):
            continue
        if _is_specced(value):  # type: ignore[arg-type]
            continue

        # _is_exempted is computed lazily — only on finding the first config-named
        # ast.Name target — because the exemption check (an upward line walk + regex)
        # is not free, and most assignments have no config-named targets.
        exempted: bool | None = None
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if not _is_config_name(target.id):
                continue
            if exempted is None:
                exempted = _is_exempted(lines, assignment_lineno, _RULE_A_CODE)
            if exempted:
                # All targets of this node share the same lineno and therefore
                # the same exemption status — no need to check further targets.
                break
            violations.append(
                Violation(
                    filename=filename,
                    lineno=assignment_lineno,
                    col_offset=target.col_offset,
                    message=_VIOLATION_MSG,
                )
            )

    # Rule B runs as an independent pass over the same tree and merges into the
    # same sorted output, so main()'s reporting and the exit-code contract are
    # untouched.  Rule A's loop above is not modified by the widening.
    violations.extend(_find_dataclass_double_violations(tree, lines, filename))

    return sorted(violations, key=lambda v: (v.lineno, v.col_offset))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Accepts file paths and/or directories.

    For directories, recursively scans for test_*.py and conftest.py files only.
    Prints violations to stdout in 'path:lineno:col: message' format (ruff-style).

    Explicit file paths are validated up front; a missing explicit path fails
    fast with exit code 2 before any scan work. Mid-scan OSErrors (e.g. a file
    yanked between rglob discovery and read) are accumulated and reported on
    stderr without discarding violations already collected.

    Returns 0 if clean, 1 if only violations were found, 2 on any fatal error
    (missing explicit path or transient read failure).
    """
    parser = argparse.ArgumentParser(
        description='Check for bare MagicMock() assignments to config-named variables in test files.'
    )
    parser.add_argument('paths', nargs='+', help='Files or directories to check')
    args = parser.parse_args(argv)

    # Phase 1: discovery + upfront validation of explicit paths.
    # rglob results are guaranteed to exist at discovery time, so only
    # non-directory (explicit) paths need the existence check.
    files_to_scan: list[Path] = []
    for path_str in args.paths:
        p = Path(path_str)
        if p.is_dir():
            files_to_scan.extend(sorted(set(p.rglob('test_*.py')) | set(p.rglob('conftest.py'))))
        else:
            if not p.exists():
                print(f'error: {p}: No such file or directory', file=sys.stderr)
                return 2
            files_to_scan.append(p)

    # Phase 2: scan. Accumulate per-file read errors without returning early,
    # so a transient OSError on one file never discards violations already
    # collected from earlier files.
    all_violations: list[Violation] = []
    read_errors: list[tuple[Path, Exception]] = []
    for file_path in files_to_scan:
        try:
            source = file_path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError) as exc:
            # UnicodeDecodeError is a ValueError subclass, not an OSError,
            # but a malformed file must be reported via the read_errors channel
            # rather than crashing with an unhandled traceback.
            read_errors.append((file_path, exc))
            continue

        violations = find_violations(source, str(file_path))
        all_violations.extend(violations)

    # Phase 3: reporting.
    # Sort across files for deterministic ruff-style (filename, lineno, col_offset) output.
    all_violations.sort(key=lambda v: (v.filename, v.lineno, v.col_offset))
    for v in all_violations:
        print(f'{v.filename}:{v.lineno}:{v.col_offset}: {v.message}')
    for file_path, exc in read_errors:
        print(f'error reading {file_path}: {exc}', file=sys.stderr)

    if read_errors:
        return 2
    return 1 if all_violations else 0


if __name__ == '__main__':
    sys.exit(main())
