#!/usr/bin/env python3
"""Lint checks: test-quality guards over test files.

This script carries THREE INDEPENDENT RULES.  Rules A and B are mock-spec
discipline and share the AST predicates ``_is_magicmock_call`` / ``_is_specced``;
Rule C is a wait-deadline rule and shares none of them.  What ALL THREE share is
only the exemption-comment contract (``_EXEMPT_TEMPLATE`` / ``_is_exempted``), the
per-file debt-budget machinery (``_debt_budget`` / ``_apply_debt_budget``), the
single ``ast.walk`` and the output format.  They have separate detection pipelines,
separate message vocabularies and separate ``# noqa`` codes.  Every statement in the
"Rule A" section below is scoped to Rule A and says nothing about Rule B or Rule C.

  Rule A — ``bare-magicmock`` (tasks 1339/1372)
      Config-NAMED bindings only, ``ast.Assign``/``ast.AnnAssign`` positions only.
      Remedies are pydantic-specific (``mock_orch_config``, ``pydantic_spec``).

  Rule B — ``bare-dataclass-double`` (task 4016)
      Registry-driven and POSITION-BLIND: any ``ast.Call`` anywhere — including
      ``return MagicMock(...)``, an argument, a comprehension body — whose literal
      kwargs match a registered stdlib-dataclass shape (``_DATACLASS_SHAPES``;
      ``VerifyResult`` today).  The binding name is never consulted.  Remedies are
      dataclass-specific (``_fake_verify_result``, ``MagicMock(spec=VerifyResult)``).
      Carries a shrink-only per-file debt BUDGET (``_DATACLASS_DOUBLE_DEBT``): a
      grandfathered file is silent while it carries at most its recorded number of
      sites and reports the overrun as soon as it carries more.

  Rule C — ``wall-clock-deadline`` (task 4246)
      NOT a mock-spec rule.  Position-blind over any ``ast.Call``: a load-bearing
      synchronisation point (a ``MergeRequest.result`` future or a ``gate*.wait()``
      barrier) awaited through a bare ``asyncio.wait_for`` instead of
      ``wait_responsive``, or carrying a raw numeric ``timeout=`` literal on either
      call shape.  Remedies are wait-specific (``wait_responsive(...)`` with a
      ``label=``, bound derived from ``MERGE_RESULT_TIMEOUT``).  Carries its own
      shrink-only per-file debt BUDGET (``_WALL_CLOCK_DEADLINE_DEBT``).

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

Task 4246 added Rule C under that same ruling, unchanged: the nine wiring edits, the
extra ``python3`` process and the second fleet-lint-coverage entry a sibling script
would have cost are all still there, and Rule C is per-``ast.Call`` exactly like
Rule B, so it folds into the SAME ``ast.walk`` for very nearly free rather than
paying for a second pass (a second full walk was measured at ~43% of total checker
runtime, 20.8s -> 30.7s over the seven scanned dirs).  Note what this ruling does NOT
say: it is not "put every future check here".  It applies because Rule C is another
AST pass over exactly the same file set with exactly the same stdlib-only budget.
A guard needing runtime import, pytest marks or a non-stdlib dependency does not
qualify and stays where it is — which is why
``orchestrator/tests/test_merge_speculation.py::TestTimeoutMarkCoverage`` was
deliberately left file-local by task 4246 rather than moved here.

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
``# noqa: bare-dataclass-double — <reason>``, or the file is on the debt baseline
and still within its recorded site budget.

Matching is ANCHOR + OVERLAP, deliberately NOT "kwargs are a subset of the fields":
every anchor must be present AND at least ``min_field_matches`` fields must match.
A kwarg that is not a field is *drift evidence* named in the message, never an
exemption — a bare MagicMock accepts any keyword silently, so an unrecognised one is
the strongest signal the double has drifted from the type it impersonates.  A subset
rule would have missed all ten sites behind task 3980, every one of which passed
``verify_skipped=`` (a MergeOutcome field:
orchestrator/src/orchestrator/merge_types.py::MergeOutcome.verify_skipped).

Why this rule exists: Rule A provably cannot see the shape, for three independent
reasons, any one of them fatal — it inspects only ``ast.Assign``/``ast.AnnAssign``
while all ten task-3980 sites were ``return MagicMock(...)``; ``_is_config_name``
matches only config/cfg/*_config/*_cfg targets; and its remedies read pydantic
``model_fields`` while ``VerifyResult`` is a stdlib dataclass.

Preferred alternatives named in the rejection message:
  • ``_fake_verify_result(...)``
    (orchestrator/tests/test_merge_queue_concurrent_verify.py::_fake_verify_result)
  • ``MagicMock(spec=VerifyResult)`` seeded from ``dataclasses.fields(VerifyResult)``

Origin: task 3477 (built the factory), task 3980 (migrated ten sites and added a
file-local guard), task 4016 (this shared, repo-wide guard).

---------------------------------------------------------------------------
Rule C — ``wall-clock-deadline``
---------------------------------------------------------------------------

Rule: a LOAD-BEARING synchronisation point — a ``MergeRequest.result`` future
(``req.result``) or an ``asyncio.Event`` gate barrier (``gate*.wait()``) — awaited
with a wall-clock deadline.  Two independent offence kinds, so ONE call can produce
TWO violations:

  1. the target is awaited through a bare ``asyncio.wait_for(...)`` rather than
     ``wait_responsive(...)``, so its deadline is charged in WALL CLOCK; and
  2. the call carries a RAW numeric ``timeout=`` literal — on EITHER call shape.
     ``wait_responsive`` takes a ``timeout`` keyword too, so a migrated site can
     have moved the accounting while keeping a hand-written number.

Suppressed by ``# noqa: wall-clock-deadline — <reason>`` on the preceding non-blank
line, or by the file's ``_WALL_CLOCK_DEADLINE_DEBT`` budget.

Why this shape and not a list: task 3980's measured failures were genuine asyncio
deadline expiries on tests whose logic had ALREADY completed — the log tail reads
``verify end (passed=True)`` beside a heartbeat of ``oldest age=46s``.  Widening the
numbers only moves the threshold; charging the budget in loop-responsive time removes
the dependence.  Task 2376's earlier sweep expressed its policy as "literals up to 15"
and the lone ``timeout=25.0`` sat just above it, surviving as one of the three measured
failures — a policy expressed as a list, or as a threshold, cannot catch what is
outside it.  Task 3980's own amendment pass then deleted a hand-maintained five-class
frozenset for the same reason.  So selection here is by call SHAPE alone: there is no
class list, no name table and no budget threshold deciding which sites are scanned.

What is deliberately NOT load-bearing: a wait on a bare ``ast.Name`` target, i.e. the
``asyncio.wait_for(worker_task, ...)`` teardown join in ``_stop_worker``.  It sits
inside ``contextlib.suppress(Exception)``, asserts nothing and swallows its own
TimeoutError, so it cannot manufacture the flake this rule exists to prevent.  That
exclusion is STRUCTURAL — the Name-vs-Attribute/Call distinction — precisely so it is
not one more hand-maintained name list.

Scope paths are NOT carried in the message (the file-local guard printed
``Class::method``): this script's output contract is ruff-style
``path:lineno:col: message``, which already locates the site.

Preferred alternatives named in the rejection message:
  • ``wait_responsive(aw, *, timeout=MERGE_RESULT_TIMEOUT, label, ...)``
    (orchestrator/tests/_orch_helpers.py::wait_responsive)
  • a bound derived from ``MERGE_RESULT_TIMEOUT``, never a written number

Origin: task 2376 (the sweep whose threshold left the gap), task 3980 (migrated the
sites and added a file-local guard), task 4246 (this shared, repo-wide guard).

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
    (orchestrator/src/orchestrator/merge_types.py::MergeOutcome.verify_skipped) that
    VerifyResult does not have, so a subset rule would have missed precisely the
    defect this rule exists to catch.
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
# VerifyResult's field list mirrors orchestrator/src/orchestrator/verify.py::VerifyResult.
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
# exempt another: the remedies are unrelated (mock_orch_config/pydantic_spec vs
# _fake_verify_result/spec=VerifyResult vs wait_responsive/MERGE_RESULT_TIMEOUT), so a
# pragma for one is not informed consent for another.  All three are built from the
# same template, so the em-dash/ASCII-hyphen and mandatory-reason contract is
# identical across rules and an author learns it once.
_EXEMPT_TEMPLATE = r'#\s*noqa:\s*{code}\s*[—\-]+\s*\S.*'

_RULE_A_CODE = 'bare-magicmock'
_RULE_B_CODE = 'bare-dataclass-double'
_RULE_C_CODE = 'wall-clock-deadline'

# Kept at its historical value so Rule A's behaviour is bit-identical.
_EXEMPT_RE = re.compile(_EXEMPT_TEMPLATE.format(code=re.escape(_RULE_A_CODE)))

_EXEMPT_RES: dict[str, re.Pattern[str]] = {
    _RULE_A_CODE: _EXEMPT_RE,
    _RULE_B_CODE: re.compile(_EXEMPT_TEMPLATE.format(code=re.escape(_RULE_B_CODE))),
    _RULE_C_CODE: re.compile(_EXEMPT_TEMPLATE.format(code=re.escape(_RULE_C_CODE))),
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
# Rule B debt baseline — SHRINK-ONLY, and CHECKED.
#
# path → the number of pre-existing dataclass-double sites that file carried when
# Rule B landed (AST census over all seven scanned tests/ directories, task 4016;
# 95 sites across 11 files).  Shipping the rule hot with no transition would have
# turned orchestrator/tests' lint_command red on day one and stalled the merge lane
# repo-wide, so these are grandfathered — Rule B ONLY; Rule A still applies in full
# to every file here.
#
# The count is a BUDGET, not a comment.  A debt file is silent while it carries at
# most its recorded number of sites and reports the overrun the moment it carries
# more, so "shrink-only" is enforced on the same hot path the rule itself runs on
# rather than trusted.  This matters most for
# orchestrator/tests/test_merge_queue.py: 63 sites in an actively-developed hub,
# where a wholesale grandfather would have made a brand-new bare double added
# tomorrow invisible to the gate.
#
# DO NOT ADD ENTRIES, AND DO NOT RAISE A NUMBER.  Both may only shrink, as files
# are migrated onto _fake_verify_result / MagicMock(spec=VerifyResult); that
# migration is filed as a follow-up task.  A NEW file with a bare dataclass double
# must fail the gate — that is the entire reason the baseline is opt-OUT rather
# than opt-in.
#
# orchestrator/tests/test_merge_speculation.py is deliberately NOT here: task
# 3980 just cleaned that module, and even a budgeted entry would let a new double
# land there silently. Its one deliberate double carries a per-site pragma instead.
_DATACLASS_DOUBLE_DEBT: dict[str, int] = {
    'orchestrator/tests/test_merge_queue.py': 63,
    'orchestrator/tests/test_concurrent_verify_boundary.py': 9,
    'orchestrator/tests/test_merge_queue_permit_conservation.py': 7,
    'orchestrator/tests/test_merge_queue_resolve_release.py': 7,
    'orchestrator/tests/test_merge_queue_request_liveness.py': 3,
    'orchestrator/tests/test_coalesce_integration_gate.py': 1,
    'orchestrator/tests/test_merge_item_union.py': 1,
    'orchestrator/tests/test_merge_queue_equivalence.py': 1,
    'orchestrator/tests/test_merge_queue_lifecycle_registry.py': 1,
    'orchestrator/tests/test_merge_queue_metrics.py': 1,
    'orchestrator/tests/test_merge_queue_single_writer_asserts.py': 1,
}


def _debt_budget(filename: str, debt: dict[str, int]) -> int | None:
    """Return *filename*'s budget in the *debt* mapping, or None if it is not listed.

    Parameterised over the mapping rather than copied per rule (task 4246): the
    trailing-component matching below is what makes repo-relative CLI paths and
    absolute pytest paths reach the same verdict, and defining it once is what stops
    the baselines drifting apart in their matching semantics.

    Compares TRAILING PATH COMPONENTS, not substrings: ``a/b/c.py`` matches an
    entry ``b/c.py`` because its last two components are exactly ``b`` and ``c.py``.
    Component-awareness is what stops ``orchestrator/tests/not_test_merge_queue.py``
    — which merely contains a debt filename as a substring — from being grandfathered.

    Trailing-component matching (rather than an exact string compare) is required
    because the nine call sites pass repo-relative paths while pytest passes
    absolutes; both must reach the same verdict.

    Returns None (not 0) for a non-debt file so callers can distinguish "no budget
    recorded — report every site" from "budget of zero".
    """
    parts = PurePosixPath(filename.replace('\\', '/')).parts
    for entry, allowed in debt.items():
        entry_parts = PurePosixPath(entry).parts
        if len(parts) >= len(entry_parts) and parts[-len(entry_parts) :] == entry_parts:
            return allowed
    return None


def _debt_overrun_msg(budget: int, found: int) -> str:
    """Build the message for a debt file that has grown past its recorded budget."""
    return (
        'this file is on the SHRINK-ONLY bare-dataclass-double debt baseline with a'
        f' recorded budget of {budget} site(s), but {found} were found —'
        f' {found - budget} over budget. The baseline may only shrink.'
        ' The reported sites are simply the LAST in source order: the anchor is'
        ' positional and is NOT a claim that these exact sites are the new ones.'
        ' Fix by migrating a site in this file onto _fake_verify_result(...) or'
        ' MagicMock(spec=VerifyResult) seeded from dataclasses.fields, or by adding'
        ' # noqa: bare-dataclass-double — <reason> above a deliberate one.'
        ' Do NOT raise the recorded budget in check_bare_magicmock_config.py.'
    )


def _apply_debt_budget(
    found: list[Violation], budget: int | None, overrun_msg: str | None = None
) -> list[Violation]:
    """Filter one rule's violations for a debt file down to just its budget overrun.

    *overrun_msg* is prebuilt by the caller (task 4246) rather than derived here,
    because each rule's overrun carries its OWN remedy: telling a Rule C overrun to
    migrate onto ``_fake_verify_result`` would send the reader down a dead end.  It
    is only consulted on the over-budget branch, so callers may pass None when they
    have no debt entry.

    - Not a debt file (*budget* is None) → every violation is reported unchanged.
    - At or under budget → silence: this is the grandfathering the baseline exists for.
    - Over budget → report exactly ``found - budget`` violations, so the noise is
      proportional to the overrun rather than dumping all 63 Rule B sites (or all
      317 Rule C ones) of test_merge_queue.py on someone who added one.

    The reported sites are the last in source order.  That choice is deterministic
    rather than diagnostic — the checker cannot know which site is new — and the
    message says so explicitly.
    """
    if budget is None:
        return found
    if len(found) <= budget:
        return []
    ordered = sorted(found, key=lambda v: (v.lineno, v.col_offset))
    message = overrun_msg if overrun_msg is not None else _debt_overrun_msg(budget, len(ordered))
    return [v._replace(message=message) for v in ordered[budget:]]


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


def _dataclass_double_violation(
    call: ast.Call, lines: list[str], filename: str
) -> Violation | None:
    """Rule B, evaluated for ONE ``ast.Call``: return a Violation, or None if clean.

    Deliberately POSITION-BLIND — the caller hands this every ``ast.Call`` in the tree
    rather than only Rule A's ``ast.Assign``/``ast.AnnAssign`` values.  All ten sites
    behind task 3980 were ``return MagicMock(...)``, which Rule A cannot see; the
    binding name is not consulted either, so Rule A's config-name gate does not apply.

    Reuses ``_is_magicmock_call`` and ``_is_specced`` unchanged, so the two rules can
    never disagree about what "a MagicMock" or "specced" means.

    Violations carry ``call.col_offset`` (the ``MagicMock(`` token), so a node that
    trips both rules yields two deterministically-ordered entries rather than a collision.

    Per-NODE rather than per-tree so ``find_violations`` can evaluate both rules in a
    SINGLE ``ast.walk``.  A second full walk cost ~43% of total checker runtime
    (measured over the seven scanned tests/ dirs: 20.8s → 30.7s), paid on every
    merge-queue verify across all nine call sites; folded into Rule A's existing walk
    it is nearly free.
    """
    if not _is_magicmock_call(call):
        return None
    if _is_specced(call):
        return None
    match = _matching_shape(call)
    if match is None:
        return None
    shape, kwargs = match
    # Computed lazily — only after a shape match — so the upward line walk keeps
    # the cost profile it has under Rule A rather than running on every call node.
    if _is_exempted(lines, call.lineno, _RULE_B_CODE):
        return None
    return Violation(
        filename=filename,
        lineno=call.lineno,
        col_offset=call.col_offset,
        message=_dataclass_violation_msg(shape, kwargs),
    )


# ---------------------------------------------------------------------------
# Rule C debt baseline — SHRINK-ONLY, and CHECKED.
#
# path → the number of pre-existing wall-clock-deadline VIOLATIONS that file
# carried when Rule C landed (AST census over all seven scanned tests/ directories,
# task 4246; 618 violations across 20 files, every one under orchestrator/tests/).
#
# The number counts VIOLATIONS, not SITES — unlike _DATACLASS_DOUBLE_DEBT above.
# One call can produce two: `asyncio.wait_for(req.result, timeout=25.0)` is
# simultaneously the wrong routing (bare-wait_for) and a written number
# (raw-literal).  The day-one split was 333 bare-wait_for + 285 raw-literal.
#
# Shipping the rule hot with no transition would have turned orchestrator/tests'
# lint_command red on day one and stalled the merge lane repo-wide — the identical
# situation Rule B faced at 95 sites/11 files, so these are grandfathered.  Rule C
# ONLY: Rules A and B still apply in full to every file here.
#
# The count is a BUDGET, not a comment.  A debt file is silent while it carries at
# most its recorded number and reports the overrun the moment it carries more, so
# "shrink-only" is enforced on the same hot path the rule itself runs on rather than
# trusted.  This matters most for orchestrator/tests/test_merge_queue.py: 317
# violations in an actively-developed hub, where a wholesale grandfather would have
# made a brand-new wall-clock wait added tomorrow invisible to the gate.
#
# DO NOT ADD ENTRIES, AND DO NOT RAISE A NUMBER.  Both may only shrink, as files are
# migrated onto wait_responsive(...) with bounds derived from MERGE_RESULT_TIMEOUT.
# A NEW file with a load-bearing wall-clock wait must FAIL the gate — that is the
# entire reason the baseline is opt-OUT rather than opt-in.  An opt-in list would
# exempt precisely the brand-new file this rule exists to catch, and would make
# "which files are covered" a hand-maintained list: the exact failure mode task
# 3980's amendment pass deleted a five-class frozenset to escape.
#
# orchestrator/tests/test_merge_speculation.py is deliberately NOT here.  It measures
# ZERO because task 3980 already migrated it, and even a budgeted entry of zero would
# be a blanket suppression letting a regression land there silently — which is what
# 3980 spent a task removing, and what makes it safe for task 4246 to delete that
# module's file-local copy of this guard.
_WALL_CLOCK_DEADLINE_DEBT: dict[str, int] = {
    'orchestrator/tests/test_merge_queue.py': 317,
    'orchestrator/tests/test_merge_queue_concurrent_verify.py': 90,
    'orchestrator/tests/test_concurrent_verify_boundary.py': 44,
    'orchestrator/tests/test_merge_queue_permit_conservation.py': 27,
    'orchestrator/tests/test_merge_queue_lifecycle_registry.py': 26,
    'orchestrator/tests/test_merge_queue_resolve_release.py': 25,
    'orchestrator/tests/test_merge_queue_invariant_integration_gate.py': 18,
    'orchestrator/tests/test_merge_queue_equivalence.py': 12,
    'orchestrator/tests/test_merge_queue_restart_hook.py': 12,
    'orchestrator/tests/test_merge_queue_request_liveness.py': 10,
    'orchestrator/tests/test_coalesce_integration_gate.py': 8,
    'orchestrator/tests/test_merge_queue_coalesce.py': 8,
    'orchestrator/tests/test_merge_queue_persistent_worktree.py': 6,
    'orchestrator/tests/test_merge_queue_single_writer_asserts.py': 4,
    'orchestrator/tests/test_merge_guard_pipeline.py': 2,
    'orchestrator/tests/test_merge_queue_supervisor.py': 2,
    'orchestrator/tests/test_merge_queue_verifier_raw_cancel.py': 2,
    'orchestrator/tests/test_merge_queue_warm_cold_shadow.py': 2,
    'orchestrator/tests/test_merge_worktree_lifecycle_integration_gate.py': 2,
    'orchestrator/tests/test_merge_queue_dispatch_fill_redispatch.py': 1,
}


def _wall_clock_overrun_msg(budget: int, found: int) -> str:
    """Build the message for a Rule C debt file that has grown past its budget.

    Carries Rule C's OWN remedy, never Rule B's: a wall-clock deadline is not fixed
    by speccing a double, so offering ``_fake_verify_result`` here would send the
    reader down a dead end.
    """
    return (
        'this file is on the SHRINK-ONLY wall-clock-deadline debt baseline with a'
        f' recorded budget of {budget} violation(s), but {found} were found —'
        f' {found - budget} over budget. The baseline may only shrink.'
        ' The reported sites are simply the LAST in source order: the anchor is'
        ' positional and is NOT a claim that these exact sites are the new ones.'
        ' Fix by migrating a load-bearing wait in this file onto wait_responsive(...)'
        ' with a descriptive label=, or by deriving its bound from'
        ' MERGE_RESULT_TIMEOUT instead of writing a number, or by adding'
        ' # noqa: wall-clock-deadline — <reason> above a deliberate one.'
        ' Do NOT raise the recorded budget in check_bare_magicmock_config.py.'
    )

# ---------------------------------------------------------------------------
# Rule C — wall-clock-deadline
# ---------------------------------------------------------------------------


def _load_bearing_wait_target(node: ast.expr) -> str | None:
    """Describe *node* if it is a load-bearing synchronisation point, else None.

    Exactly two shapes are load-bearing, and both gate a hard assertion
    downstream:

      * ``req_a.result`` — a ``MergeRequest.result`` future.  Its resolution IS
        the event the test is waiting for; a deadline here fails a test whose
        merge pipeline completed correctly.
      * ``gate_a_entered.wait()`` — an ``asyncio.Event`` barrier.  Already
        event-driven; only its deadline is wall-clock.

    Deliberately NOT load-bearing, and therefore excluded: the
    ``await asyncio.wait_for(worker_task, timeout=join_timeout)`` join in
    ``_stop_worker``.  It targets a bare ``Name`` (the worker Task), sits inside
    ``contextlib.suppress(Exception)``, asserts nothing, and swallows its own
    TimeoutError — so it cannot manufacture the flake task 3980 fixed, and
    stretching it would only slow teardown down.  The Name-vs-Attribute/Call
    distinction is what makes that exclusion STRUCTURAL rather than a
    hand-maintained name list, which is what lets this rule scan every scope of
    every file the nine call sites reach.

    Ported unchanged in behaviour from the file-local guard this rule replaced
    (task 3980, orchestrator/tests/test_merge_speculation.py).
    """
    if isinstance(node, ast.Attribute) and node.attr == 'result':
        return f'{ast.unparse(node)} (MergeRequest.result future)'
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'wait'
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id.startswith('gate')
    ):
        return f'{ast.unparse(node)} (asyncio.Event gate barrier)'
    return None

# Rule C's two offence kinds.  They are INDEPENDENT — one call can trip both
# (a bare asyncio.wait_for that also writes a number) or exactly one (a migrated
# wait_responsive that kept its literal; a bare wait_for whose bound is derived).
_WALL_CLOCK_BARE_WAIT_FOR = 'bare-wait_for'
_WALL_CLOCK_RAW_LITERAL = 'raw-literal'

# Shared by both kinds: WHY a wall-clock deadline on a load-bearing sync point is a
# defect rather than a style preference.  Task 3980's three measured failures were
# all genuine deadline expiries on tests that had already passed.
_WALL_CLOCK_CONSEQUENCE = (
    ' A deadline expiry on a load-bearing synchronisation point fails a test whose'
    ' merge pipeline completed correctly, purely because the worker was descheduled.'
)

_WALL_CLOCK_SUPPRESS = (
    ' To suppress: add # noqa: wall-clock-deadline — <reason> on the preceding'
    ' non-blank line.'
)


def _wall_clock_violation_msg(kind: str, target: str) -> str:
    """Build Rule C's rejection message for one offence *kind* on *target*.

    Deliberately shares NO vocabulary with ``_VIOLATION_MSG`` or
    ``_dataclass_violation_msg``: Rule A's remedies read pydantic ``model_fields``
    and Rule B's spec a stdlib dataclass — neither has anything to say about a
    wall-clock deadline on a future, so offering either here would send the reader
    down a dead end.  The two KINDS also keep separate remedies from each other for
    the same reason: routing through ``wait_responsive`` and deriving the bound from
    ``MERGE_RESULT_TIMEOUT`` are independent fixes, and a site can need one, the
    other, or both.
    """
    if kind == _WALL_CLOCK_BARE_WAIT_FOR:
        return (
            f'load-bearing wait on {target} is routed through a bare asyncio.wait_for,'
            ' so its deadline is charged in WALL CLOCK.'
            + _WALL_CLOCK_CONSEQUENCE
            + ' Route it through wait_responsive(...) with a descriptive label='
            ' (orchestrator/tests/_orch_helpers.py::wait_responsive), which charges its'
            ' budget in loop-responsive time and still reports a genuine hang red.'
            + _WALL_CLOCK_SUPPRESS
        )
    return (
        f'load-bearing wait on {target} carries a RAW wall-clock literal timeout=.'
        + _WALL_CLOCK_CONSEQUENCE
        + ' Derive the bound from MERGE_RESULT_TIMEOUT instead of writing a number:'
        ' a written literal is a threshold, and task 2376 measured that a policy'
        ' expressed as "literals up to N" cannot catch the one just above N.'
        + _WALL_CLOCK_SUPPRESS
    )


def _wall_clock_deadline_violations(
    call: ast.Call, lines: list[str], filename: str
) -> list[Violation]:
    """Rule C, evaluated for ONE ``ast.Call``: return 0, 1 or 2 Violations.

    Returns a LIST, not an Optional, because the two offence kinds are independent
    and a single call can trip both — ``asyncio.wait_for(req.result, timeout=25.0)``
    is simultaneously the wrong routing and a written number.

    Gating order is cheapest-first, and deliberately so:
      1. the call has at least one positional argument (no ``args[0]`` to inspect
         otherwise — and a bare ``asyncio.wait_for()`` must be skipped, not crash);
      2. the func is ``asyncio.wait_for`` or ``wait_responsive``;
      3. ``_load_bearing_wait_target`` recognises ``args[0]``.
    Only then is the exemption line-walk run (see ``_wall_clock_deadline_violations``'s
    call to ``_is_exempted``), so the upward walk keeps Rule B's cost profile rather
    than running on every ``ast.Call`` in the tree.

    Position-blind by construction: the caller hands this every ``ast.Call``, so a
    wait in a ``return``, an argument, a comprehension body or at module level is
    covered identically.  There is no class list and no scope filter — see the Rule C
    section of the module docstring for why a list, or a threshold, is not a sound key.

    Per-NODE rather than per-tree so ``find_violations`` can evaluate all three rules
    in a SINGLE ``ast.walk``; a second full walk was measured at ~43% of total checker
    runtime (20.8s -> 30.7s over the seven scanned dirs), paid on every merge-queue
    verify across all nine call sites.

    Violations carry ``call.col_offset``, so a node tripping several rules yields
    deterministically-ordered entries under ``find_violations``' final sort.  Both
    kinds from one call share a position; Python's stable sort keeps them in the
    fixed order they are appended here.

    Must never raise: a crash would fail every caller's lint over an unrelated edit.
    """
    if not call.args:
        return []

    func = call.func
    is_bare_wait_for = (
        isinstance(func, ast.Attribute)
        and func.attr == 'wait_for'
        and isinstance(func.value, ast.Name)
        and func.value.id == 'asyncio'
    )
    is_responsive = isinstance(func, ast.Name) and func.id == 'wait_responsive'
    if not (is_bare_wait_for or is_responsive):
        return []

    target = _load_bearing_wait_target(call.args[0])
    if target is None:
        return []

    # Computed lazily — only after the func shape AND the load-bearing target have
    # both matched — so the upward line walk keeps Rule B's cost profile rather than
    # running on every ast.Call in the tree.  ONE check per SITE suppresses BOTH
    # offence kinds: a pragma is consent for the site, not for one half of it.
    if _is_exempted(lines, call.lineno, _RULE_C_CODE):
        return []

    kinds: list[str] = []
    if is_bare_wait_for:
        kinds.append(_WALL_CLOCK_BARE_WAIT_FOR)

    # A raw numeric literal is an offence on EITHER call shape: wait_responsive
    # also takes a ``timeout`` keyword, so a migrated site can have moved the
    # accounting into loop-responsive time while keeping a hand-written number.
    #
    # ``bool`` is excluded explicitly because it is an int SUBCLASS — without the
    # guard, ``timeout=True`` would be reported as a wall-clock number.
    timeout_kw = next((kw for kw in call.keywords if kw.arg == 'timeout'), None)
    if (
        timeout_kw is not None
        and isinstance(timeout_kw.value, ast.Constant)
        and isinstance(timeout_kw.value.value, (int, float))
        and not isinstance(timeout_kw.value.value, bool)
    ):
        kinds.append(_WALL_CLOCK_RAW_LITERAL)

    return [
        Violation(
            filename=filename,
            lineno=call.lineno,
            col_offset=call.col_offset,
            message=_wall_clock_violation_msg(kind, target),
        )
        for kind in kinds
    ]

def _is_exempted(lines: list[str], lineno: int, code: str) -> bool:
    """Return True if the node at *lineno* (1-based) carries a valid ``code`` exemption.

    Walks upward from ``lineno - 1`` over blank lines to the nearest non-blank line.
    If that line matches ``code``'s exemption regex the node is exempt.
    Any intervening non-blank, non-matching line breaks the exemption.

    The *code* parameter keeps all three rules' suppressions strictly separate: a
    ``# noqa: bare-magicmock`` pragma does not exempt a ``bare-dataclass-double`` or
    ``wall-clock-deadline`` violation, in any direction.  Only the regex differs —
    the walk, the blank-line tolerance and the mandatory-non-empty-reason contract
    are shared verbatim.

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
    """Parse *source* and return every violation of ALL THREE rules.

    Rule A — ``bare-magicmock``: each ``<config_name> = MagicMock()`` that has at
    least one ast.Name target whose id matches the config-name set, calls MagicMock
    (by name or attribute) with no spec/spec_set, and is NOT preceded on the nearest
    non-blank source line by a valid exemption comment.  For chained assignments
    (``mock = config = MagicMock()``) each ast.Name target is evaluated independently
    and may produce a separate Violation.

    Rule B — ``bare-dataclass-double``: each unspecced ``MagicMock(...)`` in ANY
    position whose literal kwargs match a registered ``_DATACLASS_SHAPES`` entry.

    Rule C — ``wall-clock-deadline``: each load-bearing wait (a
    ``MergeRequest.result`` future or a ``gate*.wait()`` barrier) routed through a
    bare ``asyncio.wait_for``, and/or carrying a raw numeric ``timeout=`` literal.
    One call can produce TWO violations — the kinds are independent.

    Rules B and C additionally honour their own SHRINK-ONLY per-file debt budgets
    (``_DATACLASS_DOUBLE_DEBT`` / ``_WALL_CLOCK_DEADLINE_DEBT``), applied to the
    collected COUNT after the walk.  Rule A has no baseline and applies in full to
    every file.

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

    # Rule B violations are collected separately so a debt file's budget can be
    # applied to the COUNT after the walk (see _apply_debt_budget).  Looked up once
    # per file, not per node.
    dataclass_doubles: list[Violation] = []
    debt_budget = _debt_budget(filename, _DATACLASS_DOUBLE_DEBT)

    # Rule C violations are collected separately for the same reason, and against
    # their own independent baseline (task 4246).  Both budgets are looked up ONCE
    # per file, not per node.
    wall_clock: list[Violation] = []
    wall_clock_budget = _debt_budget(filename, _WALL_CLOCK_DEADLINE_DEBT)

    # ONE walk, ALL THREE rules.  Rule B was originally a second full ast.walk over the
    # same tree, which cost ~43% of total checker runtime (20.8s → 30.7s over the seven
    # scanned tests/ dirs) — a cost paid on every merge-queue verify across all nine
    # call sites.  Rule A's walk already visits every node and simply skips non-Assign
    # ones, so folding Rule B's and Rule C's per-Call handling in here makes them nearly
    # free.  Output is unchanged: the final sort by (lineno, col_offset) still
    # normalises ordering.
    for node in ast.walk(tree):
        # ---- Rule B: bare-dataclass-double, position-blind over every ast.Call ----
        # An ast.Call is never an ast.Assign/ast.AnnAssign, so this branch and Rule A's
        # below are mutually exclusive and the `continue` cannot skip Rule A work.
        if isinstance(node, ast.Call):
            double = _dataclass_double_violation(node, lines, filename)
            if double is not None:
                dataclass_doubles.append(double)
            # Rule C shares this branch rather than adding its own `continue`:
            # a second early exit here would shadow Rule B for any node Rule C
            # matched first.  Both run, then the single existing continue fires.
            wall_clock.extend(_wall_clock_deadline_violations(node, lines, filename))
            continue

        # ---- Rule A: bare-magicmock, ast.Assign/ast.AnnAssign only ----
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

    # Each rule's per-file debt budget is applied to its own collected COUNT, not to
    # individual sites: a grandfathered file stays silent while it does not grow, and
    # reports exactly its overrun once it does.  The overrun MESSAGE is built by the
    # caller because each rule's remedy differs — Rule B's is _fake_verify_result,
    # Rule C's is wait_responsive / MERGE_RESULT_TIMEOUT.
    violations.extend(
        _apply_debt_budget(
            dataclass_doubles,
            debt_budget,
            None if debt_budget is None else _debt_overrun_msg(debt_budget, len(dataclass_doubles)),
        )
    )
    violations.extend(
        _apply_debt_budget(
            wall_clock,
            wall_clock_budget,
            None
            if wall_clock_budget is None
            else _wall_clock_overrun_msg(wall_clock_budget, len(wall_clock)),
        )
    )

    return sorted(violations, key=lambda v: (v.lineno, v.col_offset))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Accepts file paths and/or directories.

    Runs all three rules — ``bare-magicmock``, ``bare-dataclass-double`` and
    ``wall-clock-deadline`` — in a single AST pass per file.

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
        description=(
            'Test-quality lint checks over test files: bare MagicMock() assigned to '
            'config-named variables (bare-magicmock), unspecced MagicMocks shaped like '
            'a registered dataclass (bare-dataclass-double), and load-bearing waits '
            'carrying a wall-clock deadline (wall-clock-deadline).'
        )
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
