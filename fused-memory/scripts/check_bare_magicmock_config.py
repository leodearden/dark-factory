#!/usr/bin/env python3
"""Lint checks: test-quality guards over test files.

This script carries THREE INDEPENDENT RULES.  Rules A and B are mock-spec
discipline and share the AST predicates ``_is_magicmock_call`` / ``_is_specced``;
Rule C is a wait-deadline rule and shares none of them.  What ALL THREE share is
only the exemption-comment contract (``_lint_cli.py::is_exempted``), the
single ``ast.walk`` and the output format.  The per-file debt-budget machinery
(``_debt_entry`` / ``_apply_debt_budget``) is Rule C's alone: Rules A and B are
hot for every scanned file.  They have separate detection pipelines,
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
      dataclass-specific (``fake_verify_result``, ``MagicMock(spec=VerifyResult)``).
      Hot for every scanned file: the transitional per-file debt baseline was
      retired once its 95 grandfathered sites were migrated (task 4354).

  Rule C — ``wall-clock-deadline`` (tasks 4246, 7800)
      NOT a mock-spec rule.  Position-blind over any ``ast.Call``: a load-bearing
      synchronisation point (a ``MergeRequest.result`` future or a zero-argument
      ``.wait()`` barrier) awaited through a bare ``asyncio.wait_for`` instead of
      ``wait_responsive``, or carrying a raw numeric literal bound (``timeout=``,
      ``max_wall_s=`` or wait_for's positional timeout).  Also any async ``while``
      poll that gives up against a wall-clock read, and any ``asyncio.timeout``
      block bounding a poll or a load-bearing await.  Remedies are wait-specific
      (``wait_responsive(...)`` or ``poll_responsive(...)`` with a ``label=``,
      bound derived from ``MERGE_RESULT_TIMEOUT``).  Carries its own shrink-only
      per-file debt BUDGET (``_WALL_CLOCK_DEADLINE_DEBT``).

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
would have cost are all still there, and Rule C is per-NODE like Rule B (it
dispatches on ``ast.Call``, ``ast.While`` and ``ast.AsyncWith``; task 7800 added the
latter two), so it folds into the SAME ``ast.walk`` for very nearly free rather than
paying for a second pass (a second full walk was measured at ~43% of total checker
runtime, 20.8s -> 30.7s over the seven scanned dirs).  Note what this ruling does NOT
say: it is not "put every future check here".  It applies because Rule C is another
AST pass over exactly the same file set with exactly the same stdlib-only budget.
A guard needing runtime import, pytest marks or a non-stdlib dependency does not
qualify and stays where it is — which is why the runtime-mark timeout-coverage
guard, ``orchestrator/tests/test_wait_budget_coverage_guard.py``, is a pytest
module rather than a rule here.

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
``# noqa: bare-dataclass-double — <reason>``.  That per-site pragma is the ONLY
suppression: Rule B carries no per-file baseline, so a bare double is never
grandfathered by the file it lives in.

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
  • ``fake_verify_result(...)``
    (orchestrator/tests/_verify_result_doubles.py::fake_verify_result)
  • ``MagicMock(spec=VerifyResult)`` seeded from ``dataclasses.fields(VerifyResult)``

Origin: task 3477 (built the factory), task 3980 (migrated ten sites and added a
file-local guard), task 4016 (this shared, repo-wide guard).

---------------------------------------------------------------------------
Rule C — ``wall-clock-deadline``
---------------------------------------------------------------------------

Rule: a LOAD-BEARING synchronisation point — a ``MergeRequest.result`` future
(``req.result``) or a zero-argument ``.wait()`` barrier (``done.wait()``,
``self._entered.wait()``) — awaited with a wall-clock deadline.  Two independent offence kinds, so ONE call can produce
TWO violations:

  1. the target is awaited through a bare ``asyncio.wait_for(...)`` rather than
     ``wait_responsive(...)``, so its deadline is charged in WALL CLOCK; and
  2. the call carries a RAW numeric literal bound — a ``timeout=`` or
     ``max_wall_s=`` keyword on ANY wait call shape, or ``asyncio.wait_for``'s
     positional timeout.  ``wait_responsive`` and ``poll_responsive`` take both
     keywords too, so a migrated site can have moved the accounting while keeping a
     hand-written number.  Only this kind applies to them, and every
     ``poll_responsive`` predicate is load-bearing: it IS the poll's
     synchronisation point.  Several literal bounds on one call are still ONE
     violation.

The same defect is also written as a POLL, so a further leg covers it:

  3. CLOCK-DEADLINE POLL (task 7800): an async ``while`` loop that gives up against a
     wall-clock read — a ``Compare`` in its test or its own body with a DIRECT
     operand that is a zero-argument ``.time()``, ``.monotonic()`` or
     ``.perf_counter()`` call, on any receiver, or a remaining time
     (``deadline - loop.time() > 0``), which is always measured against a deadline.
     One violation, at the ``while``.
     Its bound is arithmetic in another statement, so it has no raw-literal kind.
     A read held in a variable (``remaining = deadline - loop.time()``, then
     ``if remaining <= 0``) is not seen yet: ticket tkt_0RWM0CNQ4KSK1K2Y3TPDEAR3M1.
     Two structural discriminators keep it to polls.  ANOTHER-EXIT admits an
     elapsed-time operand (``loop.time() - t0 < timeout``) only when the loop can
     also end some other way: a test conjunct that reads no clock, or a ``break`` or
     ``return`` outside every clock-reading ``if``.  So a fake worker's
     ``while loop.time() - t0 < duration`` tick loop, which runs FOR a duration
     rather than waiting for an event, stays silent, while the one-line elapsed
     rewrite of a flagged poll does not.  AWAIT-REQUIRED excludes a sync
     ``time.sleep`` loop, which a loop-responsive budget cannot help.  The search
     does not descend into a nested ``while`` or scope, so a comparison belongs to
     the loop that owns it.  A ``for`` loop is not a leg of its own: its comparison
     belongs to the enclosing ``while``, so a top-level ``for`` clock poll is silent
     (none in the scanned tree, measured 2026-11-02).
  4. TIMEOUT BLOCK (task 7800): an ``async with asyncio.timeout(...)`` or
     ``asyncio.timeout_at(...)`` block whose own scope awaits a load-bearing target
     or runs an awaiting ``while`` poll.  It is ``asyncio.wait_for``'s deadline
     spelled as a context, so it counts the same way: the routing kind, plus the
     raw-literal kind when ``timeout``'s ``delay`` is a written number
     (``timeout_at`` takes an absolute time, so routing only).  LOAD-BEARING-OR-POLL
     is its discriminator because it mirrors the call leg's Name exclusion: a block
     around ``await task`` or ``await asyncio.gather(...)`` is as silent as
     ``asyncio.wait_for(task, ...)``, so neither spelling is stricter than the other,
     and ``asyncio.wait_for(ev.wait(), 5)`` cannot escape by becoming a block.

When deadline spellings nest — a block in a block, a clock poll in a block, a block
in a clock poll — the INNERMOST owns the waits inside it, so one wait is counted once.

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
frozenset for the same reason.  So no class list and no budget threshold decides which
sites are scanned.

Both legs are pure SHAPE, on any receiver (a Name, an Attribute, a Subscript or a
Call), in any scope of any file.  No receiver name is consulted:

  * the ``req.result`` leg is an ``ast.Attribute`` with ``attr == 'result'``;
  * the barrier leg is a ``.wait()`` method call taking NO arguments.  The
    zero-argument requirement is the structural discriminator: an ``asyncio.Event``
    ``.wait()`` takes none, while ``asyncio.wait(aws)`` and
    ``threading.Event.wait(timeout)`` take arguments and so stay out.

The accepted false-positive surface is an asyncio subprocess ``Process.wait()`` and a
``Condition``/``Barrier`` ``.wait()``.  The latter two are genuine synchronisation
points, so flagging them is correct; ``# noqa: wall-clock-deadline — <reason>`` is the
escape for a ``Process.wait()``.

Remedy reachability: ``wait_responsive`` and ``MERGE_RESULT_TIMEOUT`` are defined in
``orchestrator/tests/_orch_helpers.py``, and ``poll_responsive`` in
``orchestrator/tests/_poll_responsive.py``; all three are importable only under
orchestrator/tests, so every Rule C message says that elsewhere the per-site
``# noqa`` is the remedy.
Scoping the rule to orchestrator/tests instead would reintroduce exactly the
a-list-decides-coverage failure mode the paragraph above rejects.

What is deliberately NOT load-bearing: a wait on a bare ``ast.Name`` target, i.e. the
``asyncio.wait_for(worker_task, ...)`` teardown join in ``stop_merge_worker``.  It sits
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
  • ``poll_responsive(predicate, *, timeout=MERGE_RESULT_TIMEOUT, label, ...)`` for a
    poll (orchestrator/tests/_poll_responsive.py::poll_responsive)
  • a bound derived from ``MERGE_RESULT_TIMEOUT``, never a written number

Origin: task 2376 (the sweep whose threshold left the gap), task 3980 (migrated the
sites and added a file-local guard), task 4246 (this shared, repo-wide guard), task
7800 (the poll leg).

---------------------------------------------------------------------------

This script is intentionally stdlib-only (ast, collections.abc, pathlib, sys, typing,
plus its stdlib-only sibling _lint_cli.py) so
hooks/project-checks can invoke it via plain python3 without uv env-resolution overhead.
Adding a third-party dependency here would break that fast path.  This is why
``_DATACLASS_SHAPES`` never imports the dataclasses it describes: ``import
orchestrator.verify`` would need pydantic and break every caller.  Instead it reads
each one's fields from its defining module with ``dataclass_field_names``: the same
class definition orchestrator/src/orchestrator/remote_gate.py::wire_fingerprint
hashes via ``wire_classes()``, so one wire-shape change reaches both (task 7268).
The reader's agreement with ``dataclasses.fields`` is guarded in
fused-memory/tests/test_check_bare_magicmock_config.py::TestDataclassShapeRegistry.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path, PurePosixPath
from typing import NamedTuple, TypeGuard

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Sibling import that survives `python3 -I`: see _lint_cli.py's module docstring.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from _lint_cli import Violation, discover_files, is_exempted, run_cli
finally:
    del sys.path[0]


# Names that the checker considers "config objects".
# Exact matches plus suffix rules — generic names are excluded by design.
_CONFIG_EXACT: frozenset[str] = frozenset({'config', 'cfg'})
_CONFIG_SUFFIXES: tuple[str, ...] = ('_config', '_cfg')

_DATACLASS_PSEUDO_FIELD_MARKERS: frozenset[str] = frozenset({'ClassVar', 'InitVar', 'KW_ONLY'})


def _is_dataclass_pseudo_field(annotation: ast.expr) -> bool:
    """Return True if *annotation* marks a pseudo-field: ``ClassVar``, ``InitVar`` or ``KW_ONLY``."""
    marker = annotation.value if isinstance(annotation, ast.Subscript) else annotation
    if isinstance(marker, ast.Name):
        return marker.id in _DATACLASS_PSEUDO_FIELD_MARKERS
    if isinstance(marker, ast.Attribute):
        return marker.attr in _DATACLASS_PSEUDO_FIELD_MARKERS
    return False


def dataclass_field_names(path: Path, class_name: str) -> frozenset[str]:
    """Mirror ``dataclasses.fields()`` for a base-less dataclass, without importing it."""
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    matches = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f'expected exactly one top-level class {class_name!r} in {path}, '
            f'found {len(matches)}'
        )
    (cls,) = matches
    if cls.bases:
        bases = ', '.join(ast.unparse(base) for base in cls.bases)
        raise ValueError(
            f'class {class_name!r} in {path} has base classes ({bases}); '
            'inherited fields are invisible to a stdlib AST read'
        )
    return frozenset(
        stmt.target.id
        for stmt in cls.body
        if isinstance(stmt, ast.AnnAssign)
        and stmt.simple
        and isinstance(stmt.target, ast.Name)
        and not _is_dataclass_pseudo_field(stmt.annotation)
    )


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

    ``home`` is the repo-relative path of the module that defines the dataclass,
    which Rule B reads its ``fields`` from.  ``factory`` names the helper the message
    offers as the remedy, and ``factory_home`` the repo-relative path of the module
    that defines it.
    """

    name: str
    home: str
    fields: frozenset[str]
    anchors: frozenset[str]
    min_field_matches: int
    factory: str
    factory_home: str


def _registered_shape(
    *,
    name: str,
    home: str,
    anchors: frozenset[str],
    min_field_matches: int,
    factory: str,
    factory_home: str,
) -> _DataclassShape:
    return _DataclassShape(
        name=name,
        home=home,
        fields=dataclass_field_names(_REPO_ROOT / home, name),
        anchors=anchors,
        min_field_matches=min_field_matches,
        factory=factory,
        factory_home=factory_home,
    )


# Registered shapes: each one's fields are read from its ``home`` module (see the
# module docstring's last paragraph for why it is never imported).
_DATACLASS_SHAPES: tuple[_DataclassShape, ...] = (
    _registered_shape(
        name='VerifyResult',
        home='orchestrator/src/orchestrator/verify.py',
        anchors=frozenset({'passed'}),
        min_field_matches=2,
        factory='fake_verify_result',
        factory_home='orchestrator/tests/_verify_result_doubles.py',
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


# Each rule gets its OWN ``# noqa`` code, under _lint_cli.py's one pragma grammar: the
# remedies are unrelated (mock_orch_config/pydantic_spec vs
# fake_verify_result/spec=VerifyResult vs wait_responsive/MERGE_RESULT_TIMEOUT), so a
# pragma for one is not informed consent for another.
_RULE_A_CODE = 'bare-magicmock'
_RULE_B_CODE = 'bare-dataclass-double'
_RULE_C_CODE = 'wall-clock-deadline'

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
        f'unspecced MagicMock shaped like {shape.home}::{shape.name}'
        f' (matched {shape.name} fields: {", ".join(sorted(kwargs & shape.fields))}).'
        ' Reading an absent attribute on it auto-vivifies a truthy child Mock instead'
        f' of raising AttributeError.{drift_clause}'
        f' Use the {shape.factory}(...) helper ({shape.factory_home}),'
        f' or MagicMock(spec={shape.name}) seeded'
        f' from dataclasses.fields({shape.name}), so unknown-attribute reads raise'
        ' (tasks 3477/3980 built the factory; task 4016 added this guard).'
        ' To suppress: add # noqa: bare-dataclass-double — <reason> on the preceding'
        ' non-blank line.'
    )


class _DebtEntry(NamedTuple):
    """One file's row in a debt baseline.

    *key* is the baseline's OWN spelling of the path, which a remedy must quote: the
    scanned filename may be absolute, and only the key is an edit the author can apply
    verbatim.
    """

    key: str
    budget: int


def _debt_entry(filename: str, debt: dict[str, int]) -> _DebtEntry | None:
    """Return *filename*'s row in the *debt* mapping, or None if it is not a debt file.

    Parameterised over *debt* so every baseline matches paths the same way (task 4246).

    Compares TRAILING PATH COMPONENTS, not substrings: ``a/b/c.py`` matches an
    entry ``b/c.py`` because its last two components are exactly ``b`` and ``c.py``.
    Component-awareness is what stops ``orchestrator/tests/not_test_merge_queue.py``
    — which merely contains a debt filename as a substring — from being grandfathered.

    Trailing-component matching (rather than an exact string compare) is required
    because the nine call sites pass repo-relative paths while pytest passes
    absolutes; both must reach the same verdict.

    Returns None (not a zero budget) for a non-debt file so callers can distinguish
    "no budget recorded — report every site" from "budget of zero".
    """
    parts = PurePosixPath(filename.replace('\\', '/')).parts
    for entry, allowed in debt.items():
        entry_parts = PurePosixPath(entry).parts
        if len(parts) >= len(entry_parts) and parts[-len(entry_parts) :] == entry_parts:
            return _DebtEntry(entry, allowed)
    return None


def _apply_debt_budget(
    found: list[Violation],
    entry: _DebtEntry | None,
    *,
    filename: str,
    build_overrun_msg: Callable[[_DebtEntry, int], str],
    build_slack_msg: Callable[[_DebtEntry, int], str],
) -> list[Violation]:
    """Compare one rule's violation COUNT for a file against its debt budget.

    This is the single place a file's count is compared with its budget.

    Both builders are the caller's own ``(entry, found) -> str`` message builders,
    REQUIRED and keyword-only (task 4246 amendment pass).  Each rule's report carries
    its OWN remedy — telling a Rule C overrun to migrate a mock double would send the
    reader down a dead end — so there is deliberately no default: a future Rule D
    that forgets one fails at the call, rather than silently inheriting another
    rule's remedy with no type error and no test failure.  They are passed as
    builders rather than built strings so each message is computed only on its own
    branch and no caller has to invent a value for the *entry is None* branch.

    - Not a debt file (*entry* is None) → every violation is reported unchanged.
    - Exactly at budget → silence: this is the grandfathering the baseline exists for.
    - Over budget → report exactly ``found - budget`` violations, so the noise is
      proportional to the overrun rather than dumping every violation of a large
      debt file on someone who added one.  The reported sites are the last in source
      order.  That choice is deterministic rather than diagnostic — the checker
      cannot know which site is new — and the message says so explicitly.
    - Under budget → ONE file-level violation at *filename* line 1, col 0, naming
      the budget edit.  Slack has no site (the removed one is gone), and a budget
      above the measurement would silently licence that many new violations (task
      6202 moved this check here from another package's test suite).
    """
    if entry is None:
        return found
    if len(found) == entry.budget:
        return []
    if len(found) < entry.budget:
        return [
            Violation(
                filename=filename,
                lineno=1,
                col_offset=0,
                message=build_slack_msg(entry, len(found)),
            )
        ]
    ordered = sorted(found, key=lambda v: (v.lineno, v.col_offset))
    message = build_overrun_msg(entry, len(ordered))
    return [v._replace(message=message) for v in ordered[entry.budget :]]


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
    if is_exempted(lines, call.lineno, _RULE_B_CODE):
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
# carries, measured by AST census over all seven scanned tests/ directories.
#
# Provenance.  Task 4246 shipped Rule C with 618 violations across 20 files, every
# one under orchestrator/tests/ (333 bare-wait_for + 285 raw-literal); migrations
# had shrunk that to 8 files by task 5269.  Task 5269 then RE-MEASURED the census
# after WIDENING DETECTION — the barrier leg became a structural zero-argument
# ``.wait()`` on any receiver, the raw-literal leg learned ``max_wall_s=`` and
# wait_for's positional timeout, and directory discovery gained ``_*.py`` helper
# modules — and recorded 650 violations across 47 files in four packages.  Task 7800
# re-measured once more after Rule C learned asyncio.timeout/timeout_at blocks and
# clock-deadline polls, and recorded 311 violations across 91 files in five packages
# (220 across 39 before it).  Those are the two sanctioned rises: each time every
# added violation was already in the tree, newly SEEN, not new debt.  The
# non-orchestrator entries can shrink only by a per-site suppression pragma,
# ``# noqa: wall-clock-deadline`` until follow-up ticket
# tkt_0RVCE77JGW0CZGWPSFFZ00REB5 makes ``wait_responsive`` importable outside
# orchestrator/tests.
#
# The number counts VIOLATIONS, not SITES.
# One call can produce two: `asyncio.wait_for(req.result, timeout=25.0)` is
# simultaneously the wrong routing (bare-wait_for) and a written number
# (raw-literal).
#
# Shipping the rule hot with no transition would have turned the lint_command of
# every package listed here red on day one and stalled the merge lane repo-wide —
# the identical situation Rule B faced at 95 sites/11 files, so these are
# grandfathered.  Rule C ONLY: Rules A and B still apply in full to every file here.
#
# The count is an EXACT BUDGET, not a comment.  A debt file is silent ONLY while it
# carries exactly its recorded number.  Above it the checker reports the overrun;
# below it, one file-level slack violation naming the edit that lowers the number.
# So "shrink-only" is enforced on the same hot path the rule itself runs on rather
# than trusted.  This matters most for an actively-developed debt file, where a
# wholesale grandfather would have made a brand-new wall-clock wait added tomorrow
# invisible to the gate — and slack left above the measurement would license that
# many new waits just as silently.  Reporting the slack here puts the red in the
# owning package's own lint_command and the pre-commit hook, rather than in
# another package's suite (task 6202).
#
# DO NOT ADD ENTRIES, AND DO NOT RAISE A NUMBER.  Both may only shrink, as files are
# migrated onto wait_responsive(...) or poll_responsive(...) with bounds derived from
# MERGE_RESULT_TIMEOUT.
# A NEW file with a load-bearing wall-clock wait must FAIL the gate — that is the
# entire reason the baseline is opt-OUT rather than opt-in.  An opt-in list would
# exempt precisely the brand-new file this rule exists to catch, and would make
# "which files are covered" a hand-maintained list: the exact failure mode task
# 3980's amendment pass deleted a five-class frozenset to escape.
#
# orchestrator/tests/test_merge_speculation.py is deliberately NOT here.  It measures
# ZERO because task 3980 already migrated it (and task 7800 its one clock-deadline
# poll), and even a budgeted entry of zero would
# be a blanket suppression letting a regression land there silently — which is what
# 3980 spent a task removing, and what makes it safe for task 4246 to delete that
# module's file-local copy of this guard.
#
# Two entries share the trailing ``tests/test_harness.py``; ``_debt_entry`` matches
# on trailing path COMPONENTS, so the package directory keeps them distinct.
_WALL_CLOCK_DEADLINE_DEBT: dict[str, int] = {
    # orchestrator/tests
    'orchestrator/tests/test_merge_queue_concurrent_verify.py': 65,
    'orchestrator/tests/test_merge_queue_request_liveness.py': 10,
    'orchestrator/tests/test_offline_lane.py': 10,
    'orchestrator/tests/test_live_merge_worker.py': 6,
    'orchestrator/tests/test_merge_queue.py': 6,
    'orchestrator/tests/test_verify_admission_integration_gate.py': 5,
    'orchestrator/tests/test_background_service.py': 4,
    'orchestrator/tests/test_harness_lifecycle_registry.py': 4,
    'orchestrator/tests/test_merge_queue_deep_dispatch.py': 4,
    'orchestrator/tests/test_merge_queue_verifier_raw_cancel.py': 4,
    'orchestrator/tests/_task_verify_integration_rig.py': 3,
    'orchestrator/tests/test_git_ops.py': 3,
    'orchestrator/tests/test_invoke.py': 3,
    'orchestrator/tests/test_merge_queue_bounce.py': 3,
    'orchestrator/tests/test_merge_queue_deep_landing.py': 3,
    'orchestrator/tests/test_merge_queue_invariant_integration_gate.py': 3,
    'orchestrator/tests/test_merge_queue_supervisor.py': 3,
    'orchestrator/tests/test_workflow_cancellation.py': 3,
    'orchestrator/tests/test_harness.py': 2,
    'orchestrator/tests/test_merge_queue_coalesce.py': 2,
    'orchestrator/tests/test_merge_queue_deep_integration_gate.py': 2,
    'orchestrator/tests/test_merge_queue_rebased_landing_dead_base.py': 2,
    'orchestrator/tests/test_merge_skew_tripwire.py': 2,
    'orchestrator/tests/test_merge_worktree_lifecycle_integration_gate.py': 2,
    'orchestrator/tests/test_offline_lane_infra_integration.py': 2,
    'orchestrator/tests/test_offline_lane_integration.py': 2,
    'orchestrator/tests/test_verify_cancel_unwind.py': 2,
    'orchestrator/tests/test_verify_ruff_config_boundary.py': 2,
    'orchestrator/tests/test_before_done_script.py': 1,
    'orchestrator/tests/test_cancel_scope_bounded_cleanup.py': 1,
    'orchestrator/tests/test_cascade_unblock.py': 1,
    'orchestrator/tests/test_coalesce_integration_gate.py': 1,
    'orchestrator/tests/test_done_phase_wedge.py': 1,
    'orchestrator/tests/test_harness_force_release_wedged_slot.py': 1,
    'orchestrator/tests/test_harness_offline_lane.py': 1,
    'orchestrator/tests/test_harness_resume_scheduler.py': 1,
    'orchestrator/tests/test_lane_lock_leak_guard.py': 1,
    'orchestrator/tests/test_merge_lane_parked_custody.py': 1,
    'orchestrator/tests/test_merge_queue_lifecycle_registry.py': 1,
    'orchestrator/tests/test_merge_queue_orphan_reaper.py': 1,
    'orchestrator/tests/test_merge_queue_restart_hook.py': 1,
    'orchestrator/tests/test_offline_lane_records.py': 1,
    'orchestrator/tests/test_remote_gate.py': 1,
    'orchestrator/tests/test_task_verify_host_race.py': 1,
    'orchestrator/tests/test_verify.py': 1,
    'orchestrator/tests/test_verify_admission_cancel_release.py': 1,
    'orchestrator/tests/test_verify_admission_executor_bypass.py': 1,
    'orchestrator/tests/test_verify_admission_gate.py': 1,
    'orchestrator/tests/test_verify_admission_order_integration.py': 1,
    'orchestrator/tests/test_verify_admission_watch.py': 1,
    'orchestrator/tests/test_verify_hosts.py': 1,
    'orchestrator/tests/test_verify_plan_latch.py': 1,
    'orchestrator/tests/test_verify_run_cmd_reap.py': 1,
    'orchestrator/tests/test_workflow_halt_owner.py': 1,
    # dashboard/tests
    'dashboard/tests/test_db.py': 26,
    'dashboard/tests/test_mcp_fanout.py': 18,
    'dashboard/tests/test_durability.py': 4,
    'dashboard/tests/test_metrics_curator.py': 4,
    'dashboard/tests/_dashboard_helpers.py': 3,
    'dashboard/tests/test_api_curator.py': 2,
    'dashboard/tests/test_memory_exchange_isolation.py': 2,
    'dashboard/tests/test_merge_queue_data.py': 2,
    'dashboard/tests/test_quality_sampler.py': 2,
    'dashboard/tests/test_quality_marker_sampler.py': 2,
    'dashboard/tests/test_escalation_corpus.py': 1,
    # fused-memory/tests
    'fused-memory/tests/test_harness.py': 7,
    'fused-memory/tests/test_drain_signal_handler.py': 4,
    'fused-memory/tests/test_memory_service.py': 4,
    'fused-memory/tests/test_operator_signal_handler.py': 4,
    'fused-memory/tests/test_periodic_rebuild_summaries.py': 4,
    'fused-memory/tests/test_task_interceptor.py': 4,
    'fused-memory/tests/_fm_polling.py': 2,
    'fused-memory/tests/server/test_grouped_read.py': 2,
    'fused-memory/tests/test_dependency_direction_check.py': 2,
    'fused-memory/tests/test_e2e_durable_queue.py': 2,
    'fused-memory/tests/test_journaling_integration.py': 2,
    'fused-memory/tests/test_recon_claim_verification_wiring.py': 2,
    'fused-memory/tests/test_mem0_collection_provisioning.py': 2,
    'fused-memory/tests/test_event_loop_lag.py': 1,
    'fused-memory/tests/test_event_queue.py': 1,
    'fused-memory/tests/test_main_curator_usage_gate.py': 1,
    # shared/tests
    'shared/tests/test_cli_invoke.py': 2,
    'shared/tests/test_uuid_prefix_guard.py': 2,
    'shared/tests/test_async_sqlite_base.py': 1,
    'shared/tests/test_auth_failed.py': 1,
    'shared/tests/test_auth_reprobe_opt_out.py': 1,
    'shared/tests/test_cli_invoke_transcript_offloop.py': 1,
    'shared/tests/test_failover_integration.py': 1,
    'shared/tests/test_git_async.py': 1,
    'shared/tests/test_usage_gate_park_visibility.py': 1,
    # escalation/tests
    'escalation/tests/test_prefix_resolver_client.py': 2,
}


def _wall_clock_overrun_msg(entry: _DebtEntry, found: int) -> str:
    """Build the message for a Rule C debt file that has grown past its budget.

    Carries Rule C's OWN remedy, never Rule B's: a wall-clock deadline is not fixed
    by speccing a double, so offering ``fake_verify_result`` here would send the
    reader down a dead end.
    """
    budget = entry.budget
    return (
        'this file is on the SHRINK-ONLY wall-clock-deadline debt baseline with a'
        f' recorded budget of {budget} violation(s), but {found} were found —'
        f' {found - budget} over budget. The baseline may only shrink.'
        ' The reported sites are simply the LAST in source order: the anchor is'
        ' positional and is NOT a claim that these exact sites are the new ones.'
        ' Fix by migrating a load-bearing wait in this file onto wait_responsive(...)'
        f' ({_WAIT_RESPONSIVE_REF}), or a poll onto poll_responsive(predicate, ...)'
        f' ({_POLL_RESPONSIVE_REF}), with a descriptive label=, or by deriving its'
        ' bound from MERGE_RESULT_TIMEOUT instead of writing a number, or by adding'
        ' # noqa: wall-clock-deadline — <reason> above a deliberate one.'
        + _WALL_CLOCK_REMEDY_REACH
        + ' Do NOT raise the recorded budget in check_bare_magicmock_config.py.'
    )


def _wall_clock_slack_msg(entry: _DebtEntry, found: int) -> str:
    """Build the message for a Rule C debt file carrying fewer violations than its budget.

    Names the one edit that fixes it, quoting the baseline's own key.  No per-site
    suppression can fix slack and no migration remedy applies, so neither is offered.
    """
    key = f"_WALL_CLOCK_DEADLINE_DEBT['{entry.key}']"
    if found == 0:
        edit = (
            f'This file is fully migrated: delete {key} in'
            ' fused-memory/scripts/check_bare_magicmock_config.py AND its line in'
            ' _EXPECTED_WALL_CLOCK_DEBT_PATHS in'
            ' fused-memory/tests/test_check_bare_magicmock_config.py.'
        )
    else:
        edit = (
            f'Fix: lower {key} from {entry.budget} to {found} in'
            ' fused-memory/scripts/check_bare_magicmock_config.py.'
        )
    return (
        'this file is on the SHRINK-ONLY wall-clock-deadline debt baseline with a'
        f' recorded budget of {entry.budget} violation(s), but only {found} were'
        ' found. A budget above the measurement lets that many NEW wall-clock waits'
        ' land here silently, so the recorded number must be lowered in the same'
        f' change that removes a wait. {edit}'
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
      * ``done.wait()`` — a zero-argument ``.wait()`` barrier, on ANY receiver
        (``self._entered.wait()``, ``verifier.entered[0].wait()``).  Already
        event-driven; only its deadline is wall-clock.  Requiring zero arguments
        is what excludes ``asyncio.wait(aws)`` and ``threading.Event.wait(timeout)``;
        see the Rule C section of the module docstring for the accepted
        false-positive surface.

    Deliberately NOT load-bearing, and therefore excluded: the
    ``await asyncio.wait_for(worker_task, timeout=join_timeout)`` join in
    ``stop_merge_worker``.  It targets a bare ``Name`` (the worker Task), sits inside
    ``contextlib.suppress(Exception)``, asserts nothing, and swallows its own
    TimeoutError — so it cannot manufacture the flake task 3980 fixed, and
    stretching it would only slow teardown down.  The Name-vs-Attribute/Call
    distinction is what makes THAT exclusion structural rather than a
    hand-maintained name list, which is what lets this rule scan every scope of
    every file the nine call sites reach.  Neither leg selects by name.
    """
    if isinstance(node, ast.Attribute) and node.attr == 'result':
        return f'{ast.unparse(node)} (MergeRequest.result future)'
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'wait'
        and not node.args
        and not node.keywords
    ):
        return f'{ast.unparse(node)} (.wait() barrier)'
    return None


def _polled_predicate_target(node: ast.expr) -> str:
    """Describe *node* as a polled predicate: a poll's predicate IS its synchronisation point."""
    return f'{ast.unparse(node)} (polled predicate)'


class _ResponsiveHelper(NamedTuple):
    """A responsive wait helper: a Rule C call shape, and a remedy its messages cite.

    ``home`` is the repo-relative path of the module that defines it, and
    ``describe`` how it describes its first argument as a load-bearing target.
    """

    home: str
    describe: Callable[[ast.expr], str | None]


# The responsive wait helpers by the name a call site spells.  A stdlib-only literal
# copy of where they live, which this checker cannot import — the same precedent as
# ``_DATACLASS_SHAPES``, guarded from both sides.  Each ``home`` is parsed for its helper by
# fused-memory/tests/test_wall_clock_poll_deadlines.py::TestResponsiveHelperRegistry,
# and the name set must equal the one the wait-budget audit derives from the helpers
# themselves (orchestrator/tests/test_wait_budget_audit.py::TestRuleCRecognisesTheBilledHelpers).
# A responsive helper is itself the remedy, so only its raw-literal kind ever applies.
RESPONSIVE_WAIT_HELPERS: dict[str, _ResponsiveHelper] = {
    'wait_responsive': _ResponsiveHelper(
        home='orchestrator/tests/_orch_helpers.py', describe=_load_bearing_wait_target
    ),
    'poll_responsive': _ResponsiveHelper(
        home='orchestrator/tests/_poll_responsive.py', describe=_polled_predicate_target
    ),
}


def _helper_ref(name: str) -> str:
    """Cite responsive helper *name* as ``<home>::<name>``."""
    return f'{RESPONSIVE_WAIT_HELPERS[name].home}::{name}'


_WAIT_RESPONSIVE_REF = _helper_ref('wait_responsive')
_POLL_RESPONSIVE_REF = _helper_ref('poll_responsive')

# Shared by every offence kind: WHY a wall-clock deadline on a load-bearing sync
# point is a defect rather than a style preference.  Task 3980's three measured
# failures were all genuine deadline expiries on tests that had already passed.
_WALL_CLOCK_CONSEQUENCE = (
    ' A deadline expiry on a load-bearing synchronisation point fails a test whose'
    ' awaited event did happen, only because the loop was descheduled.'
)

# Shared by every offence kind and the debt overrun: every Rule C remedy names
# wait_responsive, poll_responsive or MERGE_RESULT_TIMEOUT, and all three live under
# orchestrator/tests while the rule scans every package.
_WALL_CLOCK_REMEDY_REACH = (
    ' wait_responsive, poll_responsive and MERGE_RESULT_TIMEOUT are importable only'
    ' under orchestrator/tests, so in another package the # noqa escape is the remedy.'
)

_WALL_CLOCK_SUPPRESS = (
    ' To suppress: add # noqa: wall-clock-deadline — <reason> on the preceding'
    ' non-blank line.'
)


class _DeadlineRouting(NamedTuple):
    """One SPELLING of a wall-clock deadline, and the remedy that re-routes it.

    Rule C's routing offence is the same defect however the deadline is spelled;
    only the clause naming the spelling and the helper that replaces it differ.
    """

    spelled: str
    remedy: str


_BARE_WAIT_FOR = _DeadlineRouting(
    spelled='is routed through a bare asyncio.wait_for',
    remedy=(
        ' Route it through wait_responsive(...) with a descriptive label='
        f' ({_WAIT_RESPONSIVE_REF}), which charges its'
        ' budget in loop-responsive time and still reports a genuine hang red.'
    ),
)

_CLOCK_DEADLINE = _DeadlineRouting(
    spelled='gives up against a wall-clock read',
    remedy=(
        ' Replace the loop with poll_responsive(predicate, ...) with a descriptive'
        f' label= ({_POLL_RESPONSIVE_REF}), which'
        ' polls under a loop-responsive budget and still reports a genuine hang red.'
    ),
)

_TIMEOUT_BLOCK = _DeadlineRouting(
    spelled='is bounded by an asyncio.timeout(...) block',
    remedy=(
        ' Route a load-bearing await through wait_responsive(...) and a poll through'
        ' poll_responsive(predicate, ...)'
        f' ({_POLL_RESPONSIVE_REF}), each with a'
        ' descriptive label=.'
    ),
)


def _wall_clock_routing_msg(routing: _DeadlineRouting, target: str) -> str:
    """Build Rule C's routing message: *target*'s deadline, spelled as *routing*, is wall clock.

    Deliberately shares NO vocabulary with ``_VIOLATION_MSG`` or
    ``_dataclass_violation_msg``: Rule A's remedies read pydantic ``model_fields``
    and Rule B's spec a stdlib dataclass — neither has anything to say about a
    wall-clock deadline on a future, so offering either here would send the reader
    down a dead end.
    """
    return (
        f'load-bearing wait on {target} {routing.spelled},'
        ' so its deadline is charged in WALL CLOCK.'
        + _WALL_CLOCK_CONSEQUENCE
        + routing.remedy
        + _WALL_CLOCK_REMEDY_REACH
        + _WALL_CLOCK_SUPPRESS
    )


def _wall_clock_raw_literal_msg(target: str, parameter: str) -> str:
    """Build Rule C's raw-literal message: *target*'s *parameter* bound is a written number.

    Kept apart from the routing message because the two are independent fixes:
    re-routing the wait and deriving its bound from ``MERGE_RESULT_TIMEOUT``, and a
    site can need one, the other, or both.
    """
    return (
        f'load-bearing wait on {target} carries a RAW wall-clock literal as its'
        f' {parameter} bound.'
        + _WALL_CLOCK_CONSEQUENCE
        + ' Derive the bound from MERGE_RESULT_TIMEOUT instead of writing a number:'
        ' a written literal is a threshold, and task 2376 measured that a policy'
        ' expressed as "literals up to N" cannot catch the one just above N.'
        + _WALL_CLOCK_REMEDY_REACH
        + _WALL_CLOCK_SUPPRESS
    )


# The keywords that carry a wall-clock bound on a wait call: ``timeout`` on every call
# shape, and ``max_wall_s``, the responsive helpers' wall-clock cap
# (orchestrator/tests/_orch_helpers.py::wait_responsive).
_WALL_CLOCK_BOUND_KEYWORDS: tuple[str, ...] = ('timeout', 'max_wall_s')


def _is_numeric_literal(node: ast.expr) -> bool:
    """Return True if *node* is a written int or float.

    ``bool`` is excluded explicitly because it is an int SUBCLASS — without the
    guard, ``timeout=True`` would be reported as a wall-clock number.
    """
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    )


def _raw_wall_clock_literal(call: ast.Call, *, is_bare_wait_for: bool) -> str | None:
    """Return the name of the first wall-clock bound on *call* written as a number.

    Checks the ``_WALL_CLOCK_BOUND_KEYWORDS`` keywords on any wait call shape, and
    ``asyncio.wait_for``'s positional timeout (``args[1]``) only when
    *is_bare_wait_for*: a responsive helper's ``timeout`` is keyword-only, so a
    positional second argument there is a TypeError, not a lintable site.
    """
    for kw in call.keywords:
        if kw.arg in _WALL_CLOCK_BOUND_KEYWORDS and _is_numeric_literal(kw.value):
            return kw.arg
    if is_bare_wait_for and len(call.args) >= 2 and _is_numeric_literal(call.args[1]):
        return 'timeout'
    return None


def _asyncio_function_name(call: ast.Call) -> str | None:
    """Return *attr* if *call* is ``asyncio.<attr>(...)``, spelled through the ``asyncio`` name.

    ``gate.wait_for(...)`` and ``self.timeout(...)`` are not, so they return None.
    """
    func = call.func
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == 'asyncio'
    ):
        return func.attr
    return None


def _wall_clock_deadline_violations(
    node: ast.AST, lines: list[str], filename: str
) -> list[Violation]:
    """Rule C, evaluated for ONE node of any type: return 0, 1 or 2 Violations.

    Dispatches on the node type, one leg per spelling of a wall-clock deadline:
    an ``ast.Call`` to the wait-call leg (``_wait_call_violations``), an
    ``ast.While`` to the clock-deadline poll leg (``_clock_deadline_poll_violations``)
    and an ``ast.AsyncWith`` to the timeout-block leg (``_timeout_block_violations``).
    A sync ``with`` is not inspected: ``asyncio.timeout`` is async-only.  Every other
    node yields nothing.

    Per-NODE rather than per-tree so ``find_violations`` can evaluate all three rules
    in a SINGLE ``ast.walk``; a second full walk was measured at ~43% of total checker
    runtime (20.8s -> 30.7s over the seven scanned dirs), paid on every merge-queue
    verify across all nine call sites.  Position-blind by construction: the caller
    hands this every node, so there is no class list and no scope filter — see the
    Rule C section of the module docstring for why a list, or a threshold, is not a
    sound key.

    Must never raise: a crash would fail every caller's lint over an unrelated edit.
    """
    if isinstance(node, ast.Call):
        return _wait_call_violations(node, lines, filename)
    if isinstance(node, ast.While):
        return _clock_deadline_poll_violations(node, lines, filename)
    if isinstance(node, ast.AsyncWith):
        return _timeout_block_violations(node, lines, filename)
    return []


def _wait_call_violations(call: ast.Call, lines: list[str], filename: str) -> list[Violation]:
    """Rule C's wait-call leg, for ONE ``ast.Call``: return 0, 1 or 2 Violations.

    Returns a LIST, not an Optional, because the two offence kinds are independent
    and a single call can trip both — ``asyncio.wait_for(req.result, timeout=25.0)``
    is simultaneously the wrong routing and a written number.  The written number is
    a raw numeric literal on ``timeout=`` or ``max_wall_s=``, or wait_for's
    positional timeout (``_raw_wall_clock_literal``).

    Gating order is cheapest-first, and deliberately so:
      1. the call has at least one positional argument (no ``args[0]`` to inspect
         otherwise — and a bare ``asyncio.wait_for()`` must be skipped, not crash);
      2. the func is ``asyncio.wait_for`` or a ``RESPONSIVE_WAIT_HELPERS`` helper;
      3. that shape's describer recognises ``args[0]`` (``_load_bearing_wait_target``,
         except a ``poll_responsive`` predicate, which is always load-bearing).
    Only then is the exemption line-walk run, so the upward walk keeps Rule B's cost
    profile rather than running on every ``ast.Call`` in the tree.

    Violations carry ``call.col_offset``, so a node tripping several rules yields
    deterministically-ordered entries under ``find_violations``' final sort.  Both
    kinds from one call share a position; Python's stable sort keeps them in the
    fixed order they are appended here.
    """
    if not call.args:
        return []

    func = call.func
    is_bare_wait_for = _asyncio_function_name(call) == 'wait_for'
    if is_bare_wait_for:
        describe = _load_bearing_wait_target
    elif isinstance(func, ast.Name) and func.id in RESPONSIVE_WAIT_HELPERS:
        describe = RESPONSIVE_WAIT_HELPERS[func.id].describe
    else:
        return []

    target = describe(call.args[0])
    if target is None:
        return []

    # Computed lazily — only after the func shape AND the load-bearing target have
    # both matched — so the upward line walk keeps Rule B's cost profile rather than
    # running on every ast.Call in the tree.  ONE check per SITE suppresses BOTH
    # offence kinds: a pragma is consent for the site, not for one half of it.
    if is_exempted(lines, call.lineno, _RULE_C_CODE):
        return []

    messages: list[str] = []
    if is_bare_wait_for:
        messages.append(_wall_clock_routing_msg(_BARE_WAIT_FOR, target))

    # A raw numeric literal is an offence on ANY wait call shape: the responsive
    # helpers also take ``timeout`` and ``max_wall_s`` keywords, so a migrated site can
    # have moved the accounting into loop-responsive time while keeping a
    # hand-written number.  At most ONE raw-literal violation per call, however
    # many bounds carry a number: it is one offence with one remedy.
    parameter = _raw_wall_clock_literal(call, is_bare_wait_for=is_bare_wait_for)
    if parameter is not None:
        messages.append(_wall_clock_raw_literal_msg(target, parameter))

    return [
        Violation(
            filename=filename,
            lineno=call.lineno,
            col_offset=call.col_offset,
            message=message,
        )
        for message in messages
    ]


# The nodes that open a new scope: an await, or a comparison, inside one belongs to
# that scope's own code, not to the loop or block that merely contains its definition.
_SCOPE_NODES: tuple[type[ast.AST], ...] = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.ClassDef,
)

# The loops: a ``break`` inside one ends that loop, not an enclosing one.
_LOOP_NODES: tuple[type[ast.AST], ...] = (ast.While, ast.For, ast.AsyncFor)


def _opens_scope(node: ast.AST) -> bool:
    """Return True if *node* opens a new scope (``_SCOPE_NODES``)."""
    return isinstance(node, _SCOPE_NODES)


def _walk_own_scope(
    nodes: Iterable[ast.AST], *, prune: Callable[[ast.AST], bool] = _opens_scope
) -> Iterator[ast.AST]:
    """Yield *nodes* and their descendants in source order, skipping every node *prune* accepts.

    A pruned node is skipped whole: neither it nor any of its descendants is yielded.
    """
    pending = list(reversed(list(nodes)))
    while pending:
        node = pending.pop()
        if prune(node):
            continue
        yield node
        pending.extend(reversed(list(ast.iter_child_nodes(node))))


def _awaits(
    stmts: Iterable[ast.AST], *, prune: Callable[[ast.AST], bool] = _opens_scope
) -> bool:
    """Return True if *stmts* await anything outside every node *prune* accepts."""
    return any(isinstance(node, ast.Await) for node in _walk_own_scope(stmts, prune=prune))


# The zero-argument methods that read a wall clock: ``loop.time()``, ``time.time()``,
# ``time.monotonic()``, ``time.perf_counter()``.  Matched by attribute name only.
_WALL_CLOCK_READS: frozenset[str] = frozenset({'time', 'monotonic', 'perf_counter'})


def _is_wall_clock_read(node: ast.AST) -> TypeGuard[ast.Call]:
    """Return True if *node* is a zero-argument call of a ``_WALL_CLOCK_READS`` method.

    The receiver is never consulted, like the barrier leg's ``.wait()``: no name
    list decides what is a clock.
    """
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _WALL_CLOCK_READS
        and not node.args
        and not node.keywords
    )


def _reads_clock(node: ast.AST) -> bool:
    """Return True if a wall-clock read appears anywhere in *node*."""
    return any(_is_wall_clock_read(inner) for inner in ast.walk(node))


def _elapsed_clock_read(node: ast.expr) -> ast.Call | None:
    """Return the clock read in *node* if it is an elapsed time, ``<clock read> - <start>``."""
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Sub)
        and _is_wall_clock_read(node.left)
    ):
        return node.left
    return None


def _give_up_clock_read(node: ast.expr) -> ast.Call | None:
    """Return the clock read in *node* if it is one measured against a deadline.

    That is the read itself, or a remaining time, ``<deadline> - <clock read>``.
    """
    if _is_wall_clock_read(node):
        return node
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Sub)
        and _is_wall_clock_read(node.right)
    ):
        return node.right
    return None


def _ends_other_than_by_its_clock(loop: ast.While) -> bool:
    """Return True if *loop* has an exit that no clock governs.

    That is a conjunct of its test that reads no clock and is not a constant, or a
    ``break`` or ``return`` in its own code outside the body of every ``if`` whose test
    reads a clock.  A nested loop is not searched: its ``break`` ends that loop.  A loop
    with neither runs FOR a duration — a fake worker's tick loop — rather than polling.
    """
    test = loop.test
    conjuncts = (
        test.values
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And)
        else [test]
    )
    if any(not isinstance(c, ast.Constant) and not _reads_clock(c) for c in conjuncts):
        return True
    own_body = list(
        _walk_own_scope(
            loop.body, prune=lambda node: isinstance(node, (*_SCOPE_NODES, *_LOOP_NODES))
        )
    )
    clock_governed = {
        id(node)
        for clock_check in own_body
        if isinstance(clock_check, ast.If) and _reads_clock(clock_check.test)
        for node in _walk_own_scope(clock_check.body)
    }
    return any(
        isinstance(node, (ast.Break, ast.Return)) and id(node) not in clock_governed
        for node in own_body
    )


def _clock_deadline_read(loop: ast.While) -> ast.Call | None:
    """Return the wall-clock read *loop* compares against to give up, else None.

    That is a DIRECT operand of a ``Compare`` in the loop's test or own body that is a
    wall-clock read or a remaining time (``_give_up_clock_read``).  Failing one, it is
    the read inside an elapsed-time operand (``_elapsed_clock_read``), but only for a
    loop that ``_ends_other_than_by_its_clock``.  A nested ``while`` or scope is not
    searched: its comparison is its own.  A nested ``for`` IS searched: a ``for`` loop
    is not a leg of its own, so its comparison belongs to the enclosing ``while``.
    """
    own_code = _walk_own_scope(
        [loop.test, *loop.body],
        prune=lambda node: isinstance(node, (*_SCOPE_NODES, ast.While)),
    )
    elapsed_reads: list[ast.Call] = []
    for node in own_code:
        if not isinstance(node, ast.Compare):
            continue
        for operand in (node.left, *node.comparators):
            give_up = _give_up_clock_read(operand)
            if give_up is not None:
                return give_up
            elapsed = _elapsed_clock_read(operand)
            if elapsed is not None:
                elapsed_reads.append(elapsed)
    if elapsed_reads and _ends_other_than_by_its_clock(loop):
        return elapsed_reads[0]
    return None


class _DeadlineContext(NamedTuple):
    """An ``asyncio.<attr>(...)`` deadline context's one argument.

    ``keyword`` is that argument's keyword spelling, and ``is_duration`` whether it
    is a duration, so a written number there is a raw literal, rather than an
    absolute time.
    """

    keyword: str
    is_duration: bool


_DEADLINE_CONTEXTS: dict[str, _DeadlineContext] = {
    'timeout': _DeadlineContext(keyword='delay', is_duration=True),
    'timeout_at': _DeadlineContext(keyword='when', is_duration=False),
}


def _deadline_context(item: ast.withitem) -> tuple[ast.expr, _DeadlineContext] | None:
    """Return *item*'s deadline argument and context if it is an asyncio deadline.

    None unless the item is an ``asyncio.<attr>(...)`` call with *attr* in
    ``_DEADLINE_CONTEXTS`` whose argument, positional or by that context's own
    keyword, is present and not the literal None: a missing or None deadline bounds
    nothing, and another context's keyword is a TypeError, not a deadline.
    """
    call = item.context_expr
    if not isinstance(call, ast.Call):
        return None
    name = _asyncio_function_name(call)
    if name not in _DEADLINE_CONTEXTS:
        return None
    context = _DEADLINE_CONTEXTS[name]
    keyword_bounds = [kw.value for kw in call.keywords if kw.arg == context.keyword]
    bounds = keyword_bounds or call.args[:1]
    if not bounds:
        return None
    bound = bounds[0]
    if isinstance(bound, ast.Constant) and bound.value is None:
        return None
    return bound, context


def _block_deadline(block: ast.AsyncWith) -> tuple[ast.expr, _DeadlineContext] | None:
    """Return the first asyncio deadline among *block*'s items (``_deadline_context``)."""
    return next(
        (found for item in block.items if (found := _deadline_context(item)) is not None),
        None,
    )


def _owns_its_waits(node: ast.AST) -> bool:
    """Return True if *node* opens a scope or is a deadline spelling that reports its waits.

    Either owns the waits inside it.  A scope's code is its own, and a nested
    ``asyncio`` deadline block or clock-deadline ``while`` that Rule C flags is the
    innermost deadline on what it contains, so its waits are attributed to it alone
    and one wait is counted once.  A nested spelling that flags nothing — a
    per-attempt ``asyncio.timeout`` around a non-load-bearing await — owns nothing,
    so the enclosing deadline still sees the waits inside it.
    """
    if isinstance(node, ast.AsyncWith):
        return _block_deadline(node) is not None and _timeout_block_target(node) is not None
    if isinstance(node, ast.While):
        return _clock_deadline_read(node) is not None and _awaits(
            node.body, prune=_owns_its_waits
        )
    return _opens_scope(node)


def _clock_deadline_poll_violations(
    loop: ast.While, lines: list[str], filename: str
) -> list[Violation]:
    """Rule C's clock-deadline poll leg, for ONE ``ast.While``: return 0 or 1 Violations.

    Flags a loop that gives up against a wall-clock read and awaits outside every
    nested deadline spelling, which owns its own waits (``_owns_its_waits``).  Only the
    routing kind applies: the loop's bound is arithmetic in another statement, so
    there is no literal on the inspected node to report.
    """
    read = _clock_deadline_read(loop)
    if read is None or not _awaits(loop.body, prune=_owns_its_waits):
        return []
    if is_exempted(lines, loop.lineno, _RULE_C_CODE):
        return []
    target = f'poll loop `while {ast.unparse(loop.test)}` ({ast.unparse(read)} deadline)'
    return [
        Violation(
            filename=filename,
            lineno=loop.lineno,
            col_offset=loop.col_offset,
            message=_wall_clock_routing_msg(_CLOCK_DEADLINE, target),
        )
    ]


def _timeout_block_target(block: ast.AsyncWith) -> str | None:
    """Describe what *block*'s deadline bounds, if it is a load-bearing wait, else None.

    A load-bearing await in the block's own code wins; otherwise an awaiting
    ``while`` loop in it, which is a poll.  Anything else — ``await task``,
    ``await asyncio.gather(...)``, a plain call — is not load-bearing.  A nested
    deadline spelling is not searched: it owns the waits inside it (``_owns_its_waits``).
    """
    own_code = list(_walk_own_scope(block.body, prune=_owns_its_waits))
    for node in own_code:
        if isinstance(node, ast.Await):
            target = _load_bearing_wait_target(node.value)
            if target is not None:
                return target
    for node in own_code:
        if isinstance(node, ast.While) and _awaits(node.body, prune=_owns_its_waits):
            return f'poll loop `while {ast.unparse(node.test)}`'
    return None


def _timeout_block_violations(
    block: ast.AsyncWith, lines: list[str], filename: str
) -> list[Violation]:
    """Rule C's timeout-block leg, for ONE ``ast.AsyncWith``: return 0, 1 or 2 Violations.

    The first asyncio deadline context among the block's items decides.  The routing
    kind applies whenever the block bounds a load-bearing wait; the raw-literal kind
    as well when that context's argument is a duration written as a number, exactly
    as for ``asyncio.wait_for(x, 25.0)``.
    """
    deadline = _block_deadline(block)
    if deadline is None:
        return []
    bound, context = deadline
    target = _timeout_block_target(block)
    if target is None:
        return []
    if is_exempted(lines, block.lineno, _RULE_C_CODE):
        return []

    messages = [_wall_clock_routing_msg(_TIMEOUT_BLOCK, target)]
    if context.is_duration and _is_numeric_literal(bound):
        messages.append(_wall_clock_raw_literal_msg(target, context.keyword))
    return [
        Violation(
            filename=filename,
            lineno=block.lineno,
            col_offset=block.col_offset,
            message=message,
        )
        for message in messages
    ]


def find_violations(source: str, filename: str) -> list[Violation]:
    """Parse *source* and return ``find_violations_in_tree`` over it.

    SyntaxError in *source* → returns an empty list.
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return []
    return find_violations_in_tree(tree, source, filename)


def find_violations_in_tree(tree: ast.Module, source: str, filename: str) -> list[Violation]:
    """Return every violation of ALL THREE rules in *tree*, the parse of *source*.

    The seam for a caller that already holds the parse and runs other predicates
    over the same tree, so that tree is parsed once.

    Rule A — ``bare-magicmock``: each ``<config_name> = MagicMock()`` that has at
    least one ast.Name target whose id matches the config-name set, calls MagicMock
    (by name or attribute) with no spec/spec_set, and is NOT preceded on the nearest
    non-blank source line by a valid exemption comment.  For chained assignments
    (``mock = config = MagicMock()``) each ast.Name target is evaluated independently
    and may produce a separate Violation.

    Rule B — ``bare-dataclass-double``: each unspecced ``MagicMock(...)`` in ANY
    position whose literal kwargs match a registered ``_DATACLASS_SHAPES`` entry.

    Rule C — ``wall-clock-deadline``: each load-bearing wait (a
    ``MergeRequest.result`` future or a zero-argument ``.wait()`` barrier) routed through a
    bare ``asyncio.wait_for``, and/or carrying a raw numeric literal bound
    (``timeout=``, ``max_wall_s=`` or wait_for's positional timeout).  One call can
    produce TWO violations — the kinds are independent.  Also each async ``while``
    poll that gives up against a wall-clock read (one violation, at the ``while``),
    and each ``asyncio.timeout``/``timeout_at`` block bounding such a poll or a
    load-bearing await (routing, plus raw-literal on a written ``delay``).

    Rule C additionally honours its own SHRINK-ONLY per-file debt budget
    (``_WALL_CLOCK_DEADLINE_DEBT``), compared with the collected COUNT after the
    walk: a debt file is silent only at exactly its budget, reports its overrun above
    it and one file-level slack violation below it.  Rules A and B have no baseline
    and apply in full to every file.

    Returned violations are sorted ascending by (lineno, col_offset) for
    deterministic source-order output (ast.walk yields BFS order, not source order).
    """
    lines = source.splitlines()
    violations: list[Violation] = []

    # Rule C violations are collected separately so its debt file's budget can be
    # compared with the COUNT after the walk (see _apply_debt_budget).  Looked up ONCE
    # per file, not per node.
    wall_clock: list[Violation] = []
    wall_clock_debt = _debt_entry(filename, _WALL_CLOCK_DEADLINE_DEBT)

    # ONE walk, ALL THREE rules.  Rule B was originally a second full ast.walk over the
    # same tree, which cost ~43% of total checker runtime (20.8s → 30.7s over the seven
    # scanned tests/ dirs) — a cost paid on every merge-queue verify across all nine
    # call sites.  Rule A's walk already visits every node and simply skips non-Assign
    # ones, so folding Rule B's and Rule C's per-node handling in here makes them nearly
    # free.  Output is unchanged: the final sort by (lineno, col_offset) still
    # normalises ordering.
    for node in ast.walk(tree):
        # ---- Rule C: wall-clock-deadline, every node, dispatched on its type ----
        # Runs FIRST and unconditionally: its legs read ast.Call, ast.While AND
        # ast.AsyncWith nodes, so neither Rule B's `continue` nor Rule A's type filter
        # below may skip it.
        wall_clock.extend(_wall_clock_deadline_violations(node, lines, filename))

        # ---- Rule B: bare-dataclass-double, position-blind over every ast.Call ----
        # An ast.Call is never an ast.Assign/ast.AnnAssign, so this branch and Rule A's
        # below are mutually exclusive and the `continue` cannot skip Rule A work.
        if isinstance(node, ast.Call):
            double = _dataclass_double_violation(node, lines, filename)
            if double is not None:
                violations.append(double)
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

        # is_exempted is computed lazily — only on finding the first config-named
        # ast.Name target — because the exemption check (an upward line walk + regex)
        # is not free, and most assignments have no config-named targets.
        exempted: bool | None = None
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if not _is_config_name(target.id):
                continue
            if exempted is None:
                exempted = is_exempted(lines, assignment_lineno, _RULE_A_CODE)
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

    # Rule C's per-file debt budget is compared with its collected COUNT, not with
    # individual sites: a grandfathered file stays silent only at exactly its budget,
    # reports exactly its overrun above it, and reports one slack violation below it.
    violations.extend(
        _apply_debt_budget(
            wall_clock,
            wall_clock_debt,
            filename=filename,
            build_overrun_msg=_wall_clock_overrun_msg,
            build_slack_msg=_wall_clock_slack_msg,
        )
    )

    return sorted(violations, key=lambda v: (v.lineno, v.col_offset))


# The files a DIRECTORY argument expands to, recursively: test modules, conftest
# files and ``_*.py`` test-helper modules (task 5269).  Helpers hold mock factories
# and waits as surely as test modules do.  ``__init__.py`` matches ``_*.py`` and is
# deliberately included as harmless.  Explicit file arguments bypass this.
_DISCOVERY_GLOBS: tuple[str, ...] = ('test_*.py', 'conftest.py', '_*.py')


def discover_scan_targets(directory: Path) -> list[Path]:
    """Return every file under *directory* the checker scans, sorted.

    The one directory expansion: ``main`` hands it to ``run_cli`` and the
    baseline-integrity census in fused-memory/tests/test_check_bare_magicmock_config.py
    calls it, so the gate and its census cannot scan different file sets.
    """
    return discover_files(directory, _DISCOVERY_GLOBS)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Accepts file paths and/or directories.

    Runs all three rules — ``bare-magicmock``, ``bare-dataclass-double`` and
    ``wall-clock-deadline`` — in a single AST pass per file.

    For directories, recursively scans test_*.py, conftest.py and _*.py helper
    modules only (``discover_scan_targets``).  Output and the 0/1/2 exit ladder are
    ``_lint_cli.run_cli``'s.
    """
    return run_cli(
        argv,
        description=(
            'Test-quality lint checks over test files: bare MagicMock() assigned to '
            'config-named variables (bare-magicmock), unspecced MagicMocks shaped like '
            'a registered dataclass (bare-dataclass-double), and load-bearing waits '
            'carrying a wall-clock deadline (wall-clock-deadline).'
        ),
        discover=discover_scan_targets,
        find_violations=find_violations,
    )


if __name__ == '__main__':
    sys.exit(main())
