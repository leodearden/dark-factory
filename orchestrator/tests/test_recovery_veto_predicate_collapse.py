"""The single-predicate invariant, as an executable guard (task 3541 / INV-5).

PRD `plans/task-escalation-state-graph-prd.md` D3; spec
`docs/task-escalation-state-spec.md` E7.

E7 catalogued FIVE hand-rolled copies of "does an open escalation pin this
task?", spread across the resolver, two harness sweeps, the scheduler and the
deterministic-recon pair, which could — and did — disagree about level,
severity and store-unavailability.  Task eta collapsed them onto
`escalation.pins.classify_pins`, reached through `orchestrator.recovery_pins`.

That collapse is only worth as much as its permanence, and a comment cannot
enforce permanence.  This module parses the three modules that own recovery
policy and fails if a new local re-derivation appears — naming it, so the
regression explains itself.

SCOPE, deliberate on both edges:

* Exactly `task_ground_truth.py`, `harness.py` and `scheduler.py`.  These are
  where recovery/redispatch dispositions are decided.
* NOT `workflow.py`.  Its two producer-side predicates are outside this task
  per the ratified 2026-09-08 amendment (task 5222 owns narrowing them), and
  `_is_gating_escalation` is a per-record predicate over a single `Escalation`
  rather than a collection truthiness — so it could not trip this guard even
  if scanned, and asserting on its comments would only manufacture a conflict
  with 5222.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from typing import NamedTuple

import pytest

_SRC = Path(__file__).parent.parent / 'src' / 'orchestrator'

#: The three modules that decide recovery/redispatch dispositions.
_SCANNED = ('task_ground_truth.py', 'harness.py', 'scheduler.py')

#: Identifiers that name a COLLECTION of open escalation records.  Testing one
#: of these for truthiness is the exact shape of the five copies E7 catalogued.
_RECORD_COLLECTIONS = frozenset({
    'open_escalations', 'rows', 'records', '_dedup_rows', '_deploy_rows',
    'pending', 'escalations', 'open_records',
})

#: THE ALLOWLIST — carve-outs that ask a genuinely different question and so
#: keep a bare truthiness test.  EMPTY is the healthy state: every entry is a
#: site that reads, to a grep, exactly like the copies task eta deleted.
#:
#: Keyed on `module` + `function` + `body`, where `body` is the tuple of
#: `ast.unparse`d statements the guard's `if` actually executes.  Deliberately
#: EXECUTABLE STRUCTURE, never prose: an earlier version keyed on a source
#: excerpt and required a comment marker, which meant a comment rewrite could
#: turn this guard into a false-offender report, and a function-name-only key
#: could silently allowlist a NEW bare test elsewhere in the same 700-line
#: function.  The rationale for each carve-out belongs in the source comment at
#: the site, once — not duplicated into an assertion here.
#: EMPTY as of task 3541's review pass: the last carve-out — the re-file dedup
#: guard — was given its own shared predicate
#: (`recovery_pins.records_would_duplicate_a_handoff`) rather than kept as a
#: bare truthiness test.  The matcher and its synthetic-entry coverage stay for
#: the next genuine carve-out.
_ALLOWLIST: tuple[dict, ...] = ()


def _module_source(name: str) -> str:
    return (_SRC / name).read_text()


def _enclosing_function(tree: ast.AST, target: ast.If) -> str:
    """The name of the innermost function containing *target*, or ``'<module>'``."""
    best = '<module>'
    best_span = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        end = getattr(node, 'end_lineno', None)
        if end is None or not (node.lineno <= target.lineno <= end):
            continue
        span = end - node.lineno
        if best_span is None or span < best_span:
            best, best_span = node.name, span
    return best


def _gate_check_calls(function: str) -> list[tuple[int, frozenset[str]]]:
    """Every role-scoped `get_by_task` inside *function*, as (line, keywords).

    Structural rather than textual: the gate check is identified by WHAT IT
    CALLS (`get_by_task`), WHERE (inside *function*), and the one keyword that
    distinguishes it from an unrelated read (`agent_role`) — so line wrapping,
    argument order and whitespace are all free to change.
    """
    tree = ast.parse(_module_source('harness.py'))
    enclosing = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name == function
    )
    return [
        (node.lineno, frozenset(kw.arg for kw in node.keywords if kw.arg))
        for node in ast.walk(enclosing)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'get_by_task'
        and any(
            kw.arg == 'agent_role'
            and ast.unparse(kw.value) == 'DETERMINISTIC_AGENT_ROLE'
            for kw in node.keywords
        )
    ]


class _BareTest(NamedTuple):
    """One ``if <a-collection-of-records>:`` found in a scanned module."""

    lineno: int
    function: str
    rendered: str
    #: The ``ast.unparse``d statements the guard executes — the structural key
    #: `_ALLOWLIST` matches on, so a carve-out survives a comment rewrite and
    #: a NEW bare test in the same function is still reported.
    body: tuple[str, ...]


def _bare_collection_tests(name: str) -> list[_BareTest]:
    """Every ``if <a-collection-of-records>:`` in *name*.

    Deliberately narrow: only a BARE truthiness test counts.  A `not ...`, a
    comparison, or a call to the shared predicate is a considered answer, not a
    re-derivation.
    """
    tree = ast.parse(_module_source(name))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if isinstance(test, ast.Attribute) and test.attr in _RECORD_COLLECTIONS:
            rendered = ast.unparse(test)
        elif isinstance(test, ast.Name) and test.id in _RECORD_COLLECTIONS:
            rendered = test.id
        else:
            continue
        found.append(_BareTest(
            node.lineno,
            _enclosing_function(tree, node),
            rendered,
            tuple(ast.unparse(stmt) for stmt in node.body),
        ))
    return found


def _is_allowlisted(
    module: str, function: str, body: tuple[str, ...],
    *, allowlist: tuple[dict, ...] = _ALLOWLIST,
) -> dict | None:
    """The matching carve-out entry, or ``None``.

    Matches on what the guard DOES, not on what a comment beside it says.
    """
    for entry in allowlist:
        if (
            entry['module'] == module
            and entry['function'] == function
            and tuple(entry['body']) == body
        ):
            return entry
    return None


class TestNoSiteReDerivesTheVetoLocally:
    """The grep-provable signal, executable.

    Before task eta each of these modules decided for itself whether an open
    escalation pinned a task.  Now exactly one bare truthiness test survives,
    and it is a DEDUP rather than a veto.
    """

    def test_every_bare_collection_test_is_an_allowlisted_carve_out(self) -> None:
        offenders = []
        for module in _SCANNED:
            for found in _bare_collection_tests(module):
                if _is_allowlisted(module, found.function, found.body) is None:
                    offenders.append(
                        f'{module}:{found.lineno} in {found.function}(): '
                        f'if {found.rendered}:'
                    )

        assert not offenders, (
            'a recovery/redispatch site re-derives the escalation-pin '
            'predicate locally instead of consuming '
            'orchestrator.recovery_pins (INV-5, PRD D3, spec E7). '
            'New site(s):\n  ' + '\n  '.join(offenders) + '\n'
            'Consume the shared predicates in orchestrator.recovery_pins (or '
            'the report-shaped adapters in task_ground_truth), or add an '
            'entry to _ALLOWLIST in this file explaining why this site asks a '
            'genuinely different question.'
        )

    def test_every_allowlist_entry_still_matches_exactly_one_site(self) -> None:
        """An allowlist that outlives its site silently stops guarding.

        Loops internally rather than parametrizing, so it degrades to a
        passing no-op when the allowlist is empty — which is the healthy state
        — instead of erroring on an empty parameter set.
        """
        for entry in _ALLOWLIST:
            matches = [
                found.lineno
                for found in _bare_collection_tests(entry['module'])
                if _is_allowlisted(entry['module'], found.function, found.body)
                is entry
            ]
            assert len(matches) == 1, (
                f"allowlist entry for {entry['module']}::{entry['function']} "
                f'matches {len(matches)} sites (lines {matches}) — remove it '
                f'if the carve-out is gone, or tighten its body shape if it '
                f'now matches more than one'
            )


class TestTheGateCheckStaysArchiveInclusive:
    """PRD D3's OTHER named carve-out — a call, so the AST scan cannot see it.

    The deterministic gate check asks "did a human already ACT?", which is why
    its read is archive-INCLUSIVE (a RESOLVED record counts) where every pin
    predicate reads only OPEN records.  That `status`-less read is the
    BEHAVIOUR that makes it a different question; narrowing it to
    `status='pending'` would make a human-resolved gate look like a fresh
    strand.  The rationale lives in the source comment at the site — asserting
    that prose here would duplicate it and pin wording task 5222 must be free
    to edit.
    """

    #: The sweep that owns the gate check.  `agent_role=DETERMINISTIC_AGENT_ROLE`
    #: alone appears more than once in harness.py, so the enclosing function is
    #: what makes the anchor unambiguous.
    _SWEEP = '_run_deterministic_recon_sweep'

    def test_the_gate_check_is_still_archive_inclusive(self) -> None:
        """Keyed on the parsed CALL, never on its source text.

        An earlier version matched the raw substring
        `'tid, agent_role=DETERMINISTIC_AGENT_ROLE,'`, which existed only
        because the call happened to wrap that way — so re-wrapping it, a
        formatting-only edit, failed this test with a message about archive
        inclusivity.  The keyword set is the behaviour; the line breaks are not.
        """
        calls = _gate_check_calls(self._SWEEP)

        assert len(calls) == 1, (
            f'expected exactly one role-scoped gate read in {self._SWEEP}(), '
            f'found {len(calls)} (lines {[line for line, _ in calls]})'
        )
        _, keywords = calls[0]
        assert 'status' not in keywords, (
            'narrowing the gate read to a status would make a human-resolved '
            'gate look like a fresh strand — it asks "did a human already '
            'ACT?", so its read stays archive-INCLUSIVE'
        )



class TestEveryModuleReachesTheSharedClassifier:
    """Positively: each scanned module consumes the shared predicate."""

    @pytest.mark.parametrize(
        ('module', 'expected'),
        [
            ('task_ground_truth.py', ('escalation.pins', 'orchestrator.recovery_pins')),
            ('harness.py', ('orchestrator.recovery_pins',)),
            ('scheduler.py', ('orchestrator.recovery_pins',)),
        ],
    )
    def test_module_imports_the_shared_predicate(
        self, module: str, expected: tuple[str, ...],
    ) -> None:
        tree = ast.parse(_module_source(module))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        imported |= {
            f'{node.module}.{alias.name}'
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
            for alias in node.names
        }
        for want in expected:
            assert any(
                name == want or name.startswith(want + '.') for name in imported
            ), f'{module} must reach the shared pin predicate via {want}'

    def test_the_shared_predicate_is_one_object_everywhere(self) -> None:
        from orchestrator import recovery_pins, scheduler
        from orchestrator import task_ground_truth as tgt

        shared = recovery_pins.records_pin_blocked_recovery
        assert scheduler.records_pin_blocked_recovery is shared
        assert tgt.records_pin_blocked_recovery is shared
        assert tgt.records_pin_recovery is recovery_pins.records_pin_recovery


class TestTheGuardItselfCannotSilentlyStopGuarding:
    """A scanner that finds nothing may be working — or may be broken."""

    def test_the_scanner_detects_a_planted_re_derivation(self) -> None:
        planted = textwrap.dedent('''
            def _fake_sweep(report):
                if report.open_escalations:
                    return None
                return 'acted'
        ''')
        tree = ast.parse(planted)
        hits = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Attribute)
            and node.test.attr in _RECORD_COLLECTIONS
        ]
        assert len(hits) == 1, (
            'the AST shape this guard matches must still be the shape a '
            'hand-rolled veto actually has'
        )

    def test_the_scanned_modules_all_parse(self) -> None:
        for module in _SCANNED:
            assert ast.parse(_module_source(module)) is not None

    #: A carve-out that does NOT exist in the tree — the mechanism has to stay
    #: covered once the real allowlist empties, or the next genuine carve-out
    #: would be added to an untested matcher.
    _SYNTHETIC = (
        {'module': 'harness.py', 'function': '_fake', 'body': ('return None',)},
    )

    def test_the_matcher_accepts_an_entry_whose_body_shape_agrees(self) -> None:
        assert _is_allowlisted(
            'harness.py', '_fake', ('return None',), allowlist=self._SYNTHETIC,
        ) is self._SYNTHETIC[0]

    @pytest.mark.parametrize(
        ('module', 'function', 'body'),
        [
            pytest.param('scheduler.py', '_fake', ('return None',), id='wrong-module'),
            pytest.param('harness.py', '_other', ('return None',), id='wrong-function'),
            pytest.param('harness.py', '_fake', ('continue',), id='different-body'),
            pytest.param(
                'harness.py', '_fake', ('return None', 'x = 1'), id='body-grew',
            ),
        ],
    )
    def test_the_matcher_rejects_everything_else(
        self, module: str, function: str, body: tuple[str, ...],
    ) -> None:
        """Structural, not fuzzy.

        `body-grew` is the case a prose key could never catch: a new statement
        inside an allowlisted guard is a DIFFERENT guard, and must be reported
        rather than inherited.
        """
        assert _is_allowlisted(
            module, function, body, allowlist=self._SYNTHETIC,
        ) is None

    def test_an_empty_allowlist_matches_nothing(self) -> None:
        """The healthy end state still exercises the matcher."""
        assert _is_allowlisted(
            'harness.py', '_fake', ('return None',), allowlist=(),
        ) is None
