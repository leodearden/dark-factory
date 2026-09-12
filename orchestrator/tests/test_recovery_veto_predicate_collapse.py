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

#: THE ALLOWLIST.  Each entry is a carve-out this task decided to KEEP separate,
#: keyed on the enclosing function plus a source excerpt — never on a line
#: number, which would rot on the next edit above it.
#:
#: `comment_marker` is the text the site must carry in-code, so a later reader
#: (and this test) finds the carve-out DOCUMENTED rather than merely tolerated.
_ALLOWLIST = (
    {
        'module': 'harness.py',
        'function': '_reconcile_one_stranded',
        'why': (
            'the re-file DEDUP guard: it asks "would I be stacking a SECOND '
            'record?", for which ANY open record — info or dead-L0 included — '
            'is the right answer.  It is reached only AFTER the shared '
            'predicate has already let the caller through.'
        ),
        'excerpt': 'Re-filing would stack a SECOND stranded_blocked L1',
        'comment_marker': 'task 3541',
    },
)


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


def _bare_collection_tests(name: str) -> list[tuple[int, str, str]]:
    """Every ``if <a-collection-of-records>:`` in *name*.

    Returns ``(lineno, function, rendered_test)``.  Deliberately narrow: only a
    BARE truthiness test counts.  A `not ...`, a comparison, or a call to the
    shared predicate is a considered answer, not a re-derivation.
    """
    source = _module_source(name)
    tree = ast.parse(source)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if isinstance(test, ast.Attribute) and test.attr in _RECORD_COLLECTIONS:
            rendered = f'{ast.unparse(test)}'
        elif isinstance(test, ast.Name) and test.id in _RECORD_COLLECTIONS:
            rendered = test.id
        else:
            continue
        found.append((node.lineno, _enclosing_function(tree, node), rendered))
    return found


def _source_around(name: str, lineno: int, *, before: int = 25) -> str:
    lines = _module_source(name).splitlines()
    return '\n'.join(lines[max(0, lineno - 1 - before):lineno])


def _is_allowlisted(module: str, function: str, lineno: int) -> dict | None:
    context = _source_around(module, lineno)
    for entry in _ALLOWLIST:
        if (
            entry['module'] == module
            and entry['function'] == function
            and entry['excerpt'] in context
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
            for lineno, function, rendered in _bare_collection_tests(module):
                if _is_allowlisted(module, function, lineno) is None:
                    offenders.append(f'{module}:{lineno} in {function}(): if {rendered}:')

        assert not offenders, (
            'a recovery/redispatch site re-derives the escalation-pin '
            'predicate locally instead of consuming '
            'orchestrator.recovery_pins (INV-5, PRD D3, spec E7). '
            'New site(s):\n  ' + '\n  '.join(offenders) + '\n'
            'Consume records_pin_recovery / records_pin_blocked_recovery (or '
            'the report-shaped adapters in task_ground_truth), or add an '
            'entry to _ALLOWLIST in this file explaining why this site asks a '
            'genuinely different question.'
        )

    @pytest.mark.parametrize('entry', _ALLOWLIST, ids=lambda e: e['function'])
    def test_every_allowlist_entry_still_exists(self, entry: dict) -> None:
        """An allowlist that outlives its site silently stops guarding."""
        matches = [
            (lineno, function)
            for lineno, function, _ in _bare_collection_tests(entry['module'])
            if function == entry['function']
            and _is_allowlisted(entry['module'], function, lineno) is entry
        ]
        assert len(matches) == 1, (
            f"allowlist entry for {entry['module']}::{entry['function']} matches "
            f'{len(matches)} sites — remove it if the carve-out is gone, or '
            f'tighten its excerpt if it now matches more than one'
        )

    @pytest.mark.parametrize('entry', _ALLOWLIST, ids=lambda e: e['function'])
    def test_every_carve_out_names_this_tasks_decision_in_code(
        self, entry: dict,
    ) -> None:
        """DOCUMENTED, not merely tolerated.

        Without the in-code note this site reads to a grep exactly like the
        copies eta deleted, and the next reader has to re-derive whether it was
        missed work or a decision.
        """
        lineno = _bare_collection_tests(entry['module'])[0][0]
        for ln, function, _ in _bare_collection_tests(entry['module']):
            if function == entry['function']:
                lineno = ln
                break
        context = _source_around(entry['module'], lineno)
        assert entry['comment_marker'] in context, (
            f"{entry['module']}::{entry['function']}'s carve-out must name task "
            f"3541's decision in-code, so a reader finds it decided rather "
            f'than missed.  Reason on record: {entry["why"]}'
        )


class TestTheArchiveInclusiveGateCheckIsDocumented:
    """PRD D3's OTHER named carve-out — a call, so the AST scan cannot see it.

    The deterministic gate check asks "did a human already ACT?", which is why
    it is archive-inclusive (a RESOLVED record counts) where every pin
    predicate reads only OPEN records.  It is not a `bool(open)` copy, so it
    would never trip the guard above — which is exactly why its carve-out has
    to be asserted separately rather than assumed covered.
    """

    #: The gate check's OWN call — the two-argument, `status`-less form.
    #: `agent_role=DETERMINISTIC_AGENT_ROLE` alone appears more than once in
    #: harness.py, so anchoring on that would find an unrelated site.
    _MARKER = 'tid, agent_role=DETERMINISTIC_AGENT_ROLE,'

    def test_the_gate_check_is_still_archive_inclusive(self) -> None:
        source = _module_source('harness.py')
        assert self._MARKER in source
        index = source.index(self._MARKER)
        window = source[max(0, index - 2500):index + 400]
        assert "status='pending'" not in window.split(self._MARKER)[-1][:200], (
            'narrowing the gate read to pending would make a human-resolved '
            'gate look like a fresh strand'
        )

    def test_the_gate_check_names_this_tasks_decision(self) -> None:
        source = _module_source('harness.py')
        index = source.index(self._MARKER)
        window = source[max(0, index - 2500):index]
        assert 'task 3541' in window or '3541' in window, (
            "the gate check must carry a comment naming task 3541's decision "
            'that it STAYS separate, and pointing at the producer-side '
            'boundary (workflow.py::_is_gating_escalation, owned by task 5222)'
        )

    def test_the_producer_side_boundary_is_recorded_harness_side(self) -> None:
        """task 5222's ownership is documented WITHOUT reaching into workflow.py.

        The 2026-09-08 amendment dropped `workflow.py` from this task's scope:
        a comment there would assert a permanence 5222 is chartered to
        overturn, and would conflict textually with 5222's own edit.  The
        boundary is stated here instead.
        """
        source = _module_source('harness.py')
        index = source.index(self._MARKER)
        window = source[max(0, index - 2500):index]
        assert '_is_gating_escalation' in window and '5222' in window, (
            'the harness-side carve-out comment is where this task records '
            'that the producer-side predicate stays untouched and who owns it'
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
