"""Repo-wide gate: every first-party module that constructs a ``UsageGate``
also tears one down (task 3288).

``UsageGate.shutdown`` is the only thing that cancels a gate's per-account
resume and auth-reprobe tasks, drains its fire-and-forget cost-event saves and
cleans its probe dirs, so a gate that is built and never shut down leaks all
of them for the life of its process. Two construction sites drifted into that
shape unnoticed, and a third appeared after the task naming them was filed,
so the pairing is enforced here rather than remembered.

THE RULE. A module that CONSTRUCTS a gate (a call whose callee is the name or
attribute ``UsageGate``) must REFERENCE a teardown: an attribute ``shutdown``
on a receiver whose source text has ``gate`` as a whole word between
non-alphanumerics (``self.usage_gate``, ``self._gate``; not ``self.delegate``).

* MODULE granularity: the reconciliation harness builds its gate in
  ``__init__`` and tears it down in ``run_loop``, which a per-function rule
  would flag.
* A REFERENCE, not a call, so a teardown handed over as a bare callable
  (``_run_shielded(..., curator_usage_gate.shutdown)``) counts.
* The receiver filter stops an unrelated ``server.shutdown()`` from blessing
  a module.

DELIBERATE LIMITS. The gate proves a teardown exists in the module, not that
it covers every exit path; behavioural tests at each site carry that half. An
aliased import (``from shared.usage_gate import UsageGate as G``, then
``G(cfg)``) is an accepted MISS: in a merge-gating whole-tree sweep a false
RED costs more than a miss.

A factory that RETURNS its gate transfers ownership rather than leaking, so it
is recorded in :data:`ALLOWLIST` with where its callers tear the gate down;
``test_allowlist_entries_are_live`` keeps each record falsifiable.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import pytest
from silent_fallthrough_scan import ParsedFile

_GATE_CLASS = 'UsageGate'
_RECEIVER_WORD_SEPARATOR = re.compile(r'[^a-z0-9]+')

ALLOWLIST: dict[str, str] = {
    'scripts/legibility/account_pool.py': (
        'build_pool is a factory: it returns the gate, so its callers own the '
        'teardown. scripts/legibility/session_runner.py shuts the gate down in '
        'its close path. scripts/sitting/nightly_prepare.py is a synchronous '
        'one-shot whose pool config (account_pool._pool_config) disables the '
        'resume and auth-reprobe loops, and its probe dirs ride task 3086\'s '
        'atexit hook (shared/config_dir.py).'
    ),
}


class GatePairing(NamedTuple):
    constructors: frozenset[str]
    unpaired: frozenset[str]


def _constructs_a_gate(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (isinstance(func, ast.Name) and func.id == _GATE_CLASS) or (
        isinstance(func, ast.Attribute) and func.attr == _GATE_CLASS
    )


def _references_a_gate_teardown(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == 'shutdown'
        and 'gate' in _RECEIVER_WORD_SEPARATOR.split(ast.unparse(node.value).lower())
    )


def scan_gate_pairing(records: Sequence[ParsedFile]) -> GatePairing:
    """Relpaths that construct a gate, and the subset with no teardown reference.

    Records whose source lacks ``UsageGate`` are skipped first, which is exact:
    the scan matches only identifiers, and an identifier is a literal substring
    of its file's source. A skipped file's syntax error is not this gate's
    concern; a retained file's is re-raised, so no construction site can hide
    behind an unparseable file.
    """
    constructors: set[str] = set()
    unpaired: set[str] = set()
    for record in records:
        if _GATE_CLASS not in record.source:
            continue
        if record.syntax_error is not None:
            raise record.syntax_error
        tree = record.tree
        assert tree is not None, (
            f'{record.relpath}: ParsedFile carries neither a tree nor a syntax_error'
        )
        nodes = list(ast.walk(tree))
        if not any(_constructs_a_gate(node) for node in nodes):
            continue
        constructors.add(record.relpath)
        if not any(_references_a_gate_teardown(node) for node in nodes):
            unpaired.add(record.relpath)
    return GatePairing(frozenset(constructors), frozenset(unpaired))


def _synthetic(source: str, relpath: str = 'pkg/mod.py') -> ParsedFile:
    try:
        tree, error = ast.parse(source), None
    except SyntaxError as exc:
        tree, error = None, exc
    return ParsedFile(Path('/synthetic') / relpath, relpath, source, tree, error)


def _scan_one(source: str) -> GatePairing:
    return scan_gate_pairing([_synthetic(source)])


class TestScanner:
    @pytest.mark.parametrize('construction', ['UsageGate(cfg)', 'usage_gate.UsageGate(cfg)'])
    def test_a_construction_with_no_teardown_is_unpaired(self, construction):
        assert _scan_one(f'gate = {construction}\n') == GatePairing(
            frozenset({'pkg/mod.py'}), frozenset({'pkg/mod.py'}),
        )

    def test_an_awaited_teardown_pairs(self):
        pairing = _scan_one(
            'async def run(cfg):\n'
            '    gate = UsageGate(cfg)\n'
            '    try:\n'
            '        await work(gate)\n'
            '    finally:\n'
            '        await gate.shutdown()\n'
        )
        assert pairing == GatePairing(frozenset({'pkg/mod.py'}), frozenset())

    def test_a_teardown_in_another_method_pairs(self):
        pairing = _scan_one(
            'class Owner:\n'
            '    def __init__(self, cfg):\n'
            '        self.usage_gate = UsageGate(cfg)\n'
            '    async def close(self):\n'
            '        await self.usage_gate.shutdown()\n'
        )
        assert pairing.unpaired == frozenset()

    @pytest.mark.parametrize(
        'receiver', ['server', 'self.delegate', 'self.aggregates', 'navigator_gateway'],
    )
    def test_an_unrelated_shutdown_does_not_pair(self, receiver):
        pairing = _scan_one(
            'async def run(self, cfg, server, navigator_gateway):\n'
            '    gate = UsageGate(cfg)\n'
            f'    await {receiver}.shutdown()\n'
        )
        assert pairing.unpaired == frozenset({'pkg/mod.py'})

    @pytest.mark.parametrize('receiver', ['self._gate', 'curator_usage_gate', 'self._get_gate()'])
    def test_a_receiver_naming_gate_as_a_word_pairs(self, receiver):
        pairing = _scan_one(
            'async def run(self, cfg, curator_usage_gate):\n'
            '    self._gate = UsageGate(cfg)\n'
            f'    await {receiver}.shutdown()\n'
        )
        assert pairing.unpaired == frozenset()

    def test_a_teardown_passed_as_a_callable_pairs(self):
        pairing = _scan_one(
            'async def run(cfg):\n'
            '    gate = UsageGate(cfg)\n'
            '    await shielded("gate.shutdown", gate.shutdown)\n'
        )
        assert pairing.unpaired == frozenset()

    def test_the_class_definition_is_not_a_construction(self):
        assert _scan_one('class UsageGate:\n    async def shutdown(self): ...\n') == GatePairing(
            frozenset(), frozenset(),
        )

    def test_an_unparseable_file_that_never_names_the_class_is_skipped(self):
        assert _scan_one('def broken(:\n') == GatePairing(frozenset(), frozenset())

    def test_an_unparseable_file_that_names_the_class_raises(self):
        with pytest.raises(SyntaxError):
            _scan_one('gate = UsageGate(\n')


@pytest.fixture(scope='module')
def pairing(first_party_tree: Sequence[ParsedFile]) -> GatePairing:
    return scan_gate_pairing(first_party_tree)


def test_every_gate_constructor_tears_down(pairing):
    offenders = sorted(pairing.unpaired - set(ALLOWLIST))
    assert not offenders, (
        'These modules construct a UsageGate but never reference a gate '
        'teardown (`<gate>.shutdown`):\n'
        + '\n'.join(f'  {relpath}' for relpath in offenders)
        + '\n\nA gate that is never shut down leaks its account resume and '
          'auth-reprobe tasks, its background cost-event saves and its probe '
          'dirs for the life of the process. Tear it down best-effort in a '
          'try/finally around its use; an eval campaign should own it through '
          'orchestrator/src/orchestrator/evals/runner.py::campaign_usage_gate. '
          'A factory that returns the gate to callers who tear it down belongs '
          'in ALLOWLIST, with a justification naming those callers.'
    )


def test_allowlist_entries_are_live(pairing):
    stale = sorted(set(ALLOWLIST) - pairing.unpaired)
    assert not stale, (
        'Stale ALLOWLIST entries — these modules no longer construct a gate '
        'without a teardown:\n'
        + '\n'.join(f'  {relpath}' for relpath in stale)
        + '\n\nThe site moved, stopped constructing, or now tears its gate down '
          'itself. Delete the entry rather than leaving a record of code that '
          'no longer exists.'
    )


class TestGateSelfIntegrity:
    """The gate must not pass vacuously."""

    def test_the_scan_finds_the_known_constructors(self, pairing):
        assert len(pairing.constructors) >= 5, (
            f'Only {len(pairing.constructors)} UsageGate-constructing module(s) '
            f'found: {sorted(pairing.constructors)}. At least five are known, so '
            'a trip means the SCAN is broken (enumeration, prefilter or matcher), '
            'not that the tree lost its gates.'
        )

    def test_the_teardown_detector_fires_on_real_code(self, pairing):
        controls = {
            'orchestrator/src/orchestrator/harness.py',
            'fused-memory/src/fused_memory/server/main.py',
        }
        assert controls <= pairing.constructors, (
            f'Positive controls missing from the constructor set: '
            f'{sorted(controls - pairing.constructors)}. Both build a UsageGate '
            'today; if one moved, re-point the control at its new home.'
        )
        assert not controls & pairing.unpaired, (
            f'Positive controls read as unpaired: {sorted(controls & pairing.unpaired)}. '
            'Both tear their gate down today, so the teardown detector is broken '
            'or the teardown was removed.'
        )
