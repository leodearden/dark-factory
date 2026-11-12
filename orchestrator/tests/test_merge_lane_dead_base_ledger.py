"""The INV-3 dead-base ledger and its killer provenance (task 8523).

``DeadBaseLedger`` is the bounded record of merge commits that will never
reach main; each entry remembers which lane event first recorded it dead
(``DeadBaseKiller``), so a ``verdict_voided`` row can name what killed the
base it voided on.
"""
from __future__ import annotations

import pytest

from orchestrator.merge_lane.dead_base_ledger import (
    DeadBaseKiller,
    DeadBaseLedger,
    DeadBaseSite,
)

_HEAD = DeadBaseKiller(DeadBaseSite.HEAD_FAILED, '7')
_GATE = DeadBaseKiller(DeadBaseSite.GATE_REBASED, '42')


class TestBoundedFifo:
    def test_past_the_cap_the_oldest_is_evicted(self) -> None:
        ledger = DeadBaseLedger(cap=3)
        for sha in ('c0', 'c1', 'c2', 'c3'):
            ledger.record(sha, _HEAD)

        assert 'c0' not in ledger
        assert all(sha in ledger for sha in ('c1', 'c2', 'c3'))
        assert len(ledger) == 3

    def test_blank_shas_are_no_ops(self) -> None:
        ledger = DeadBaseLedger(cap=3)
        ledger.record('', _HEAD)
        ledger.record('   \n', _HEAD)
        assert len(ledger) == 0

    def test_a_rerecord_keeps_the_first_killer_and_position(self) -> None:
        ledger = DeadBaseLedger(cap=2)
        ledger.record('c0', _HEAD)
        ledger.record('c1', _HEAD)
        ledger.record('c0', _GATE)
        ledger.record('c2', _HEAD)

        assert ledger.killer_of('c1') == _HEAD
        assert 'c0' not in ledger, 'a re-record must not refresh FIFO position'
        ledger.record('c3', _GATE)
        ledger.record('c3', _HEAD)
        assert ledger.killer_of('c3') == _GATE

    def test_iteration_is_oldest_first(self) -> None:
        ledger = DeadBaseLedger(cap=5)
        for sha in ('c2', 'c0', 'c1'):
            ledger.record(sha, _HEAD)
        assert list(ledger) == ['c2', 'c0', 'c1']

    def test_a_cap_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match='cap'):
            DeadBaseLedger(cap=0)


class TestLookup:
    def test_killer_of_returns_the_recorded_killer(self) -> None:
        ledger = DeadBaseLedger(cap=3)
        ledger.record('c0', _GATE)
        assert ledger.killer_of('c0') == _GATE

    def test_killer_of_an_unknown_or_evicted_sha_is_none(self) -> None:
        ledger = DeadBaseLedger(cap=1)
        ledger.record('c0', _GATE)
        ledger.record('c1', _GATE)
        assert ledger.killer_of('c0') is None
        assert ledger.killer_of('never') is None

    def test_membership_and_record_strip_whitespace(self) -> None:
        ledger = DeadBaseLedger(cap=3)
        ledger.record(' c0\n', _HEAD)
        assert 'c0' in ledger
        assert ' c0 ' in ledger
        assert ledger.killer_of('c0\n') == _HEAD

    def test_an_empty_ledger_is_falsy(self) -> None:
        ledger = DeadBaseLedger(cap=3)
        assert not ledger
        ledger.record('c0', _HEAD)
        assert ledger


class TestKillerWire:
    def test_to_wire_is_plain_strings(self) -> None:
        wire = DeadBaseKiller(DeadBaseSite.GATE_REBASED, '42').to_wire()
        assert wire == {'site': 'gate_rebased', 'task_id': '42'}
        assert all(type(v) is str for v in wire.values())

    def test_site_vocabulary_is_closed(self) -> None:
        assert {site.value for site in DeadBaseSite} == {
            'head_failed', 'cascade_remerge', 'adoption_void', 'runner_unavailable',
            'gate_rebased', 'gate_reanchor', 'gate_unlanded', 'dispatch_void',
            'stale_discard', 'chain_released',
        }
