"""Tests for scripts/sitting/gates.py — the pure six-gate carve-out evaluator (task 5376)."""
from __future__ import annotations

from dataclasses import replace

import pytest
from sitting import gates as mod

RULING = 'Leo 2026-09-01: esc-3881-3 option C — retarget task 3881 to the safe-A shape'


def _closeable_facts(**overrides) -> mod.CarveoutFacts:
    facts = mod.CarveoutFacts(
        escalation_id='esc-3881-3',
        ruling=mod.Fact(held=True, evidence=RULING, source_kind='task_description'),
        executed=mod.Fact(held=True, evidence='task 3881 description opens RETARGETED 2026-09-01'),
        session_terminated=mod.Fact(held=True, evidence='session unblock-df-3881-4242 status=done'),
        pin_declared_by=(),
        pins_recovery=(),
        root_cause='design-concern:3881:identity-seam',
        do_not_close_companions=(),
        sideways=mod.Fact(held=True, evidence='get_pending_escalations(task_id=3881): no twin L2'),
        members=(mod.MemberOutcome('esc-3881-2', 'pending', ''),),
    )
    return replace(facts, **overrides)


SINGLE_MISSES = {
    'ruling_is_leos_own': {'ruling': mod.Fact(held=True, evidence=RULING, source_kind='triage_note')},
    'ruling_names_this_record': {
        'ruling': mod.Fact(held=True, evidence='Leo: retarget 3881', source_kind='task_description'),
    },
    'ruling_was_executed': {'executed': mod.Fact(held=False, evidence='task 3881 unchanged')},
    'session_terminated': {'session_terminated': mod.UNKNOWN},
    'record_is_not_a_pin': {'pin_declared_by': ('leo',)},
    'sideways_check_ran': {'sideways': mod.UNKNOWN},
}


class TestAllGatesHold:
    def test_closeable_only_when_all_six_hold(self):
        verdict = mod.evaluate_carveout(_closeable_facts(), recommend_only=False)

        assert verdict.disposition == 'closeable'
        assert verdict.missed_gates == ()
        assert [g.number for g in verdict.gates] == [1, 2, 3, 4, 5, 6]
        assert all(g.held for g in verdict.gates)
        evidence = {g.name: g.evidence for g in verdict.gates}
        assert evidence['ruling_is_leos_own'] == RULING
        assert evidence['ruling_names_this_record'] == RULING
        assert evidence['ruling_was_executed'] == 'task 3881 description opens RETARGETED 2026-09-01'
        assert evidence['session_terminated'] == 'session unblock-df-3881-4242 status=done'
        assert evidence['sideways_check_ran'] == 'get_pending_escalations(task_id=3881): no twin L2'
        assert 'design-concern:3881:identity-seam' in evidence['record_is_not_a_pin']

    @pytest.mark.parametrize('gate', sorted(SINGLE_MISSES))
    def test_any_single_miss_is_report_only_naming_that_gate(self, gate):
        verdict = mod.evaluate_carveout(_closeable_facts(**SINGLE_MISSES[gate]), recommend_only=False)

        assert verdict.disposition == 'report_only'
        assert verdict.missed_gates == (gate,)

    def test_single_miss_table_covers_every_gate(self):
        assert set(SINGLE_MISSES) == set(mod.GATE_NAMES)


class TestFactInvariant:
    def test_held_fact_without_evidence_is_refused(self):
        with pytest.raises(ValueError):
            mod.Fact(held=True, evidence='')
        with pytest.raises(ValueError):
            mod.Fact(held=True, evidence='   ')

    def test_unheld_and_unknown_facts_need_no_evidence(self):
        assert mod.Fact(held=False, evidence='').held is False
        assert mod.UNKNOWN.held is None


class TestGateOneRejectsInferredAuthority:
    @pytest.mark.parametrize('kind', ['triage_note', 'suggested_action', 'agent_recommendation', ''])
    def test_inferred_sources_do_not_hold(self, kind):
        facts = _closeable_facts(ruling=mod.Fact(held=True, evidence=RULING, source_kind=kind))

        verdict = mod.evaluate_carveout(facts, recommend_only=False)

        assert 'ruling_is_leos_own' in verdict.missed_gates

    @pytest.mark.parametrize('kind', [
        'task_description', 'commit_message', 'memory_ruling_record', 'escalation_resolution',
        'task_metadata_ruling',
    ])
    def test_documented_sources_hold(self, kind):
        facts = _closeable_facts(ruling=mod.Fact(held=True, evidence=RULING, source_kind=kind))

        assert mod.evaluate_carveout(facts, recommend_only=False).disposition == 'closeable'


class TestGateTwoNamesThisRecord:
    def test_literal_id_in_ruling_text_holds_without_a_supplied_fact(self):
        verdict = mod.evaluate_carveout(_closeable_facts(), recommend_only=False)

        assert verdict.gate('ruling_names_this_record').held

    def test_a_longer_id_sharing_the_prefix_does_not_count(self):
        ruling = mod.Fact(held=True, evidence='Leo: esc-3881-31 option C', source_kind='task_description')

        verdict = mod.evaluate_carveout(_closeable_facts(ruling=ruling), recommend_only=False)

        assert verdict.missed_gates == ('ruling_names_this_record',)

    def test_a_supplied_fact_holds_when_the_text_does_not_name_the_id(self):
        ruling = mod.Fact(held=True, evidence='Leo: retarget 3881', source_kind='task_description')
        named = mod.Fact(held=True, evidence='the ruling answers esc-3881-3 question verbatim')

        verdict = mod.evaluate_carveout(
            _closeable_facts(ruling=ruling, names_this_record=named), recommend_only=False,
        )

        assert verdict.disposition == 'closeable'
        assert verdict.gate('ruling_names_this_record').evidence == named.evidence


class TestUnknownFailsClosed:
    def test_gates_three_and_four_default_to_unknown(self):
        bare = mod.CarveoutFacts(escalation_id='esc-1-1')

        assert bare.executed is mod.UNKNOWN
        assert bare.session_terminated is mod.UNKNOWN
        verdict = mod.evaluate_carveout(bare, recommend_only=False)
        assert {'ruling_was_executed', 'session_terminated'} <= set(verdict.missed_gates)


class TestGateFivePinProtection:
    @pytest.mark.parametrize('override', [
        {'pin_declared_by': ('leo',)},
        {'pins_recovery': ('task 3546 mu-gate specimen',)},
        {'pins_recovery': None},
        {'root_cause': 'veto-pin-do-not-close:3881'},
        {'do_not_close_companions': ('esc-3881-9',)},
        {'do_not_close_companions': None},
    ])
    def test_each_pin_signal_or_unknown_forces_report_only(self, override):
        verdict = mod.evaluate_carveout(_closeable_facts(**override), recommend_only=False)

        assert verdict.disposition == 'report_only'
        assert verdict.missed_gates == ('record_is_not_a_pin',)

    def test_pins_recovery_defaults_to_unknown(self):
        assert mod.CarveoutFacts(escalation_id='esc-1-1').pins_recovery is None

    def test_esc_3105_3_pin_evaluates_report_only(self):
        ruled = tuple(
            mod.MemberOutcome(f'esc-3105-{n}', 'dismissed', f'Leo ruled the mu-gate question ({n})')
            for n in range(10, 25)
        )
        pin = mod.CarveoutFacts(
            escalation_id='esc-3105-3',
            ruling=mod.Fact(held=True, evidence='esc-3105-3 ruled via member cascade',
                            source_kind='escalation_resolution'),
            executed=mod.Fact(held=True, evidence='task 3105 retargeted'),
            session_terminated=mod.Fact(held=True, evidence='no live session'),
            pin_declared_by=(),
            pins_recovery=(),
            root_cause='design-concern:3105:mu-gate',
            do_not_close_companions=('esc-3105-5',),
            sideways=mod.Fact(held=True, evidence='15/15 members ruled'),
            members=ruled,
        )

        verdict = mod.evaluate_carveout(pin, recommend_only=False)

        assert len(ruled) == 15
        assert verdict.disposition == 'report_only'
        assert verdict.missed_gates == ('record_is_not_a_pin',)
        assert 'esc-3105-5' in verdict.gate('record_is_not_a_pin').note


class TestGateSixReadsResolutionText:
    def test_dedup_marker_member_does_not_satisfy_the_sideways_check(self):
        members = (mod.MemberOutcome('esc-3881-2', 'dismissed', 'DUPLICATE of esc-3881-7 (survivor, stays open)'),)

        verdict = mod.evaluate_carveout(_closeable_facts(members=members), recommend_only=False)

        assert verdict.missed_gates == ('sideways_check_ran',)

    def test_substantive_member_ruling_satisfies_it(self):
        members = (mod.MemberOutcome('esc-3881-2', 'dismissed', 'Leo: option C, retarget 3881'),)

        verdict = mod.evaluate_carveout(_closeable_facts(members=members), recommend_only=False)

        assert verdict.disposition == 'closeable'

    @pytest.mark.parametrize(('status', 'resolution', 'expected'), [
        ('dismissed', 'DUPLICATE of esc-9-1 (survivor, stays open)', 'dedup_marker'),
        ('resolved', 'Leo ruled B', 'ruled'),
        ('dismissed', 'Leo ruled B', 'ruled'),
        ('pending', '', 'open'),
    ])
    def test_member_classification(self, status, resolution, expected):
        assert mod.classify_member(status, resolution) == expected


class TestRecommendOnly:
    def test_demotes_a_would_be_close_and_keeps_the_gate_results(self):
        facts = _closeable_facts()
        free = mod.evaluate_carveout(facts, recommend_only=False)

        demoted = mod.evaluate_carveout(facts, recommend_only=True)

        assert demoted.disposition == 'report_only'
        assert demoted.demoted_by == 'recommend-only'
        assert demoted.gates == free.gates
        assert demoted.all_held

    def test_does_not_mark_an_ordinary_miss_as_demoted(self):
        verdict = mod.evaluate_carveout(_closeable_facts(sideways=mod.UNKNOWN), recommend_only=True)

        assert verdict.demoted_by == ''
        assert not verdict.all_held


class TestPurity:
    def test_equal_facts_give_equal_verdicts(self):
        assert mod.evaluate_carveout(_closeable_facts(), recommend_only=False) == mod.evaluate_carveout(
            _closeable_facts(), recommend_only=False,
        )


class TestClosingEvidence:
    def test_renders_every_gate_verbatim_one_block_each(self):
        verdict = mod.evaluate_carveout(_closeable_facts(), recommend_only=False)

        text = mod.closing_evidence(verdict)

        blocks = text.split('\n\n')
        assert len(blocks) == 6
        for block, gate in zip(blocks, verdict.gates, strict=True):
            assert block.splitlines()[0].startswith(f'gate {gate.number} {gate.name}')
            assert gate.evidence in block

    def test_refuses_a_verdict_that_is_not_closeable(self):
        verdict = mod.evaluate_carveout(_closeable_facts(), recommend_only=True)

        with pytest.raises(ValueError):
            mod.closing_evidence(verdict)
