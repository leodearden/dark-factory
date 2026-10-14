"""The consolidation-proposal row's lifecycle after it is proposed (task 5240).

``reconciliation/consolidation_proposal.py`` owns the row's identity, payload
keys and states, so the states an executor moves a row through, the outcome it
records, and the two pure readers over a topic's rows live there too. Every
test runs against a real ``ReconLedgerStore`` on ``tmp_path`` with rows written
by ``record_proposal``.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest
import pytest_asyncio

from fused_memory.reconciliation.consolidation_auto import AutoOutcome
from fused_memory.reconciliation.consolidation_proposal import (
    ConsolidationProposal,
    ProposalOutcome,
    ProposalState,
    from_ledger_record,
    newest_per_topic,
    outcome_from_record,
    record_outcome,
    record_proposal,
    refusal_streak,
)
from fused_memory.reconciliation.recon_ledger import ReconLedgerRecord, ReconLedgerStore

PROJECT_ID = 'dark_factory'
TOPIC = 'memory-consolidation'
NOW = datetime(2026, 7, 1, tzinfo=UTC)
FAR_FUTURE = '2099-01-01T00:00:00+00:00'
MEMBERS = (
    '11111111-1111-4111-8111-111111111111',
    '22222222-2222-4222-8222-222222222222',
)


@pytest_asyncio.fixture
async def ledger(tmp_path):
    store = ReconLedgerStore(tmp_path / 'reconciliation.db')
    await store.initialize()
    yield store
    await store.close()


def _proposal(run_id='run-1', topic=TOPIC):
    return ConsolidationProposal(
        project_id=PROJECT_ID,
        run_id=run_id,
        topic=topic,
        member_ids=MEMBERS,
        category='procedural_knowledge',
        claim='The executor stamps a topic onto every retained member',
        rationale='Two records say the same thing.',
    )


async def _seed(ledger, run_id='run-1', topic=TOPIC):
    proposal = _proposal(run_id, topic)
    assert await record_proposal(ledger, proposal, now=NOW, ttl_hours=168)
    stored = await ledger.get_by_identity(**proposal.ledger_key())
    assert stored is not None
    return stored


def _outcome(state, **overrides):
    fields = {
        'state': state,
        'acted_run_id': 'run-exec',
        'acted_at': '2026-07-01T01:00:00+00:00',
    }
    fields.update(overrides)
    return ProposalOutcome(**fields)


def _executed_outcome(state=ProposalState.ADDRESSED):
    return _outcome(
        state,
        verdict=AutoOutcome.PASS,
        predicate_version='1',
        canonical_id='99999999-9999-4999-8999-999999999999',
        canonical_minted=True,
        tagged_ids=MEMBERS,
        closure_codes=('closed',),
    )


class TestStates:
    def test_the_lifecycle_states(self):
        assert {s.value for s in ProposalState} == {
            'proposed', 'addressed', 'dry_run', 'refused', 'gated', 'superseded', 'reverted',
        }


class TestRecordOutcome:
    @pytest.mark.asyncio
    async def test_moves_the_row_and_keeps_the_proposal_decodable(self, ledger):
        record = await _seed(ledger)
        outcome = _executed_outcome()

        assert await record_outcome(ledger, record, outcome) is True

        stored = await ledger.get_by_identity(**_proposal().ledger_key())
        assert stored is not None
        assert stored.state == 'addressed'
        assert from_ledger_record(stored) == _proposal()
        assert outcome_from_record(stored) == outcome

    @pytest.mark.asyncio
    async def test_every_field_round_trips(self, ledger):
        record = await _seed(ledger)
        outcome = _outcome(
            ProposalState.GATED,
            verdict=AutoOutcome.FAIL,
            reason_codes=('canonical_carries_correction', 'member_unreadable'),
            predicate_version='7',
            canonical_id='c-1',
            canonical_minted=False,
            tagged_ids=('a',),
            tag_failure_ids=('b',),
            stripped_ids=('c',),
            gate_task_id='77',
            closure_codes=('unstamped_cluster_member',),
        )
        await record_outcome(ledger, record, outcome)

        stored = await ledger.get_by_identity(**_proposal().ledger_key())
        assert outcome_from_record(stored) == outcome

    @pytest.mark.asyncio
    async def test_a_row_never_acted_on_has_no_outcome(self, ledger):
        record = await _seed(ledger)
        assert outcome_from_record(record) is None

    @pytest.mark.asyncio
    async def test_cas_refuses_a_row_moved_since_it_was_read(self, ledger):
        record = await _seed(ledger)
        assert await record_outcome(ledger, record, _outcome(ProposalState.REFUSED))

        stale_write = await record_outcome(ledger, record, _executed_outcome())

        assert stale_write is False
        stored = await ledger.get_by_identity(**_proposal().ledger_key())
        assert stored.state == 'refused'

    @pytest.mark.asyncio
    async def test_an_outcome_cannot_return_a_row_to_proposed(self, ledger):
        record = await _seed(ledger)
        with pytest.raises(ValueError, match='proposed'):
            await record_outcome(ledger, record, _outcome(ProposalState.PROPOSED))

    @pytest.mark.asyncio
    async def test_a_re_proposal_of_an_acted_row_is_still_refused(self, ledger):
        record = await _seed(ledger)
        await record_outcome(ledger, record, _executed_outcome())

        assert await record_proposal(ledger, _proposal(), now=NOW, ttl_hours=168) is False
        stored = await ledger.get_by_identity(**_proposal().ledger_key())
        assert stored.state == 'addressed'


class TestExpiry:
    @pytest.mark.parametrize('state', [ProposalState.ADDRESSED, ProposalState.GATED])
    @pytest.mark.asyncio
    async def test_an_executed_row_survives_gc(self, ledger, state):
        record = await _seed(ledger)
        await record_outcome(ledger, record, _executed_outcome(state))

        await ledger.gc(PROJECT_ID, now=FAR_FUTURE, terminal_task_ids=[])

        stored = await ledger.get_by_identity(**_proposal().ledger_key())
        assert stored is not None
        assert stored.expires_at is None

    @pytest.mark.parametrize(
        'outcome',
        [
            _outcome(ProposalState.REFUSED, reason_codes=('member_count_out_of_range',)),
            _outcome(ProposalState.DRY_RUN, verdict=AutoOutcome.PASS),
            _outcome(ProposalState.SUPERSEDED),
            _outcome(ProposalState.ADDRESSED, verdict=AutoOutcome.NOOP),
        ],
        ids=['refused', 'dry_run', 'superseded', 'noop_addressed'],
    )
    @pytest.mark.asyncio
    async def test_a_non_executed_row_keeps_its_ttl_and_is_reaped(self, ledger, outcome):
        record = await _seed(ledger)
        await record_outcome(ledger, record, outcome)

        stored = await ledger.get_by_identity(**_proposal().ledger_key())
        assert stored.expires_at == record.expires_at

        await ledger.gc(PROJECT_ID, now=FAR_FUTURE, terminal_task_ids=[])

        assert await ledger.get_by_identity(**_proposal().ledger_key()) is None


def _row(state, run_id='r', created_at='2026-07-01T00:00:00+00:00', topic=TOPIC):
    return ReconLedgerRecord(
        project_id=PROJECT_ID,
        record_kind='consolidation_proposal',
        flag_type=topic,
        run_id=run_id,
        payload_json='{}',
        state=state,
        created_at=created_at,
    )


class TestRefusalStreak:
    def test_trailing_refusals_after_an_addressed_row(self):
        rows = [
            _row('refused'), _row('refused'), _row('addressed'),
            _row('refused'), _row('refused'), _row('proposed'),
        ]
        assert refusal_streak(rows) == 2

    def test_all_refused(self):
        assert refusal_streak([_row('refused')] * 5) == 5

    def test_empty(self):
        assert refusal_streak([]) == 0

    def test_a_superseded_row_neither_counts_nor_resets(self):
        rows = [_row('refused'), _row('superseded'), _row('refused')]
        assert refusal_streak(rows) == 2

    def test_an_unknown_state_is_refused_loudly(self):
        with pytest.raises(ValueError, match='bogus'):
            refusal_streak([_row('bogus')])


class TestNewestPerTopic:
    def test_newest_created_at_wins_and_the_rest_are_superseded(self):
        old_a = _row('proposed', 'r1', '2026-07-01T00:00:00+00:00', 'topic-a')
        b = _row('proposed', 'r2', '2026-07-01T00:00:00+00:00', 'topic-b')
        new_a = _row('proposed', 'r3', '2026-07-02T00:00:00+00:00', 'topic-a')

        keep, superseded = newest_per_topic([old_a, b, new_a])

        assert keep == (b, new_a)
        assert superseded == (old_a,)

    def test_run_id_breaks_a_created_at_tie(self):
        run1 = _row('proposed', 'run-1')
        run2 = _row('proposed', 'run-2')

        keep, superseded = newest_per_topic([run2, run1])

        assert keep == (run2,)
        assert superseded == (run1,)

    def test_one_row_per_topic_is_kept_as_is(self):
        rows = [_row('proposed', 'r', topic=t) for t in ('x', 'y', 'z')]
        keep, superseded = newest_per_topic(rows)
        assert keep == tuple(rows)
        assert superseded == ()


def test_outcome_is_frozen():
    outcome = _outcome(ProposalState.REFUSED)
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.state = ProposalState.ADDRESSED  # type: ignore[misc]
