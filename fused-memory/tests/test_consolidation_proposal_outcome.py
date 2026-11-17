"""The consolidation-proposal row's lifecycle after it is proposed (task 5240).

``reconciliation/consolidation_proposal.py`` owns the row's identity, payload
keys and states, so the states an executor moves a row through, the outcome it
records, and the two pure readers over a topic's rows live there too. Every
test runs against a real ``ReconLedgerStore`` on ``tmp_path`` with rows written
by ``record_proposal``.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from fused_memory.reconciliation.consolidation_auto import AutoOutcome
from fused_memory.reconciliation.consolidation_proposal import (
    SHOWN_TEXT_CHARS,
    AdjudicationRecord,
    ConsolidationProposal,
    ProposalOutcome,
    ProposalRef,
    ProposalState,
    ShownMember,
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
            'dropped', 'unresolved',
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
    async def test_an_undecodable_row_can_still_be_refused(self, ledger):
        record = ReconLedgerRecord(
            project_id=PROJECT_ID,
            record_kind='consolidation_proposal',
            flag_type=TOPIC,
            run_id='run-x',
            payload_json='not json',
            state='proposed',
            created_at=NOW.isoformat(),
        )
        await ledger.upsert(record)
        outcome = _outcome(ProposalState.REFUSED, reason_codes=('proposal_undecodable',))

        assert await record_outcome(ledger, record, outcome) is True

        stored = await ledger.get_by_identity(
            project_id=PROJECT_ID,
            record_kind='consolidation_proposal',
            task_id='',
            flag_type=TOPIC,
            run_id='run-x',
        )
        assert stored.state == 'refused'
        assert outcome_from_record(stored) == outcome
        assert 'not json' in stored.payload_json

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



def _adjudication(**overrides):
    fields = {
        'decision': 'drop',
        'reason_code': 'not_same_lesson',
        'reason': 'r',
        'waivers': (('member_carries_correction_banner', MEMBERS[0]),),
        'claim_suggestion': None,
        'identity': 'i' * 64,
        'context_key': 'k' * 64,
        'resolved_model': 'gpt-x',
        'input_tokens': 10,
        'output_tokens': 5,
        'cost_usd': 0.0073,
        'latency_s': 5.4,
        'shown': (ShownMember(id=MEMBERS[0], content_sha256='b' * 64, text='t'),),
    }
    fields.update(overrides)
    return AdjudicationRecord(**fields)


def _adjudicated_outcome():
    """An outcome with every field the adjudicator flow adds set to a non-default."""
    return _outcome(
        ProposalState.DROPPED,
        members_key='a' * 64,
        kept_ids=MEMBERS[:1],
        repaired={'member_not_found': ('m9',)},
        effective_claim='c',
        effective_category='procedural_knowledge',
        reused_from=ProposalRef(topic=TOPIC, run_id='run-0'),
        adjudication=_adjudication(),
    )


#: The thirteen outcome keys a row written before the adjudicator carries.
_PRE_ADJUDICATOR_OUTCOME = {
    'state': 'refused',
    'acted_run_id': 'run-exec',
    'acted_at': '2026-07-01T01:00:00+00:00',
    'verdict': None,
    'reason_codes': ['mixed_category'],
    'predicate_version': None,
    'canonical_id': None,
    'canonical_minted': False,
    'tagged_ids': [],
    'tag_failure_ids': [],
    'stripped_ids': [],
    'gate_task_id': None,
    'closure_codes': [],
}


def _corrupt(path, value):
    """A to_payload() of the adjudicated outcome with one nested key set to *value*."""
    def corrupted():
        payload = _adjudicated_outcome().to_payload()
        target = payload
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        return payload
    return corrupted


class TestAdjudicatorFlowFields:
    """What the adjudicator flow records on the row (PRD consolidation-adjudicator K6)."""

    @pytest.mark.parametrize(
        'outcome',
        [_adjudicated_outcome(), _outcome(ProposalState.UNRESOLVED)],
        ids=['dropped_with_every_field', 'unresolved_with_defaults'],
    )
    @pytest.mark.asyncio
    async def test_every_new_field_round_trips(self, ledger, outcome):
        record = await _seed(ledger)

        assert await record_outcome(ledger, record, outcome) is True

        stored = await ledger.get_by_identity(**_proposal().ledger_key())
        assert stored.state == outcome.state.value
        assert outcome_from_record(stored) == outcome

    def test_a_row_written_before_the_adjudicator_decodes_with_defaults(self):
        record = ReconLedgerRecord(
            project_id=PROJECT_ID,
            record_kind='consolidation_proposal',
            flag_type=TOPIC,
            run_id='run-old',
            payload_json=json.dumps({
                'member_ids': list(MEMBERS),
                'category': 'procedural_knowledge',
                'claim': 'c',
                'rationale': 'r',
                'seeded_from': None,
                'outcome': _PRE_ADJUDICATOR_OUTCOME,
            }),
            state='refused',
            created_at=NOW.isoformat(),
        )

        outcome = outcome_from_record(record)

        assert outcome is not None
        assert outcome.reason_codes == ('mixed_category',)
        assert outcome.members_key is None
        assert outcome.kept_ids == ()
        assert dict(outcome.repaired) == {}
        assert outcome.effective_claim is None
        assert outcome.effective_category is None
        assert outcome.reused_from is None
        assert outcome.adjudication is None

    @pytest.mark.parametrize(
        ('corrupted', 'key'),
        [
            (_corrupt(('kept_ids',), 'abc'), 'kept_ids'),
            (_corrupt(('repaired',), {'member_not_found': 'm9'}), 'repaired'),
            (_corrupt(('reused_from',), {'topic': TOPIC}), 'reused_from'),
            (_corrupt(('adjudication', 'shown'), [{'id': 'x'}]), 'shown'),
            (_corrupt(('adjudication', 'input_tokens'), True), 'input_tokens'),
        ],
        ids=['kept_ids', 'repaired', 'reused_from', 'shown', 'input_tokens'],
    )
    def test_a_malformed_new_field_is_refused_loudly(self, corrupted, key):
        with pytest.raises(ValueError, match=key):
            ProposalOutcome.from_payload(corrupted())

    def test_repaired_cannot_be_changed_through_the_callers_dict(self):
        repaired = {'member_not_found': ('m9',)}
        outcome = _outcome(ProposalState.REFUSED, repaired=repaired)

        repaired['member_not_found'] = ('m8',)
        repaired['member_different_topic'] = ('m7',)

        assert dict(outcome.repaired) == {'member_not_found': ('m9',)}
        with pytest.raises(TypeError):
            outcome.repaired['x'] = ('y',)

    def test_shown_text_is_capped(self):
        assert SHOWN_TEXT_CHARS == 1500
        with pytest.raises(ValueError):
            ShownMember(id='m1', content_sha256='b' * 64, text='x' * (SHOWN_TEXT_CHARS + 1))

        cut = ShownMember.cut('m1', 'b' * 64, 'x' * 5000)

        assert len(cut.text) == SHOWN_TEXT_CHARS


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


#: When the retention tests' outcomes were acted on.
ACTED_AT = '2026-07-02T00:00:00+00:00'


class TestRetention:
    """A non-executed outcome lives ``retention`` past ``acted_at`` (PRD K6)."""

    @pytest.mark.parametrize(
        'state',
        [
            ProposalState.REFUSED,
            ProposalState.DROPPED,
            ProposalState.UNRESOLVED,
            ProposalState.DRY_RUN,
            ProposalState.REVERTED,
        ],
        ids=lambda state: state.value,
    )
    @pytest.mark.asyncio
    async def test_a_non_executed_outcome_expires_retention_after_acted_at(self, ledger, state):
        record = await _seed(ledger)

        await record_outcome(
            ledger, record, _outcome(state, acted_at=ACTED_AT), retention=timedelta(days=90),
        )

        stored = await ledger.get_by_identity(**_proposal().ledger_key())
        assert stored.expires_at == '2026-09-30T00:00:00+00:00'
        await ledger.gc(PROJECT_ID, now='2026-09-29T23:59:59+00:00', terminal_task_ids=[])
        assert await ledger.get_by_identity(**_proposal().ledger_key()) is not None
        await ledger.gc(PROJECT_ID, now='2026-09-30T00:00:01+00:00', terminal_task_ids=[])
        assert await ledger.get_by_identity(**_proposal().ledger_key()) is None

    @pytest.mark.asyncio
    async def test_an_executed_outcome_never_expires_whatever_the_retention(self, ledger):
        record = await _seed(ledger)

        await record_outcome(ledger, record, _executed_outcome(), retention=timedelta(days=90))

        stored = await ledger.get_by_identity(**_proposal().ledger_key())
        assert stored.expires_at is None
        await ledger.gc(PROJECT_ID, now=FAR_FUTURE, terminal_task_ids=[])
        assert await ledger.get_by_identity(**_proposal().ledger_key()) is not None

    @pytest.mark.asyncio
    async def test_an_acted_at_with_an_offset_is_stored_in_utc(self, ledger):
        """``gc`` compares TEXT, so the stored expiry must be canonical UTC."""
        record = await _seed(ledger)
        outcome = _outcome(ProposalState.REFUSED, acted_at='2026-07-02T02:00:00+02:00')

        await record_outcome(ledger, record, outcome, retention=timedelta(days=1))

        stored = await ledger.get_by_identity(**_proposal().ledger_key())
        assert stored.expires_at == '2026-07-03T00:00:00+00:00'

    @pytest.mark.parametrize(
        ('retention', 'acted_at'),
        [
            (timedelta(0), ACTED_AT),
            (timedelta(days=-1), ACTED_AT),
            (timedelta(days=1), '2026-07-02T00:00:00'),
        ],
        ids=['zero_retention', 'negative_retention', 'naive_acted_at'],
    )
    @pytest.mark.asyncio
    async def test_a_bad_retention_or_acted_at_is_refused(self, ledger, retention, acted_at):
        record = await _seed(ledger)
        outcome = _outcome(ProposalState.REFUSED, acted_at=acted_at)

        with pytest.raises(ValueError):
            await record_outcome(ledger, record, outcome, retention=retention)

        stored = await ledger.get_by_identity(**_proposal().ledger_key())
        assert stored.state == 'proposed'


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


def _acted_row(state, *, reused=False):
    """A row acted on in *state*; *reused* makes it a memo copy of an earlier row (PRD K4)."""
    outcome = ProposalOutcome(
        state=state,
        acted_run_id='r',
        acted_at='2026-07-01T00:00:00+00:00',
        reused_from=ProposalRef(topic=TOPIC, run_id='orig') if reused else None,
    )
    return dataclasses.replace(
        _row(state.value), payload_json=json.dumps({'outcome': outcome.to_payload()}),
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

    def test_a_reused_refusal_neither_counts_nor_resets(self):
        rows = [
            _acted_row(ProposalState.REFUSED),
            _acted_row(ProposalState.REFUSED, reused=True),
            _acted_row(ProposalState.REFUSED),
        ]
        assert refusal_streak(rows) == 2

    def test_a_reused_row_of_any_state_is_skipped(self):
        rows = [
            _acted_row(ProposalState.REFUSED),
            _acted_row(ProposalState.DROPPED, reused=True),
            _acted_row(ProposalState.UNRESOLVED, reused=True),
            _acted_row(ProposalState.REFUSED),
        ]
        assert refusal_streak(rows) == 2

    @pytest.mark.parametrize(
        'state', [ProposalState.DROPPED, ProposalState.UNRESOLVED], ids=lambda s: s.value,
    )
    def test_an_original_drop_or_unresolved_resets(self, state):
        rows = [
            _acted_row(ProposalState.REFUSED),
            _acted_row(state),
            _acted_row(ProposalState.REFUSED),
        ]
        assert refusal_streak(rows) == 1

    def test_a_row_whose_outcome_cannot_be_decoded_counts_as_an_original(self):
        undecodable = dataclasses.replace(
            _row('refused'), payload_json=json.dumps({'outcome': {'state': 'refused'}}),
        )
        rows = [undecodable, _acted_row(ProposalState.REFUSED)]
        assert refusal_streak(rows) == 2


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
    attribute = 'state'
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(outcome, attribute, ProposalState.ADDRESSED)
