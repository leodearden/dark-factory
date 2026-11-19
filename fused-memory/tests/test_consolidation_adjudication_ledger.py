"""The ledger shape the adjudicator flow keeps beside the proposal row.

PRD ``plans/consolidation-adjudicator-prd.md`` K4 (the members key a memo and a
bounded retry are counted against), K5 (the calibration record) and K6 (the
content hash a shown member carries). Member records are plain
``{'id', 'content', 'metadata'}`` dicts, the shape
``MemoryService.get_memory_by_id`` returns.
"""

from __future__ import annotations

import hashlib
import re

import pytest
import pytest_asyncio

from fused_memory.reconciliation.consolidation_auto import UNREADABLE
from fused_memory.reconciliation.consolidation_proposal import (
    RECORD_KIND_ADJUDICATION_ATTEMPT,
    RECORD_KIND_ADJUDICATOR_CALIBRATION,
    adjudication_attempt_key,
    adjudicator_calibration_key,
    content_sha256,
    members_key,
)
from fused_memory.reconciliation.recon_ledger import (
    SERVICE_SCOPE_PROJECT_ID,
    ReconLedgerRecord,
    ReconLedgerStore,
)
from fused_memory.utils.validation import validate_project_id

CREATED_AT = '2026-07-01T00:00:00+00:00'
PAST = '2026-07-02T00:00:00+00:00'
FAR_FUTURE = '2099-01-01T00:00:00+00:00'
MEMBERS_KEY = 'a' * 64
IDENTITY = 'i' * 64


@pytest_asyncio.fixture
async def ledger(tmp_path):
    store = ReconLedgerStore(tmp_path / 'reconciliation.db')
    await store.initialize()
    yield store
    await store.close()


def _record(member_id, content='body', **metadata):
    return {'id': member_id, 'content': content, 'metadata': metadata}


def _reads(*records):
    return {record['id']: record for record in records}


class TestMembersKey:
    def test_content_sha256_is_the_plain_hash_of_the_content(self):
        content = 'The scheduler drains its queue — before the watchdog fires.'

        assert content_sha256(_record('a', content)) == hashlib.sha256(
            content.encode('utf-8'),
        ).hexdigest()
        assert content_sha256(None) == 'missing'
        assert content_sha256(UNREADABLE) == 'unreadable'
        assert content_sha256(['not', 'a', 'record']) == 'unreadable'
        assert content_sha256({'id': 'a', 'content': None, 'metadata': {}}) == 'unreadable'

    def test_the_key_ignores_order(self):
        reads = _reads(_record('a', 'x'), _record('b', 'y'))

        key = members_key(('a', 'b'), reads)

        assert key == members_key(('b', 'a'), reads)
        assert re.fullmatch(r'[0-9a-f]{64}', key)

    def test_a_content_change_changes_the_key(self):
        before = _reads(_record('a', 'x'), _record('b', 'y'))
        after = _reads(_record('a', 'x'), _record('b', 'y, edited'))

        assert members_key(('a', 'b'), before) != members_key(('a', 'b'), after)

    def test_a_metadata_change_does_not(self):
        """Re-categorising or re-topicing a member keeps its key: members and content only."""
        before = _reads(
            _record('a', 'x', category='procedural_knowledge'),
            _record('b', 'y'),
        )
        after = _reads(
            _record('a', 'x', category='observations_and_summaries', topic='other-topic'),
            _record('b', 'y'),
        )

        assert members_key(('a', 'b'), before) == members_key(('a', 'b'), after)

    def test_a_missing_member_hashes_as_missing(self):
        read_as_none = {'a': _record('a', 'x'), 'b': None}
        not_read = {'a': _record('a', 'x')}
        present = _reads(_record('a', 'x'), _record('b', 'y'))

        key = members_key(('a', 'b'), read_as_none)

        assert key == members_key(('a', 'b'), not_read)
        assert key != members_key(('a', 'b'), present)

    def test_unreadable_is_not_missing(self):
        unreadable = {'a': _record('a', 'x'), 'b': UNREADABLE}
        missing = {'a': _record('a', 'x'), 'b': None}

        assert members_key(('a', 'b'), unreadable) != members_key(('a', 'b'), missing)

    def test_a_repeated_id_is_part_of_the_key(self):
        reads = _reads(_record('a', 'x'), _record('b', 'y'))

        assert members_key(('a', 'a', 'b'), reads) != members_key(('a', 'b'), reads)


class TestAdjudicationLedgerKinds:
    """The bounded-retry row (K4) and the calibration row (K5), beside the proposal row."""

    def test_the_kind_names(self):
        """The report and check scripts query these strings."""
        assert RECORD_KIND_ADJUDICATION_ATTEMPT == 'consolidation_adjudication_attempt'
        assert RECORD_KIND_ADJUDICATOR_CALIBRATION == 'consolidation_adjudicator_calibration'

    @pytest.mark.asyncio
    async def test_an_attempt_row_is_one_per_project_and_members_key(self, ledger):
        key = adjudication_attempt_key('dark_factory', MEMBERS_KEY)
        for attempts in (1, 2):
            await ledger.upsert(ReconLedgerRecord(
                **key, payload_json=f'{{"attempts": {attempts}}}', state='active',
                created_at=CREATED_AT,
            ))
        await ledger.upsert(ReconLedgerRecord(
            **adjudication_attempt_key('reify', MEMBERS_KEY), payload_json='{"attempts": 9}',
            state='active', created_at=CREATED_AT,
        ))

        stored = await ledger.get_by_identity(**key)

        assert stored is not None
        assert stored.payload_json == '{"attempts": 2}'
        assert await ledger.list_by_kind(
            'dark_factory', RECORD_KIND_ADJUDICATION_ATTEMPT, 'active',
        ) == [stored]
        other = await ledger.get_by_identity(**adjudication_attempt_key('reify', MEMBERS_KEY))
        assert other is not None
        assert other.payload_json == '{"attempts": 9}'

    @pytest.mark.asyncio
    async def test_a_calibration_row_belongs_to_no_project(self, ledger):
        key = adjudicator_calibration_key(IDENTITY)
        record = ReconLedgerRecord(
            **key, payload_json='{}', state='pass', created_at=CREATED_AT, expires_at=PAST,
        )
        await ledger.upsert(record)

        assert key['project_id'] == SERVICE_SCOPE_PROJECT_ID
        assert await ledger.get_by_identity(**key) == record
        assert await ledger.list_by_kind(
            'dark_factory', RECORD_KIND_ADJUDICATOR_CALIBRATION, 'pass',
        ) == []
        await ledger.gc('dark_factory', now=FAR_FUTURE, terminal_task_ids=[])
        assert await ledger.get_by_identity(**key) == record

    def test_the_service_scope_cannot_name_a_real_project(self):
        assert validate_project_id(SERVICE_SCOPE_PROJECT_ID) is not None

    @pytest.mark.parametrize(
        ('build', 'argument'),
        [
            (lambda: adjudication_attempt_key('', MEMBERS_KEY), 'project_id'),
            (lambda: adjudication_attempt_key(SERVICE_SCOPE_PROJECT_ID, MEMBERS_KEY), 'project_id'),
            (lambda: adjudication_attempt_key('dark_factory', ''), 'members_key'),
            (lambda: adjudicator_calibration_key(''), 'identity'),
        ],
        ids=['blank_project', 'service_scope_project', 'blank_members_key', 'blank_identity'],
    )
    def test_keys_refuse_blank_or_reserved_input(self, build, argument):
        with pytest.raises(ValueError, match=argument):
            build()
