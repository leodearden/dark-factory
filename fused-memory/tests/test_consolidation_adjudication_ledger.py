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

from fused_memory.reconciliation.consolidation_auto import UNREADABLE
from fused_memory.reconciliation.consolidation_proposal import content_sha256, members_key


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
