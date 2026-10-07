"""Tests for the audit-trail rotation module (task 5771, docs/task-authoring.md §10).

Task dicts are built in the backend's ``get_task`` shape: ``id``, ``title``,
``description``, ``details``, ``status`` and ``metadata`` as a dict.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from fused_memory.reconciliation.audit_trail_rotation import (
    ROTATE_TARGET_BYTES,
    ROTATE_THRESHOLD_BYTES,
    plan_rotation,
    task_payload_bytes,
)

NOW = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)


def make_task(
    *,
    task_id: str = '42',
    title: str = 'Gate: decide the thing',
    description: str = 'A short description.',
    details: str = '',
    status: str = 'pending',
    metadata: Any = None,
) -> dict[str, Any]:
    return {
        'id': task_id,
        'title': title,
        'description': description,
        'details': details,
        'status': status,
        'metadata': {} if metadata is None else metadata,
    }


def dated_family(stem: str, days: list[int]) -> dict[str, Any]:
    return {f'{stem}_2026_09_{day:02d}': {'cycle': day, 'note': 'unchanged'} for day in days}


class TestPayloadAndNoOps:
    def test_payload_counts_utf8_bytes_of_every_column(self):
        metadata = {'files': ['a.py'], 'note': 'café'}
        task = make_task(
            title='Tïtle',
            description='em — dash',
            details='détails',
            metadata=metadata,
        )
        expected = (
            len('Tïtle'.encode())
            + len('em — dash'.encode())
            + len('détails'.encode())
            + len(json.dumps(metadata).encode())
        )
        assert task_payload_bytes(task) == expected
        assert task_payload_bytes(task) > len('Tïtle') + len('em — dash') + len('détails')

    def test_missing_or_none_columns_count_zero(self):
        task = {'id': '1', 'title': 'abc', 'description': None, 'metadata': None}
        assert task_payload_bytes(task) == 3

    def test_small_task_without_dated_family_needs_no_plan(self):
        task = make_task(metadata={'files': ['a.py'], 'task_kind': 'deterministic'})
        assert plan_rotation(task, now=NOW) is None

    def test_terminal_tasks_are_never_planned(self):
        oversize = 'x' * (ROTATE_THRESHOLD_BYTES + 1000)
        for status in ('done', 'cancelled'):
            task = make_task(
                status=status,
                description=oversize,
                metadata=dated_family('x_recon_relay', [10, 11, 12]),
            )
            assert task_payload_bytes(task) > ROTATE_THRESHOLD_BYTES
            assert plan_rotation(task, now=NOW) is None

    def test_non_dict_metadata_is_never_planned(self):
        oversize = '\n\n'.join(f'block {i} ' + 'y' * 900 for i in range(30))
        for metadata in ('{"not": "parsed"}', ['a', 'list'], 7):
            task = make_task(description=oversize, metadata=metadata)
            assert task_payload_bytes(task) > ROTATE_THRESHOLD_BYTES
            assert plan_rotation(task, now=NOW) is None

    def test_threshold_constants_match_docs_section_10(self):
        assert ROTATE_THRESHOLD_BYTES == 20_000
        assert ROTATE_TARGET_BYTES == 10_000
