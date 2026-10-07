"""Tests for the audit-trail rotation module (task 5771, docs/task-authoring.md §10).

Task dicts are built in the backend's ``get_task`` shape: ``id``, ``title``,
``description``, ``details``, ``status`` and ``metadata`` as a dict.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from fused_memory.reconciliation.audit_trail_rotation import (
    HISTORY_KEEP,
    HISTORY_MAX,
    ROLLUP_KEY,
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


METADATA_SECTION = '=== METADATA ENTRIES ROTATED OUT (verbatim JSON) ==='


def archive_section(archive_text: str, header: str) -> str:
    body = archive_text.split(header + '\n', 1)[1]
    return body.split('\n=== ', 1)[0]


def history_entry(source_key: str, recorded_on: str, value: Any) -> dict[str, Any]:
    return {'source_key': source_key, 'recorded_on': recorded_on, 'value': value}


GATE_MARKERS = {
    'task_kind': 'deterministic',
    'execution_class': 'decision',
    'operational_mode': 'x',
    'always_escalates': True,
    'files': ['docs/task-authoring.md'],
}


class TestDatedKeyFold:
    def test_two_dated_keys_fold_into_a_newest_first_history_array(self):
        older = {'relay': 'first', 'ids': [1, 2]}
        newer = {'relay': 'second', 'ids': [3]}
        metadata = {
            'x_recon_evidence_relay_2026_09_12': older,
            'x_recon_evidence_relay_2026_09_14': newer,
            **GATE_MARKERS,
        }
        plan = plan_rotation(make_task(metadata=metadata), now=NOW)
        assert plan is not None
        rendered = plan.render(None).metadata
        assert 'x_recon_evidence_relay_2026_09_12' not in rendered
        assert 'x_recon_evidence_relay_2026_09_14' not in rendered
        assert rendered['x_recon_evidence_relay_history'] == [
            history_entry('x_recon_evidence_relay_2026_09_14', '2026-09-14', newer),
            history_entry('x_recon_evidence_relay_2026_09_12', '2026-09-12', older),
        ]
        for key, value in GATE_MARKERS.items():
            assert rendered[key] == value
        assert plan.archive_text is None

    def test_undated_stem_key_joins_and_suffix_letters_and_hyphen_dates_sort(self):
        metadata = {
            'index_health_reconfirmed': 'bare',
            'index_health_reconfirmed_2026_07_19': 'plain',
            'index_health_reconfirmed_2026_07_19b': 'lettered',
            'index_health_reconfirmed_2026-09-10': 'hyphen',
        }
        plan = plan_rotation(make_task(metadata=metadata), now=NOW)
        assert plan is not None
        history = plan.render(None).metadata['index_health_reconfirmed_history']
        assert [entry['source_key'] for entry in history] == [
            'index_health_reconfirmed_2026-09-10',
            'index_health_reconfirmed_2026_07_19b',
            'index_health_reconfirmed_2026_07_19',
            'index_health_reconfirmed',
        ]
        assert [entry['value'] for entry in history] == ['hyphen', 'lettered', 'plain', 'bare']
        assert history[0]['recorded_on'] == '2026-09-10'
        assert 'index_health_reconfirmed' not in plan.render(None).metadata

    def test_single_dated_key_is_not_a_family(self):
        task = make_task(metadata={'dark_factory_3708_ruling_2026_09_21': {'ruling': 'A'}})
        assert plan_rotation(task, now=NOW) is None

    def test_one_dated_key_extends_an_owned_history_array(self):
        existing = [history_entry('foo_2026_09_19', '2026-09-19', 'old')]
        metadata = {
            'foo_history': existing,
            'foo_2026_09_20': 'new',
            ROLLUP_KEY: {'history_keys': ['foo_history']},
        }
        plan = plan_rotation(make_task(metadata=metadata), now=NOW)
        assert plan is not None
        rendered = plan.render(None).metadata
        assert rendered['foo_history'] == [
            history_entry('foo_2026_09_20', '2026-09-20', 'new'),
            *existing,
        ]
        assert 'foo_2026_09_20' not in rendered

    def test_owned_array_past_history_max_is_trimmed_and_shed_entries_archived(self):
        existing = [
            history_entry(f'foo_2026_08_{day:02d}', f'2026-08-{day:02d}', {'day': day})
            for day in range(HISTORY_MAX, 0, -1)
        ]
        metadata = {
            'foo_history': existing,
            'foo_2026_09_20': 'new',
            ROLLUP_KEY: {'history_keys': ['foo_history']},
        }
        plan = plan_rotation(make_task(metadata=metadata), now=NOW)
        assert plan is not None
        kept = plan.render('mem-1').metadata['foo_history']
        assert len(kept) == HISTORY_KEEP
        assert kept[0]['source_key'] == 'foo_2026_09_20'
        assert kept[1:] == existing[: HISTORY_KEEP - 1]
        assert plan.archive_text is not None
        shed = json.loads(archive_section(plan.archive_text, METADATA_SECTION))
        assert shed['foo_history'] == existing[HISTORY_KEEP - 1 :]

    def test_existing_non_list_history_key_leaves_the_family_alone(self):
        metadata = {
            'foo_history': 'hand-written prose',
            'foo_2026_09_19': 'a',
            'foo_2026_09_20': 'b',
        }
        assert plan_rotation(make_task(metadata=metadata), now=NOW) is None

    def test_rollup_history_keys_lists_every_array_the_fold_created(self):
        metadata = {
            **dated_family('alpha', [1, 2]),
            **dated_family('beta', [3, 4, 5]),
            'gamma_2026_09_06': 'lonely',
        }
        plan = plan_rotation(make_task(metadata=metadata), now=NOW)
        assert plan is not None
        rendered = plan.render(None).metadata
        assert set(rendered[ROLLUP_KEY]['history_keys']) == {'alpha_history', 'beta_history'}
        assert rendered['gamma_2026_09_06'] == 'lonely'

    def test_reopened_task_with_done_provenance_is_rotated_with_it_passed_through(self):
        provenance = {'kind': 'merged', 'commit': 'a' * 40}
        metadata = {'done_provenance': provenance, **dated_family('x_relay', [20, 21])}
        plan = plan_rotation(make_task(status='pending', metadata=metadata), now=NOW)
        assert plan is not None
        assert plan.render(None).metadata['done_provenance'] == provenance
