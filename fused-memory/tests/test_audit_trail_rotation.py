"""Tests for the audit-trail rotation module (task 5771, docs/task-authoring.md §10).

Task dicts are built in the backend's ``get_task`` shape: ``id``, ``title``,
``description``, ``details``, ``status`` and ``metadata`` as a dict.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from typing import Any, get_args

import pytest
from shared.task_metadata import parse_metadata

from fused_memory import memory_metadata
from fused_memory.reconciliation.audit_trail_rotation import (
    ARCHIVE_MAX_BYTES,
    HISTORY_KEEP,
    HISTORY_MAX,
    ROLLUP_KEY,
    ROTATE_TARGET_BYTES,
    ROTATE_THRESHOLD_BYTES,
    RotationOutcome,
    RotationStatus,
    TaskRewrite,
    archive_link_query,
    bound_audit_trail,
    near_duplicate_key,
    plan_rotation,
    render_audit_trail_rotation_section,
    task_fingerprint,
    task_payload_bytes,
    unrotatable_reason,
)
from fused_memory.reconciliation.context_assembler import HINT_QUERIES_EXECUTED
from fused_memory.reconciliation.prompts.stage2 import (
    STAGE2_SYSTEM_PROMPT,
    build_stage2_system_prompt,
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

    @pytest.mark.parametrize(
        'agent_array',
        [
            [{'cycle': 19, 'note': 'agent-shaped'}],
            [history_entry('bar_2026_09_18', '2026-09-18', 'another stem')],
        ],
        ids=['agent-shaped-entries', 'entries-of-another-stem'],
    )
    def test_an_array_the_harness_does_not_own_leaves_the_family_alone(self, agent_array):
        metadata = {'foo_history': agent_array, 'foo_2026_09_19': 'a', 'foo_2026_09_20': 'b'}
        assert plan_rotation(make_task(metadata=metadata), now=NOW) is None

    def test_a_harness_shaped_array_is_owned_again_after_the_rollup_was_dropped(self):
        existing = [history_entry('foo_2026_09_19', '2026-09-19', 'old')]
        metadata = {'foo_history': existing, 'foo_2026_09_20': 'new'}
        plan = plan_rotation(make_task(metadata=metadata), now=NOW)
        assert plan is not None
        rendered = plan.render(None).metadata
        assert rendered['foo_history'] == [
            history_entry('foo_2026_09_20', '2026-09-20', 'new'),
            *existing,
        ]
        assert rendered[ROLLUP_KEY]['history_keys'] == ['foo_history']

    @pytest.mark.parametrize(
        ('stem', 'value'),
        [
            ('files', ['docs/task-authoring.md']),
            ('pending_since', '2026-09-01T00:00:00.000Z'),
            ('done_provenance', {'kind': 'merged', 'commit': 'a' * 40}),
        ],
    )
    def test_an_undated_task_metadata_vocabulary_key_never_joins_its_family(self, stem, value):
        metadata = {stem: value, **dated_family(stem, [20, 21])}
        plan = plan_rotation(make_task(metadata=metadata), now=NOW)
        assert plan is not None
        rendered = plan.render(None).metadata
        assert rendered[stem] == value
        assert [entry['source_key'] for entry in rendered[f'{stem}_history']] == [
            f'{stem}_2026_09_21',
            f'{stem}_2026_09_20',
        ]

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


QUERY_SECTION = '=== MEMORY HINT QUERIES ROTATED OUT ==='


class TestMemoryHintQueryBound:
    def test_near_duplicate_key_ignores_case_spacing_dates_and_ids(self):
        first = 'index_health reconfirmed 2026-09-10 run 6f0b5d50-2434-4f30-a9d0-769a220b34a4'
        second = 'Index_health  reconfirmed 2026-09-11 run c820dde4-b716-4af8-bf63-abb8d5f9fc57'
        assert near_duplicate_key(first) == near_duplicate_key(second)
        assert near_duplicate_key('commit 4cb45448c3 landed') == near_duplicate_key(
            'commit 055e9c15a0 landed'
        )
        assert near_duplicate_key('cycle 12 relay') == near_duplicate_key('cycle 13 relay')

    def test_near_duplicate_key_keeps_different_queries_apart(self):
        assert near_duplicate_key('index_health reconfirmed') != near_duplicate_key(
            'consolidation backlog status'
        )
        assert near_duplicate_key('gate ruling effaced') != near_duplicate_key('gate ruling')

    def test_queries_past_history_max_are_deduped_and_trimmed_after_executed_prefix(self):
        executed = [
            'consolidation backlog status',
            'index_health drift on episodes',
            'gate 3524 ruling',
        ]
        queries = [
            *executed,
            'relay evidence 2026-09-01 run 6f0b5d50-2434-4f30-a9d0-769a220b34a4',
            'relay evidence 2026-09-02 run c820dde4-b716-4af8-bf63-abb8d5f9fc57',
            'scheduler wait anchor',
            'Consolidation  backlog status',
            'merge lane verify slot',
            'pending_since backfill',
            'relay evidence 2026-09-03 run 0e1d2c3b-4a59-6877-8695-a4b3c2d1e0f9',
            'curator drop decisions',
            'archive chain for task 42',
        ]
        assert len(queries) > HISTORY_MAX
        entities = ['task:42', 'gate']
        task = make_task(metadata={'memory_hints': {'entities': entities, 'queries': queries}})
        plan = plan_rotation(task, now=NOW)
        assert plan is not None
        hints = plan.render(None).metadata['memory_hints']
        assert len(executed) == HINT_QUERIES_EXECUTED
        assert hints['queries'] == [
            *executed,
            'curator drop decisions',
            'archive chain for task 42',
        ]
        assert len(hints['queries']) == HISTORY_KEEP
        assert hints['entities'] == entities
        assert plan.archive_text is not None
        archived = archive_section(plan.archive_text, QUERY_SECTION)
        for shed in set(queries) - set(hints['queries']):
            assert shed in archived

    @pytest.mark.parametrize(
        'variants',
        [
            [archive_link_query(f'{n:08x}-2434-4f30-a9d0-769a220b34a4', '42') for n in range(9)],
            [f'consolidation status as of 2026-{month:02d}-01' for month in range(1, 10)],
        ],
        ids=['archive-links', 'dated-status-queries'],
    )
    def test_the_newest_of_a_near_duplicate_class_survives(self, variants):
        executed = ['consolidation backlog status', 'index_health drift', 'gate 3524 ruling']
        queries = [*executed, *variants]
        assert len(queries) > HISTORY_MAX
        assert len({near_duplicate_key(variant) for variant in variants}) == 1
        task = make_task(metadata={'memory_hints': {'entities': [], 'queries': queries}})
        plan = plan_rotation(task, now=NOW)
        assert plan is not None
        assert plan.render(None).metadata['memory_hints']['queries'] == [*executed, variants[-1]]

    def test_the_executed_prefix_is_kept_even_when_it_holds_near_duplicates(self):
        executed = [f'relay evidence 2026-09-{day:02d}' for day in (1, 2, 3)]
        queries = [*executed, *(f'distinct topic {chr(97 + n)}' for n in range(HISTORY_MAX))]
        assert len(executed) == HINT_QUERIES_EXECUTED
        task = make_task(metadata={'memory_hints': {'entities': [], 'queries': queries}})
        plan = plan_rotation(task, now=NOW)
        assert plan is not None
        kept = plan.render(None).metadata['memory_hints']['queries']
        assert kept[:HINT_QUERIES_EXECUTED] == executed
        assert len(kept) == HISTORY_KEEP

    def test_queries_within_history_max_do_not_trigger_a_rotation(self):
        queries = [f'relay evidence cycle {n}' for n in range(HISTORY_MAX)]
        task = make_task(metadata={'memory_hints': {'entities': [], 'queries': queries}})
        assert plan_rotation(task, now=NOW) is None

    def test_legacy_list_memory_hints_are_left_untouched(self):
        legacy = [{'entity': f'e{n}', 'query': f'topic {chr(97 + n)}'} for n in range(12)]
        task = make_task(metadata={'memory_hints': legacy})
        assert plan_rotation(task, now=NOW) is None
        task = make_task(metadata={'memory_hints': legacy, **dated_family('x_relay', [1, 2])})
        plan = plan_rotation(task, now=NOW)
        assert plan is not None
        assert plan.render('abc-uuid').metadata['memory_hints'] == legacy

    def test_archive_link_query_is_appended_after_bounding(self):
        queries = [f'distinct topic {chr(97 + n)}' for n in range(HISTORY_MAX + 2)]
        task = make_task(metadata={'memory_hints': {'entities': [], 'queries': queries}})
        plan = plan_rotation(task, now=NOW)
        assert plan is not None
        rendered = plan.render('abc-uuid').metadata['memory_hints']['queries']
        assert len(rendered) == HISTORY_KEEP + 1
        assert rendered[:-1] == plan.render(None).metadata['memory_hints']['queries']
        assert 'abc-uuid' in rendered[-1]
        assert '42' in rendered[-1]

    def test_archive_link_creates_memory_hints_when_absent(self):
        existing = [
            history_entry(f'foo_2026_08_{day:02d}', f'2026-08-{day:02d}', day)
            for day in range(HISTORY_MAX + 1, 0, -1)
        ]
        metadata = {'foo_history': existing, ROLLUP_KEY: {'history_keys': ['foo_history']}}
        plan = plan_rotation(make_task(metadata=metadata), now=NOW)
        assert plan is not None
        hints = plan.render('abc-uuid').metadata['memory_hints']
        assert hints['entities'] == []
        assert len(hints['queries']) == 1
        assert 'abc-uuid' in hints['queries'][0]


DESCRIPTION_SECTION = '=== DESCRIPTION BLOCKS ROTATED OUT (verbatim) ==='


def cycle_blocks(count: int, *, filler_repeats: int) -> list[str]:
    return [
        f'Cycle {index:02d} relay summary.\n' + 'evidence unchanged; ' * filler_repeats
        for index in range(count)
    ]


def rewritten(task: dict[str, Any], rewrite: Any) -> dict[str, Any]:
    description = task['description'] if rewrite.description is None else rewrite.description
    return {**task, 'description': description, 'metadata': rewrite.metadata}


def split_rendered_description(original: str, rendered: str, first: str) -> tuple[str, str, str]:
    """(pointer, shed span, kept remainder) of a rotated description."""
    assert rendered.startswith(first + '\n\n')
    pointer, remainder = rendered[len(first) + 2 :].split('\n\n', 1)
    assert original.endswith(remainder)
    shed_span = original[len(first) + 2 : len(original) - len(remainder) - 2]
    return pointer, shed_span, remainder


class TestSizeTrigger:
    def test_description_middle_is_archived_verbatim_behind_a_pointer_block(self):
        blocks = cycle_blocks(30, filler_repeats=35)
        description = '\n\n'.join(blocks)
        task = make_task(description=description, details='Ruling: none yet.', metadata=dict(GATE_MARKERS))
        assert task_payload_bytes(task) > ROTATE_THRESHOLD_BYTES
        plan = plan_rotation(task, now=NOW)
        assert plan is not None
        assert plan.bytes_before == task_payload_bytes(task)
        rewrite = plan.render('mem-1')
        assert rewrite.description is not None
        assert rewrite.description.endswith(blocks[-1])
        pointer, shed_span, _ = split_rendered_description(description, rewrite.description, blocks[0])
        shed_blocks = shed_span.split('\n\n')
        assert shed_blocks == blocks[1 : 1 + len(shed_blocks)]
        assert 'mem-1' in pointer
        assert str(len(shed_blocks)) in pointer
        assert f'{len(shed_span.encode()):,}' in pointer
        for block in shed_blocks:
            assert block.splitlines()[0] in pointer
        assert plan.archive_text is not None
        assert shed_span in description
        assert archive_section(plan.archive_text, DESCRIPTION_SECTION) == shed_span
        assert task_payload_bytes(rewritten(task, rewrite)) <= ROTATE_TARGET_BYTES
        assert not plan.over_threshold_after

    def test_owned_arrays_are_trimmed_to_history_keep_under_the_size_trigger(self):
        existing = [
            history_entry(f'foo_2026_08_{day:02d}', f'2026-08-{day:02d}', {'day': day})
            for day in range(HISTORY_KEEP + 3, 0, -1)
        ]
        metadata = {'foo_history': existing, ROLLUP_KEY: {'history_keys': ['foo_history']}}
        description = '\n\n'.join(cycle_blocks(30, filler_repeats=35))
        plan = plan_rotation(make_task(description=description, metadata=metadata), now=NOW)
        assert plan is not None
        assert plan.render('mem-1').metadata['foo_history'] == existing[:HISTORY_KEEP]
        assert plan.archive_text is not None
        shed = json.loads(archive_section(plan.archive_text, METADATA_SECTION))
        assert shed['foo_history'] == existing[HISTORY_KEEP:]

    def test_archive_budget_stops_shedding_oldest_first(self):
        blocks = cycle_blocks(60, filler_repeats=50)
        description = '\n\n'.join(blocks)
        task = make_task(description=description)
        assert task_payload_bytes(task) > 3 * ROTATE_THRESHOLD_BYTES
        plan = plan_rotation(task, now=NOW)
        assert plan is not None
        assert plan.archive_text is not None
        assert len(plan.archive_text.encode()) <= ARCHIVE_MAX_BYTES
        rewrite = plan.render('mem-1')
        assert rewrite.description is not None
        _, shed_span, remainder = split_rendered_description(
            description, rewrite.description, blocks[0]
        )
        assert shed_span.startswith(blocks[1])
        assert remainder.startswith(blocks[1 + len(shed_span.split('\n\n'))])
        assert blocks[-2] in remainder
        assert plan.over_threshold_after

    def test_details_dominated_task_is_unrotatable_with_column_sizes(self):
        details = 'd' * 41_000
        task = make_task(description='One block only.', details=details, metadata=dict(GATE_MARKERS))
        assert plan_rotation(task, now=NOW) is None
        report = unrotatable_reason(task, now=NOW)
        assert report is not None
        assert report['reason'] == 'nothing_sheddable'
        assert report['payload_bytes'] == task_payload_bytes(task)
        assert report['column_bytes']['details'] == len(details)
        assert sum(report['column_bytes'].values()) == task_payload_bytes(task)

    def test_unrotatable_reason_is_none_under_threshold_or_when_a_plan_exists(self):
        assert unrotatable_reason(make_task(), now=NOW) is None
        description = '\n\n'.join(cycle_blocks(30, filler_repeats=35))
        assert unrotatable_reason(make_task(description=description), now=NOW) is None

    def test_terminal_over_threshold_task_reports_its_status_as_the_reason(self):
        task = make_task(status='done', details='d' * 41_000)
        report = unrotatable_reason(task, now=NOW)
        assert report is not None
        assert report['reason'] == 'terminal_status'

    def test_description_of_fewer_than_three_blocks_is_never_rewritten(self):
        description = 'a' * 11_000 + '\n\n' + 'b' * 11_000
        assert plan_rotation(make_task(description=description), now=NOW) is None
        task = make_task(description=description, metadata=dated_family('x_relay', [1, 2]))
        plan = plan_rotation(task, now=NOW)
        assert plan is not None
        assert plan.render(None).description is None

    def test_rewrite_carries_only_description_and_metadata(self):
        assert {field.name for field in dataclasses.fields(TaskRewrite)} == {
            'description',
            'metadata',
        }
        metadata = {**GATE_MARKERS, 'x_note': {'kept': True}, 'priority_hint': 'high'}
        description = '\n\n'.join(cycle_blocks(30, filler_repeats=35))
        plan = plan_rotation(make_task(description=description, metadata=metadata), now=NOW)
        assert plan is not None
        rendered = plan.render('mem-1').metadata
        for key, value in metadata.items():
            assert rendered[key] == value


def prior_record(index: int) -> dict[str, Any]:
    return {
        'rotated_at': f'2026-09-{index + 1:02d}T00:00:00+00:00',
        'trigger': 'pattern',
        'archive_memory_id': f'prior-archive-{index}',
        'bytes_before': 21_000,
        'bytes_after': 9_000,
    }


class TestRollupRecord:
    def _size_rotated(self) -> tuple[dict[str, Any], Any]:
        existing = [
            history_entry(f'foo_2026_08_{day:02d}', f'2026-08-{day:02d}', {'day': day})
            for day in range(HISTORY_KEEP + 2, 0, -1)
        ]
        queries = [f'distinct topic {chr(97 + n)}' for n in range(HISTORY_KEEP + 2)]
        metadata = {
            **GATE_MARKERS,
            **dated_family('x_relay', [20, 21]),
            'foo_history': existing,
            'memory_hints': {'entities': [], 'queries': queries},
            ROLLUP_KEY: {'history_keys': ['foo_history']},
        }
        description = '\n\n'.join(cycle_blocks(30, filler_repeats=35))
        task = make_task(description=description, details='Ruling: none yet.', metadata=metadata)
        plan = plan_rotation(task, now=NOW)
        assert plan is not None
        return task, plan

    def test_rollup_records_what_left_and_where_it_went(self):
        task, plan = self._size_rotated()
        rewrite = plan.render('mem-1')
        rollup = rewrite.metadata[ROLLUP_KEY]
        assert isinstance(rollup['standing_instruction'], str)
        assert 'history_keys' in rollup['standing_instruction']
        assert set(rollup['history_keys']) == {'foo_history', 'x_relay_history'}
        record = rollup['rotations'][0]
        assert record['rotated_at'] == NOW.isoformat()
        assert record['trigger'] == 'size'
        assert record['archive_memory_id'] == 'mem-1'
        assert record['bytes_before'] == task_payload_bytes(task)
        assert record['bytes_after'] == task_payload_bytes(rewritten(task, rewrite))
        assert record['description_blocks_shed'] > 0
        assert record['description_blocks_kept'] + record['description_blocks_shed'] == 30
        assert len(record['shed_block_leads']) == record['description_blocks_shed']
        assert record['shed_block_leads'][0] == 'Cycle 01 relay summary.'
        assert record['history_entries_kept'] == {'foo_history': HISTORY_KEEP, 'x_relay_history': 2}
        assert record['history_entries_shed'] == {'foo_history': 2}
        assert set(record['folded_keys']) == {'x_relay_2026_09_20', 'x_relay_2026_09_21'}
        assert record['memory_hint_queries_shed'] == 2

    def test_pattern_rotation_is_recorded_as_pattern(self):
        existing = [
            history_entry(f'foo_2026_08_{day:02d}', f'2026-08-{day:02d}', day)
            for day in range(HISTORY_MAX + 1, 0, -1)
        ]
        metadata = {'foo_history': existing, ROLLUP_KEY: {'history_keys': ['foo_history']}}
        plan = plan_rotation(make_task(metadata=metadata), now=NOW)
        assert plan is not None
        record = plan.render('mem-1').metadata[ROLLUP_KEY]['rotations'][0]
        assert record['trigger'] == 'pattern'
        assert record['history_entries_shed'] == {'foo_history': HISTORY_MAX + 1 - HISTORY_KEEP}

    def test_rotation_records_are_capped_and_the_oldest_is_archived(self):
        prior = [prior_record(index) for index in range(HISTORY_KEEP)]
        existing = [
            history_entry(f'foo_2026_08_{day:02d}', f'2026-08-{day:02d}', day)
            for day in range(HISTORY_MAX + 1, 0, -1)
        ]
        metadata = {
            'foo_history': existing,
            ROLLUP_KEY: {'history_keys': ['foo_history'], 'rotations': prior},
        }
        plan = plan_rotation(make_task(metadata=metadata), now=NOW)
        assert plan is not None
        rotations = plan.render('mem-1').metadata[ROLLUP_KEY]['rotations']
        assert len(rotations) == HISTORY_KEEP
        assert rotations[0]['archive_memory_id'] == 'mem-1'
        assert rotations[1:] == prior[: HISTORY_KEEP - 1]
        assert plan.archive_text is not None
        shed = json.loads(archive_section(plan.archive_text, METADATA_SECTION))
        assert shed[ROLLUP_KEY] == {'rotations': prior[HISTORY_KEEP - 1 :]}

    def test_archive_metadata_uses_blessed_and_experimental_keys_only(self):
        prior = [prior_record(index) for index in range(2)]
        existing = [
            history_entry(f'foo_2026_08_{day:02d}', f'2026-08-{day:02d}', day)
            for day in range(HISTORY_MAX + 1, 0, -1)
        ]
        metadata = {
            'foo_history': existing,
            ROLLUP_KEY: {'history_keys': ['foo_history'], 'rotations': prior},
        }
        task = make_task(metadata=metadata)
        plan = plan_rotation(task, now=NOW)
        assert plan is not None
        assert plan.archive_metadata == {
            'source': 'audit_trail_rotation',
            'task_id': '42',
            'x_rotated_at': NOW.isoformat(),
            'x_bytes_before': task_payload_bytes(task),
            'x_previous_archive_memory_id': 'prior-archive-0',
        }
        assert memory_metadata.classify_unknown_keys(dict(plan.archive_metadata)) == []
        assert (
            memory_metadata.validate_memory_metadata(
                dict(plan.archive_metadata), enforce_kind_registry=True
            )
            == []
        )

    def test_first_archive_has_no_previous_archive(self):
        _, plan = self._size_rotated()
        assert plan.archive_metadata['x_previous_archive_memory_id'] is None

    def test_fold_only_plan_updates_history_keys_without_a_rotation_record(self):
        prior = [prior_record(index) for index in range(2)]
        metadata = {
            **dated_family('x_relay', [20, 21]),
            ROLLUP_KEY: {'history_keys': [], 'rotations': prior},
        }
        plan = plan_rotation(make_task(metadata=metadata), now=NOW)
        assert plan is not None
        assert plan.archive_text is None
        rollup = plan.render(None).metadata[ROLLUP_KEY]
        assert rollup['history_keys'] == ['x_relay_history']
        assert rollup['rotations'] == prior

    def test_rollup_key_is_a_blessed_task_metadata_key(self):
        _, plan = self._size_rotated()
        _, warnings = parse_metadata(plan.render('mem-1').metadata, direction='read')
        assert [w for w in warnings if w.code == 'unknown_key' and w.field == ROLLUP_KEY] == []


_STORED = object()


class FakeArchive:
    """In-memory AuditTrailArchive; ``read_returns`` overrides what a read sees."""

    def __init__(
        self,
        *,
        write_id: str | None = 'mem-1',
        read_returns: Any = _STORED,
        discard_error: Exception | None = None,
    ):
        self.write_id = write_id
        self.read_returns = read_returns
        self.discard_error = discard_error
        self.store: dict[str, str] = {}
        self.writes: list[dict[str, Any]] = []
        self.reads: list[dict[str, Any]] = []
        self.discards: list[dict[str, Any]] = []

    async def write(self, *, project_id: str, content: str, metadata: dict[str, Any]) -> str | None:
        self.writes.append({'project_id': project_id, 'content': content, 'metadata': metadata})
        if self.write_id is not None:
            self.store[self.write_id] = content
        return self.write_id

    async def read(self, *, project_id: str, memory_id: str) -> str | None:
        self.reads.append({'project_id': project_id, 'memory_id': memory_id})
        if self.read_returns is not _STORED:
            return self.read_returns
        return self.store.get(memory_id)

    async def discard(self, *, project_id: str, memory_id: str) -> None:
        self.discards.append({'project_id': project_id, 'memory_id': memory_id})
        if self.discard_error is not None:
            raise self.discard_error
        self.store.pop(memory_id, None)


class RecordingCommit:
    def __init__(self, task: dict[str, Any], *, superseded: bool = False):
        self.task = task
        self.superseded = superseded
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self, *, expected_fingerprint: str, description: str | None, metadata: dict[str, Any]
    ) -> dict[str, Any] | None:
        self.calls.append(
            {
                'expected_fingerprint': expected_fingerprint,
                'description': description,
                'metadata': metadata,
            }
        )
        if self.superseded:
            return None
        new_description = self.task['description'] if description is None else description
        return {**self.task, 'description': new_description, 'metadata': metadata}


def size_triggered_task() -> dict[str, Any]:
    description = '\n\n'.join(cycle_blocks(30, filler_repeats=35))
    return make_task(description=description, details='Ruling: none yet.', metadata=dict(GATE_MARKERS))


async def bound(task: dict[str, Any], archive: FakeArchive, commit: RecordingCommit) -> Any:
    return await bound_audit_trail(task, project_id='p', now=NOW, archive=archive, commit=commit)


class TestBoundAuditTrailExecutor:
    @pytest.mark.asyncio
    async def test_size_rotation_archives_confirms_then_commits(self):
        task = size_triggered_task()
        plan = plan_rotation(task, now=NOW)
        assert plan is not None
        archive, commit = FakeArchive(), RecordingCommit(task)
        outcome = await bound(task, archive, commit)
        assert archive.writes == [
            {'project_id': 'p', 'content': plan.archive_text, 'metadata': plan.archive_metadata}
        ]
        assert archive.reads == [{'project_id': 'p', 'memory_id': 'mem-1'}]
        rewrite = plan.render('mem-1')
        assert commit.calls == [
            {
                'expected_fingerprint': task_fingerprint(task),
                'description': rewrite.description,
                'metadata': rewrite.metadata,
            }
        ]
        assert outcome.status == 'rotated'
        assert outcome.archive_memory_id == 'mem-1'
        assert outcome.bytes_before == task_payload_bytes(task)
        assert outcome.committed_task is not None
        assert outcome.bytes_after == task_payload_bytes(outcome.committed_task)
        assert outcome.bytes_after <= ROTATE_TARGET_BYTES
        assert archive.discards == []
        assert archive.store == {'mem-1': plan.archive_text}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'archive',
        [
            FakeArchive(write_id=None),
            FakeArchive(read_returns='not what was written'),
            FakeArchive(read_returns=None),
        ],
        ids=['write-returned-no-id', 'read-back-differs', 'read-back-missing'],
    )
    async def test_unconfirmed_archive_never_commits(self, archive: FakeArchive):
        task = size_triggered_task()
        commit = RecordingCommit(task)
        outcome = await bound(task, archive, commit)
        assert commit.calls == []
        assert outcome.status == 'archive_unconfirmed'
        assert outcome.bytes_before == task_payload_bytes(task)
        assert outcome.committed_task is None
        assert archive.store == {}

    @pytest.mark.asyncio
    async def test_an_unconfirmed_archive_with_an_id_is_discarded(self):
        task = size_triggered_task()
        archive = FakeArchive(read_returns='not what was written')
        outcome = await bound(task, archive, RecordingCommit(task))
        assert archive.discards == [{'project_id': 'p', 'memory_id': 'mem-1'}]
        assert outcome.archive_discarded

    @pytest.mark.asyncio
    async def test_moved_fingerprint_reports_superseded_and_discards_the_archive(self):
        task = size_triggered_task()
        archive = FakeArchive()
        outcome = await bound(task, archive, RecordingCommit(task, superseded=True))
        assert outcome.status == 'superseded'
        assert outcome.archive_memory_id == 'mem-1'
        assert outcome.committed_task is None
        assert archive.discards == [{'project_id': 'p', 'memory_id': 'mem-1'}]
        assert archive.store == {}
        assert outcome.archive_discarded

    @pytest.mark.asyncio
    async def test_a_failed_discard_still_reports_the_outcome(self):
        task = size_triggered_task()
        archive = FakeArchive(discard_error=RuntimeError('mem0 down'))
        outcome = await bound(task, archive, RecordingCommit(task, superseded=True))
        assert outcome.status == 'superseded'
        assert outcome.archive_memory_id == 'mem-1'
        assert not outcome.archive_discarded

    @pytest.mark.asyncio
    async def test_fold_only_plan_commits_without_an_archive(self):
        task = make_task(metadata=dated_family('x_relay', [20, 21]))
        archive, commit = FakeArchive(), RecordingCommit(task)
        outcome = await bound(task, archive, commit)
        assert archive.writes == []
        assert len(commit.calls) == 1
        assert commit.calls[0]['description'] is None
        assert outcome.status == 'folded'
        assert outcome.archive_memory_id is None

    @pytest.mark.asyncio
    async def test_nothing_to_do_returns_none(self):
        task = make_task()
        archive, commit = FakeArchive(), RecordingCommit(task)
        assert await bound(task, archive, commit) is None
        assert archive.writes == [] and commit.calls == []

    @pytest.mark.asyncio
    async def test_unrotatable_task_reports_column_sizes_without_side_effects(self):
        task = make_task(description='One block only.', details='d' * 41_000)
        archive, commit = FakeArchive(), RecordingCommit(task)
        outcome = await bound(task, archive, commit)
        assert outcome.status == 'unrotatable'
        assert outcome.unrotatable == unrotatable_reason(task, now=NOW)
        assert archive.writes == [] and commit.calls == []

    @pytest.mark.asyncio
    async def test_outcome_as_dict_is_json_serialisable(self):
        task = size_triggered_task()
        rotated = await bound(task, FakeArchive(), RecordingCommit(task))
        unrotatable = await bound(
            make_task(details='d' * 41_000), FakeArchive(), RecordingCommit(task)
        )
        for outcome in (rotated, unrotatable):
            payload = json.loads(json.dumps(outcome.as_dict()))
            assert payload['status'] == outcome.status

    @pytest.mark.asyncio
    async def test_a_failure_outcome_has_the_wire_shape_of_every_other(self):
        task = size_triggered_task()
        rotated = await bound(task, FakeArchive(), RecordingCommit(task))
        failed = RotationOutcome.failed('42', RuntimeError('mem0 down'))
        assert failed.status in get_args(RotationStatus)
        assert failed.as_dict().keys() == rotated.as_dict().keys()
        assert 'mem0 down' in failed.as_dict()['error']


class TestStage2PromptStatesTheRule:
    """The rotation section is wired into every assembled Stage 2 prompt."""

    @pytest.mark.parametrize('project_id', ['dark_factory', 'autopilot_video'])
    def test_the_section_is_in_every_assembled_stage2_prompt(self, project_id):
        section = render_audit_trail_rotation_section()
        assert section in STAGE2_SYSTEM_PROMPT
        assert section in build_stage2_system_prompt(project_id)
