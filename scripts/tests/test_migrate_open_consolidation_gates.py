"""Tests for scripts/migrate_open_consolidation_gates.py (task 5244, PRD θ).

The script reads every open consolidation gate from a real tasks.db, re-reads
each gate's topic cluster live, rules on it with the auto predicate, and
(under ``--execute``) writes through the executor's retain arm. The stores
here are a real tasks.db seeded with gate metadata from the real
``build_consolidation_gate_task`` and a small stateful fake of the public
MemoryService surface, defined below. ``run`` is driven with ``--json`` and
the structured payload is what the assertions read.
"""

from __future__ import annotations

import asyncio
import copy
import json
import sqlite3
import uuid
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import migrate_open_consolidation_gates as mod
import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from fused_memory.config.schema import ConsolidationAutoConfig, Mem0UpdateConfig
from fused_memory.models.memory import AddMemoryResponse
from fused_memory.reconciliation.consolidation_auto import build_auto_canonical
from fused_memory.reconciliation.consolidation_gate import (
    GATE_METADATA_KEY,
    build_consolidation_gate_task,
)
from shared.project_layout import escalation_queue_dir

PROJECT_ID = 'dark_factory'
CATEGORY = 'procedural_knowledge'
CREATED_AT = '2026-06-01T00:00:00+00:00'
TOPIC = 'gate-topic-alpha'
OTHER_TOPIC = 'gate-topic-beta'
CLAIM = 'Stage one keeps reproposing the identical unaddressed duplicate cluster'


def new_id() -> str:
    return str(uuid.uuid4())


def enabled_config(**overrides: Any) -> ConsolidationAutoConfig:
    fields: dict[str, Any] = {'enabled': True, 'enabled_projects': [PROJECT_ID]}
    fields.update(overrides)
    return ConsolidationAutoConfig(**fields)


class FakeMemoryService:
    """A stateful Mem0 over ``{id: {'id', 'content', 'metadata'}}``.

    Only the public MemoryService surface the script and its shared helpers
    touch. Scroll rows omit ``content``, as the real service's do. Injection
    points: ``read_errors`` (id -> exception), ``scroll_errors`` (topic ->
    exception), ``update_returns`` (id -> the dict ``update_memory`` returns
    instead of landing the patch).
    """

    def __init__(self, *, consolidation_auto: ConsolidationAutoConfig | None = None) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.read_errors: dict[str, BaseException] = {}
        self.scroll_errors: dict[str, BaseException] = {}
        self.update_returns: dict[str, dict[str, Any]] = {}
        self.reads = 0
        self.updates = 0
        self.adds = 0
        self.config = SimpleNamespace(
            mem0_update=Mem0UpdateConfig(),
            consolidation_auto=consolidation_auto or enabled_config(),
        )

    # ---- seeding and inspection ------------------------------------------

    def seed(self, *, content: str | None = None, **metadata: Any) -> str:
        memory_id = new_id()
        self.records[memory_id] = {
            'id': memory_id,
            'content': content or f'record {memory_id}',
            'metadata': {'category': CATEGORY, **metadata},
        }
        return memory_id

    def seed_many(self, n: int, **metadata: Any) -> list[str]:
        return [self.seed(**metadata) for _ in range(n)]

    def metadata_of(self, memory_id: str) -> dict[str, Any]:
        return self.records[memory_id]['metadata']

    def snapshot(self, memory_id: str) -> dict[str, Any]:
        return copy.deepcopy(self.records[memory_id])

    def canonicals_of(self, topic: str) -> list[dict[str, Any]]:
        return [
            r for r in self.records.values()
            if r['metadata'].get('topic') == topic and r['metadata'].get('canonical') is True
        ]

    def writes(self) -> tuple[int, int]:
        return self.updates, self.adds

    # ---- the MemoryService surface ---------------------------------------

    async def get_memory_by_id(self, project_id: str, memory_id: str) -> dict[str, Any] | None:
        self.reads += 1
        if memory_id in self.read_errors:
            raise self.read_errors[memory_id]
        record = self.records.get(memory_id)
        return copy.deepcopy(record) if record is not None else None

    def _matching(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {'id': r['id'], 'created_at': CREATED_AT, 'metadata': copy.deepcopy(r['metadata'])}
            for r in self.records.values()
            if all(r['metadata'].get(k) == v for k, v in filters.items())
        ]

    async def get_memories_by_metadata(
        self, project_id: str, filters: dict[str, Any], limit: int = 1000,
    ) -> list[dict[str, Any]]:
        self.reads += 1
        topic = filters.get('topic')
        if topic in self.scroll_errors:
            raise self.scroll_errors[topic]
        return self._matching(filters)[:limit]

    async def count_memories_by_metadata(self, project_id: str, filters: dict[str, Any]) -> int:
        self.reads += 1
        topic = filters.get('topic')
        if topic in self.scroll_errors:
            raise self.scroll_errors[topic]
        return len(self._matching(filters))

    async def update_memory(
        self,
        memory_id: str,
        project_id: str = 'main',
        content: str | None = None,
        metadata_patch: dict[str, Any] | None = None,
        metadata_mode: str = 'merge',
        **_: Any,
    ) -> dict[str, Any]:
        self.updates += 1
        if memory_id in self.update_returns:
            return self.update_returns[memory_id]
        record = self.records.get(memory_id)
        if record is None:
            return {'error': f'{memory_id} not found', 'error_type': 'MemoryNotFound'}
        record['metadata'].update(metadata_patch or {})
        if content is not None:
            record['content'] = content
        return {'status': 'updated', 'store': 'mem0', 'id': memory_id}

    async def add_memory(
        self,
        content: str,
        category: str | None = None,
        project_id: str = 'main',
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> AddMemoryResponse:
        self.adds += 1
        memory_id = new_id()
        self.records[memory_id] = {
            'id': memory_id,
            'content': content,
            'metadata': {**(metadata or {}), 'category': category},
        }
        return AddMemoryResponse(memory_ids=[memory_id], message='ok')


# ---- fixtures and drivers ----------------------------------------------------


@pytest.fixture(autouse=True)
def writable_mem0_dir(tmp_path, monkeypatch):
    mem0_dir = tmp_path / 'mem0'
    mem0_dir.mkdir()
    monkeypatch.setenv('MEM0_DIR', str(mem0_dir))
    return mem0_dir


@pytest.fixture
def root(tmp_path) -> Path:
    project_root = tmp_path / 'project'
    (project_root / '.taskmaster' / 'tasks').mkdir(parents=True)
    return project_root


@pytest.fixture
def seed_tasks(root, make_tasks_db):
    def _seed(rows: list[dict[str, Any]]) -> Path:
        return make_tasks_db(rows, directory=root / '.taskmaster' / 'tasks')

    return _seed


@pytest.fixture
def fake() -> FakeMemoryService:
    return FakeMemoryService()


def gate_row(
    task_id: int,
    topic: str,
    *,
    observed: Sequence[str] = (),
    status: str = 'blocked',
    title: str | None = None,
    **builder: Any,
) -> dict[str, Any]:
    spec = build_consolidation_gate_task(
        topic=topic, observed_members=observed, detector='stage2', **builder,
    )
    return {
        'id': task_id,
        'title': title or spec.title,
        'status': status,
        'metadata': spec.metadata,
    }


def drive(root: Path, fake: FakeMemoryService, capsys, *extra: str) -> tuple[int, dict]:
    argv = ['--project-root', str(root), '--project-id', PROJECT_ID, '--json', *extra]
    code = asyncio.run(mod.run(mod.build_parser().parse_args(argv), memory=fake))
    out = capsys.readouterr().out
    return code, json.loads(out)


def gate_of(payload: dict, task_id: int) -> dict:
    (found,) = [g for g in payload['gates'] if g['task_id'] == str(task_id)]
    return found


def ids_by_code(gate: dict) -> list[dict]:
    return [{'code': r['code'], 'ids': r['ids']} for r in gate['reasons']]


def task_rows(db_path: Path) -> list[tuple]:
    with closing(sqlite3.connect(db_path)) as con:
        return con.execute('SELECT id, status, metadata FROM tasks ORDER BY id').fetchall()


# ---- enumeration and dry-run verdicts ---------------------------------------


class TestEnumeration:
    def test_only_open_gates_carrying_a_block_are_listed(self, root, fake, seed_tasks, capsys):
        seed_tasks([
            gate_row(11, 'a-b-c', title='Consolidation gate: topic a-b-c'),
            gate_row(12, 'd-e-f', title='Human gate: consolidate something else'),
            {'id': 13, 'status': 'blocked', 'metadata': {'operational_mode': 'gate'}},
            gate_row(14, 'g-h-i', status='done'),
            {
                'id': 15,
                'status': 'pending',
                'metadata': {'operational_mode': 'llm', GATE_METADATA_KEY: {'topic': 'j-k-l'}},
            },
            {
                'id': 16,
                'status': 'blocked',
                'metadata': {
                    'operational_mode': 'gate',
                    GATE_METADATA_KEY: {'topic': 'Not A Slug'},
                },
            },
        ])

        code, payload = drive(root, fake, capsys, '--dry-run')

        assert code == 0
        assert [g['task_id'] for g in payload['gates']] == ['11', '12', '16']
        assert gate_of(payload, 11)['topic'] == 'a-b-c'
        assert gate_of(payload, 12)['topic'] == 'd-e-f'
        malformed = gate_of(payload, 16)
        assert malformed['disposition'] == 'refused'
        assert malformed['codes'] == ['gate_topic_unusable']

    def test_the_header_names_the_project_mode_and_run(self, root, fake, seed_tasks, capsys):
        seed_tasks([])

        code, payload = drive(root, fake, capsys, '--dry-run')

        assert code == 0
        assert payload['project_id'] == PROJECT_ID
        assert payload['mode'] == 'dry_run'
        assert payload['run_id'].startswith('migrate-gates-')
        assert payload['gates'] == []


class TestDryRunVerdict:
    def test_a_fresh_cluster_with_a_claim_would_execute(self, root, fake, seed_tasks, capsys):
        members = fake.seed_many(3)
        seed_tasks([gate_row(21, TOPIC, observed=members)])

        code, payload = drive(root, fake, capsys, '--dry-run', '--claim', f'{TOPIC}={CLAIM}')

        assert code == 0
        gate = gate_of(payload, 21)
        assert gate['verdict'] == 'pass'
        assert gate['disposition'] == 'would_execute'
        assert gate['codes'] == []
        assert sorted(gate['members']) == sorted(members)
        assert gate['category'] == CATEGORY

    def test_a_pass_with_no_claim_is_refused_claim_required(
        self, root, fake, seed_tasks, capsys,
    ):
        seed_tasks([gate_row(22, TOPIC, observed=fake.seed_many(3))])

        code, payload = drive(root, fake, capsys, '--dry-run')

        assert code == 0
        gate = gate_of(payload, 22)
        assert gate['verdict'] == 'pass'
        assert gate['disposition'] == 'refused'
        assert 'claim_required' in gate['codes']

    def test_dry_run_writes_nothing_anywhere(self, root, fake, seed_tasks, capsys):
        db_path = seed_tasks([gate_row(23, TOPIC, observed=fake.seed_many(3))])
        before = task_rows(db_path)
        records_before = copy.deepcopy(fake.records)

        code, _ = drive(root, fake, capsys, '--dry-run', '--claim', f'{TOPIC}={CLAIM}')

        assert code == 0
        assert fake.writes() == (0, 0)
        assert fake.records == records_before
        assert task_rows(db_path) == before

    def test_dry_run_needs_no_writable_store(
        self, root, fake, seed_tasks, capsys, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv('MEM0_DIR', str(tmp_path / 'no-such-mem0-dir'))
        seed_tasks([gate_row(24, TOPIC, observed=fake.seed_many(3))])

        code, payload = drive(root, fake, capsys, '--dry-run', '--claim', f'{TOPIC}={CLAIM}')

        assert code == 0
        assert gate_of(payload, 24)['disposition'] == 'would_execute'


class TestModeIsRequired:
    @pytest.mark.parametrize('argv', [[], ['--dry-run', '--execute']])
    def test_exactly_one_mode_is_required(self, argv):
        with pytest.raises(SystemExit) as exc:
            mod.build_parser().parse_args(argv)

        assert exc.value.code == 2


class TestLiveReRead:
    """INV-3: the live cluster is re-read; the gate's snapshot alone is never acted on."""

    def test_stamped_scroll_members_join_the_observed_snapshot(
        self, root, fake, seed_tasks, capsys,
    ):
        snapshot = fake.seed_many(2)
        stamped = fake.seed_many(2, topic=TOPIC)
        seed_tasks([gate_row(31, TOPIC, observed=snapshot)])

        _, payload = drive(root, fake, capsys, '--dry-run', '--claim', f'{TOPIC}={CLAIM}')

        assert sorted(gate_of(payload, 31)['members']) == sorted([*snapshot, *stamped])

    def test_an_observed_member_that_no_longer_exists_is_disclosed_and_dropped(
        self, root, fake, seed_tasks, capsys,
    ):
        live = fake.seed_many(3)
        ghost = new_id()
        seed_tasks([gate_row(32, TOPIC, observed=[*live, ghost])])

        _, payload = drive(root, fake, capsys, '--dry-run', '--claim', f'{TOPIC}={CLAIM}')

        gate = gate_of(payload, 32)
        assert gate['absent_observed'] == [ghost]
        assert ghost not in gate['members']
        assert gate['verdict'] == 'pass'
        assert gate['disposition'] == 'would_execute'

    def test_an_unreadable_member_is_left_to_the_sitting(self, root, fake, seed_tasks, capsys):
        members = fake.seed_many(3)
        fake.read_errors[members[1]] = TimeoutError('read timed out')
        seed_tasks([gate_row(33, TOPIC, observed=members)])

        _, payload = drive(root, fake, capsys, '--dry-run', '--claim', f'{TOPIC}={CLAIM}')

        gate = gate_of(payload, 33)
        assert gate['disposition'] == 'left_to_sitting'
        assert {'code': 'member_unreadable', 'ids': [members[1]]} in ids_by_code(gate)

    def test_a_correction_banner_is_left_to_the_sitting(self, root, fake, seed_tasks, capsys):
        members = fake.seed_many(2)
        corrected = fake.seed(content='[CORRECTION 2026-01-01] the rule was reversed')
        seed_tasks([gate_row(34, TOPIC, observed=[*members, corrected])])

        _, payload = drive(root, fake, capsys, '--dry-run', '--claim', f'{TOPIC}={CLAIM}')

        gate = gate_of(payload, 34)
        assert gate['disposition'] == 'left_to_sitting'
        assert {'code': 'member_carries_correction_banner', 'ids': [corrected]} in ids_by_code(gate)


class TestFailClosedScroll:
    def test_a_raising_scroll_refuses_that_gate_and_the_run_goes_on(
        self, root, fake, seed_tasks, capsys,
    ):
        fake.scroll_errors[TOPIC] = TimeoutError('qdrant down')
        seed_tasks([
            gate_row(41, TOPIC, observed=fake.seed_many(3)),
            gate_row(42, OTHER_TOPIC, observed=fake.seed_many(3)),
        ])

        code, payload = drive(
            root, fake, capsys, '--dry-run',
            '--claim', f'{TOPIC}={CLAIM}', '--claim', f'{OTHER_TOPIC}={CLAIM}',
        )

        assert code == 0
        failed = gate_of(payload, 41)
        assert failed['disposition'] == 'refused'
        assert failed['codes'] == ['member_scroll_unavailable']
        assert gate_of(payload, 42)['disposition'] == 'would_execute'

    def test_a_full_scroll_page_refuses_the_gate_as_incomplete(
        self, root, fake, seed_tasks, capsys,
    ):
        fake.seed_many(mod.closure_cli.SCROLL_LIMIT, topic=TOPIC)
        seed_tasks([gate_row(43, TOPIC)])

        _, payload = drive(root, fake, capsys, '--dry-run', '--claim', f'{TOPIC}={CLAIM}')

        gate = gate_of(payload, 43)
        assert gate['disposition'] == 'refused'
        assert gate['codes'] == ['member_scroll_incomplete']


class TestActionableRefusals:
    def test_a_category_without_flood_weight_is_not_consolidatable(
        self, root, fake, seed_tasks, capsys,
    ):
        members = fake.seed_many(3, category='decisions_and_rationale')
        seed_tasks([gate_row(51, TOPIC, observed=members)])

        _, payload = drive(root, fake, capsys, '--dry-run', '--claim', f'{TOPIC}={CLAIM}')

        gate = gate_of(payload, 51)
        assert gate['disposition'] == 'refused'
        assert gate['codes'] == ['category_not_consolidatable']

    def test_more_members_than_member_max_is_out_of_range(self, root, fake, seed_tasks, capsys):
        too_many = fake.config.consolidation_auto.member_max + 1
        seed_tasks([gate_row(52, TOPIC, observed=fake.seed_many(too_many))])

        _, payload = drive(root, fake, capsys, '--dry-run', '--claim', f'{TOPIC}={CLAIM}')

        gate = gate_of(payload, 52)
        assert gate['disposition'] == 'refused'
        assert 'member_count_out_of_range' in gate['codes']

    def test_a_claim_the_validator_rejects_is_refused(self, root, fake, seed_tasks, capsys):
        seed_tasks([gate_row(53, TOPIC, observed=fake.seed_many(3))])

        _, payload = drive(root, fake, capsys, '--dry-run', '--claim', f'{TOPIC}=INDEX — foo bar')

        gate = gate_of(payload, 53)
        assert gate['disposition'] == 'refused'
        assert 'claim_not_index_shaped' in gate['codes']


class TestRunLevelRefusals:
    @pytest.mark.parametrize(
        'claims',
        [
            ['no-equals-sign-here'],
            [f'Not A Slug={CLAIM}'],
            [f'{TOPIC}={CLAIM}', f'{TOPIC}=Another claim entirely for the very same topic'],
        ],
    )
    def test_a_bad_claim_argument_stops_the_run_before_any_read(
        self, root, fake, seed_tasks, capsys, caplog, claims,
    ):
        seed_tasks([gate_row(61, TOPIC, observed=fake.seed_many(3))])
        argv = ['--dry-run'] + [arg for claim in claims for arg in ('--claim', claim)]

        code = asyncio.run(mod.run(mod.build_parser().parse_args([
            '--project-root', str(root), '--project-id', PROJECT_ID, *argv,
        ]), memory=fake))

        assert code == 2
        assert fake.reads == 0
        assert claims[-1].split('=')[0] in caplog.text

    def test_a_full_canonical_slug_page_stops_the_run(self, root, fake, seed_tasks, capsys):
        from fused_memory.reconciliation.consolidation_auto_executor import CANONICAL_SLUG_LIMIT

        for n in range(CANONICAL_SLUG_LIMIT):
            fake.seed(topic=f'filler-topic-{n}', canonical=True)
        seed_tasks([gate_row(62, TOPIC, observed=fake.seed_many(3))])

        code = asyncio.run(mod.run(mod.build_parser().parse_args([
            '--project-root', str(root), '--project-id', PROJECT_ID, '--dry-run',
        ]), memory=fake))

        assert code == 2
        assert fake.writes() == (0, 0)


def run_code(root: Path, fake: FakeMemoryService, *extra: str) -> int:
    argv = ['--project-root', str(root), '--project-id', PROJECT_ID, *extra]
    return asyncio.run(mod.run(mod.build_parser().parse_args(argv), memory=fake))


class TestExecuteGating:
    def test_a_store_that_cannot_be_mutated_refuses_before_any_read(
        self, root, fake, seed_tasks, caplog, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv('MEM0_DIR', str(tmp_path / 'no-such-mem0-dir'))
        seed_tasks([gate_row(71, TOPIC, observed=fake.seed_many(3))])

        with caplog.at_level('ERROR', logger='migrate_open_consolidation_gates'):
            code = run_code(root, fake, '--execute', '--claim', f'{TOPIC}={CLAIM}')

        assert code == 2
        assert fake.writes() == (0, 0)
        assert fake.reads == 0
        assert any(
            r.name == 'migrate_open_consolidation_gates' and 'no-such-mem0-dir' in r.getMessage()
            for r in caplog.records
        )

    @pytest.mark.parametrize(
        ('config', 'leaf'),
        [
            (enabled_config(enabled=False), 'consolidation_auto.enabled'),
            (enabled_config(enabled_projects=['someone_else']), 'consolidation_auto.enabled_projects'),
        ],
    )
    def test_the_kill_switch_refuses_execute_and_allows_dry_run(
        self, root, seed_tasks, capsys, caplog, config, leaf,
    ):
        fake = FakeMemoryService(consolidation_auto=config)
        seed_tasks([gate_row(72, TOPIC, observed=fake.seed_many(3))])

        code = run_code(root, fake, '--execute', '--claim', f'{TOPIC}={CLAIM}')

        assert code == 2
        assert fake.writes() == (0, 0)
        assert fake.reads == 0
        assert leaf in caplog.text
        dry_code, payload = drive(root, fake, capsys, '--dry-run', '--claim', f'{TOPIC}={CLAIM}')
        assert dry_code == 0
        assert gate_of(payload, 72)['disposition'] == 'would_execute'

    def test_execute_proceeds_with_the_switch_on_and_a_writable_store(
        self, root, fake, seed_tasks, capsys,
    ):
        seed_tasks([])

        code, payload = drive(root, fake, capsys, '--execute')

        assert code == 0
        assert payload['mode'] == 'execute'
        assert payload['gates'] == []


def seed_gate_escalation(root: Path, task_id: int) -> str:
    queue = EscalationQueue(escalation_queue_dir(root))
    escalation_id = queue.make_id(str(task_id))
    queue.submit(Escalation(
        id=escalation_id,
        task_id=str(task_id),
        agent_role='orchestrator-deterministic',
        severity='critical',
        category='milestone_gate',
        summary='gate',
        level=2,
    ))
    return escalation_id


def tools_of(gate: dict) -> list[str]:
    return [call['tool'] for call in gate['next']]


class TestExecuteWrites:
    def test_a_pass_mints_stamps_closes_and_lists_the_resume(
        self, root, fake, seed_tasks, capsys,
    ):
        members = fake.seed_many(3)
        seed_tasks([gate_row(81, TOPIC, observed=members)])
        escalation_id = seed_gate_escalation(root, 81)

        code, payload = drive(root, fake, capsys, '--execute', '--claim', f'{TOPIC}={CLAIM}')

        assert code == 0
        gate = gate_of(payload, 81)
        assert gate['disposition'] == 'executed'
        (canonical,) = fake.canonicals_of(TOPIC)
        assert canonical['content'] == build_auto_canonical(CLAIM, TOPIC, 3, payload['run_id'])
        for member in members:
            assert fake.metadata_of(member)['topic'] == TOPIC
            assert fake.metadata_of(member)['x_auto_consolidated_run'] == payload['run_id']
        assert gate['execution']['canonical_id'] == canonical['id']
        assert gate['execution']['minted'] is True
        assert sorted(gate['execution']['tagged']) == sorted(members)
        assert gate['closure']['closed'] is True
        (resume,) = gate['next']
        assert resume['tool'] == 'resolve_issue'
        assert resume['arguments']['escalation_id'] == escalation_id
        assert resume['arguments']['action'] == 'resume'
        assert payload['run_id'] in resume['arguments']['resolution']

    def test_tag_only_stamps_the_observed_members_and_leaves_the_canonical(
        self, root, fake, seed_tasks, capsys,
    ):
        canonical = fake.seed(topic=TOPIC, canonical=True)
        fake.seed_many(2, topic=TOPIC)
        fresh = fake.seed_many(2)
        before = fake.snapshot(canonical)
        seed_tasks([gate_row(82, TOPIC, observed=fresh)])
        escalation_id = seed_gate_escalation(root, 82)

        code, payload = drive(root, fake, capsys, '--execute')

        assert code == 0
        gate = gate_of(payload, 82)
        assert gate['verdict'] == 'pass_tag_only'
        assert gate['disposition'] == 'executed'
        for member in fresh:
            assert fake.metadata_of(member)['topic'] == TOPIC
        assert fake.snapshot(canonical) == before
        assert fake.adds == 0
        assert gate['execution']['minted'] is False
        assert gate['closure']['closed'] is True
        assert [c['arguments']['escalation_id'] for c in gate['next']] == [escalation_id]

    def test_a_cleared_hazard_gate_is_stamped_before_it_is_resumed(
        self, root, fake, seed_tasks, capsys,
    ):
        members = fake.seed_many(3)
        seed_tasks([gate_row(
            83,
            TOPIC,
            observed=members,
            report_run='run-hazard',
            reason_codes={'member_carries_correction_banner': [members[0]]},
            auto_predicate_version='1',
            category=CATEGORY,
        )])
        seed_gate_escalation(root, 83)

        code, payload = drive(root, fake, capsys, '--execute', '--claim', f'{TOPIC}={CLAIM}')

        assert code == 0
        gate = gate_of(payload, 83)
        assert gate['disposition'] == 'executed'
        assert tools_of(gate) == ['update_task', 'resolve_issue']
        stamp = gate['next'][0]['arguments']
        assert stamp['id'] == '83'
        assert stamp['project_root'] == str(root)
        assert set(stamp['metadata']) == {'human_curator_adjudicated_at'}
        assert stamp['metadata_mode'] == 'merge'

    def test_a_closed_gate_with_no_pending_escalation_says_so(
        self, root, fake, seed_tasks, capsys,
    ):
        seed_tasks([gate_row(84, TOPIC, observed=fake.seed_many(3))])

        _, payload = drive(root, fake, capsys, '--execute', '--claim', f'{TOPIC}={CLAIM}')

        gate = gate_of(payload, 84)
        assert gate['closure']['closed'] is True
        assert 'resolve_issue' not in tools_of(gate)
        assert 'no pending milestone_gate escalation' in gate['note']

    def test_a_hazard_gate_never_lists_a_resume(self, root, fake, seed_tasks, capsys):
        fake.seed(
            topic=TOPIC, canonical=True, content='[CORRECTION 2026-01-01] the rule was reversed',
        )
        stamped = fake.seed_many(2, topic=TOPIC)
        seed_tasks([gate_row(85, TOPIC, observed=stamped)])
        seed_gate_escalation(root, 85)

        _, payload = drive(root, fake, capsys, '--execute')

        gate = gate_of(payload, 85)
        assert gate['disposition'] == 'left_to_sitting'
        assert gate['next'] == []

    def test_execute_leaves_tasks_db_byte_identical(self, root, fake, seed_tasks, capsys):
        db_path = seed_tasks([
            gate_row(86, TOPIC, observed=fake.seed_many(3)),
            gate_row(87, OTHER_TOPIC, observed=fake.seed_many(3)),
        ])
        before = task_rows(db_path)

        code, payload = drive(
            root, fake, capsys, '--execute',
            '--claim', f'{TOPIC}={CLAIM}', '--claim', f'{OTHER_TOPIC}={CLAIM}',
        )

        assert code == 0
        assert {g['disposition'] for g in payload['gates']} == {'executed'}
        assert task_rows(db_path) == before

    def test_the_text_report_carries_the_resume_call(self, root, fake, seed_tasks, capsys):
        seed_tasks([gate_row(88, TOPIC, observed=fake.seed_many(3))])
        escalation_id = seed_gate_escalation(root, 88)

        code = run_code(root, fake, '--execute', '--claim', f'{TOPIC}={CLAIM}')

        assert code == 0
        assert f"resolve_issue(escalation_id='{escalation_id}'" in capsys.readouterr().out
