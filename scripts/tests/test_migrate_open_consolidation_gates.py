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
from fused_memory.config.schema import ConsolidationAutoConfig, Mem0UpdateConfig
from fused_memory.models.memory import AddMemoryResponse
from fused_memory.reconciliation.consolidation_gate import (
    GATE_METADATA_KEY,
    build_consolidation_gate_task,
)

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
