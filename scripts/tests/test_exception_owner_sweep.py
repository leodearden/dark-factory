"""Tests for ``scripts/exception_owner_sweep.py`` — INV-12's nightly owner sweep.

Under test, through public names only: the register input boundary and the
finding key; the scan/act core against a real temp tasks.db and an in-process
stub of both MCP servers (``StubMcp``); filing confirmation, the storm bound,
the ``x_inv12_owns`` pointer note, the run stamp, ``--dry-run``, fail-loud and
the CLI.  ``plans/inv12-exceptions-owned-or-ratified-prd.md`` D11, Contract
"Sweep", scenarios 16-24.  Fixture markers live in string literals only.
"""

from __future__ import annotations

import datetime
import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path

import exception_register
import pytest
from _task_db_scan import tasks_db_path
from exception_owner_sweep import (
    FINDING_KEY_FIELD,
    INLINE_SCOPE,
    OVERFLOW_KEY,
    OWNS_NOTE_FIELD,
    TIGHTEN_KEY,
    ActOutcome,
    Citation,
    FailureStage,
    FindingKey,
    FindingKind,
    Ports,
    RegisterView,
    SweepFailure,
    act,
    parse_register,
    read_register,
    scan,
)
from inline_suppression_fixtures import RUFF_CONFIG, track_fixture_tree
from shared.governed_exceptions import (
    Debt,
    Disposition,
    Policy,
    TaskRef,
    TicketRef,
    disposition_record,
    owner_record,
)

TICKET = TicketRef('tkt_0A')


def _site(path: str, line: int, owner: TaskRef | TicketRef) -> dict[str, object]:
    """One ``inline.debt[]`` entry, in the shape the register publishes."""
    return {'path': path, 'line': line, 'kind': 'noqa', 'codes': ['F401'], 'owner': owner_record(owner)}


def _declaration(
    list_id: str,
    *,
    default: Disposition | None = None,
    overrides: tuple[tuple[str, Disposition], ...] = (),
) -> dict[str, object]:
    """One ``declarations[]`` entry, in the shape the register publishes."""
    return {
        'list_id': list_id,
        'rule': 'r',
        'path': 'tests/test_decl.py',
        'line': 3,
        'default': None if default is None else disposition_record(default),
        'default_covers': 0 if default is None else 1,
        'overrides': [{'key': key, 'disposition': disposition_record(value)} for key, value in overrides],
    }


def register_document(
    *,
    declarations: tuple[dict[str, object], ...] = (),
    debt: tuple[dict[str, object], ...] = (),
    slack_total: int | None = None,
) -> dict[str, object]:
    """A register ``--json`` document carrying only what the sweep reads."""
    return {
        'schema_version': 1,
        'declarations': list(declarations),
        'inline': {'debt': list(debt)},
        'baseline': {'present': slack_total is not None, 'slack_total': slack_total},
    }


# ---------------------------------------------------------------------------
# (1) parse_register


def test_one_citation_per_scope_and_owner_listing_every_site():
    document = register_document(
        declarations=(
            _declaration(
                'pkg.mod.LIST',
                default=Debt(TaskRef(6760)),
                overrides=(
                    ('k1', Debt(TaskRef(6760))),
                    ('k2', Policy('inv12-x')),
                    ('k3', Debt(TaskRef(5149))),
                ),
            ),
            _declaration('pkg.other.LIST', default=Policy('inv12-x')),
        ),
        debt=(
            _site('b.py', 12, TaskRef(6760)),
            _site('a.py', 10, TaskRef(6760)),
            _site('a.py', 9, TaskRef(6760)),
            _site('c.py', 1, TICKET),
        ),
    )
    declared = ('tests/test_decl.py',)
    assert parse_register(document).citations == (
        Citation(INLINE_SCOPE, TaskRef(6760), ('a.py:9', 'a.py:10', 'b.py:12'), ('a.py', 'b.py')),
        Citation(INLINE_SCOPE, TICKET, ('c.py:1',), ('c.py',)),
        Citation('pkg.mod.LIST', TaskRef(5149), ('k3',), declared),
        Citation('pkg.mod.LIST', TaskRef(6760), ('default', 'k1'), declared),
    )


def test_task_owners_sort_by_number_not_by_text():
    document = register_document(debt=(_site('m.py', 1, TaskRef(10)), _site('m.py', 2, TaskRef(9))))
    assert [citation.owner for citation in parse_register(document).citations] == [
        TaskRef(9),
        TaskRef(10),
    ]


def test_policy_dispositions_cite_no_owner():
    document = register_document(declarations=(_declaration('pkg.mod.LIST', default=Policy('inv12-x')),))
    assert parse_register(document).citations == ()


@pytest.mark.parametrize(('slack', 'expected'), [(None, 0), (0, 0), (54, 54)])
def test_slack_total_reads_null_as_zero(slack, expected):
    assert parse_register(register_document(slack_total=slack)).slack_total == expected


def _without_inline() -> dict[str, object]:
    document = register_document()
    del document['inline']
    return document


def _malformed_owner() -> dict[str, object]:
    site = _site('m.py', 1, TaskRef(77))
    site['owner'] = {'type': 'task', 'id': '77'}
    return register_document(debt=(site,))


def _malformed_disposition() -> dict[str, object]:
    declaration = _declaration('pkg.mod.LIST', default=Debt(TaskRef(77)))
    declaration['default'] = {'type': 'mystery'}
    return register_document(declarations=(declaration,))


@pytest.mark.parametrize(
    'document',
    [
        {**register_document(), 'schema_version': 2},
        _without_inline(),
        _malformed_owner(),
        _malformed_disposition(),
        'not a document',
    ],
)
def test_a_document_the_sweep_cannot_read_is_a_register_failure(document):
    with pytest.raises(SweepFailure) as excinfo:
        parse_register(document)
    assert excinfo.value.stage is FailureStage.REGISTER


# ---------------------------------------------------------------------------
# (2) FindingKey


DEAD_INLINE = FindingKey(FindingKind.DEAD_OWNER, INLINE_SCOPE, TaskRef(6760))


def test_a_dead_owner_key_records_its_kind_scope_and_owner():
    assert DEAD_INLINE.record() == {
        'kind': 'dead_owner',
        'scope': 'inline',
        'owner': {'type': 'task', 'id': 6760},
    }


@pytest.mark.parametrize(
    'key',
    [DEAD_INLINE, FindingKey(FindingKind.DEAD_OWNER, 'pkg.mod.LIST', TICKET), TIGHTEN_KEY, OVERFLOW_KEY],
)
def test_a_key_round_trips_through_its_record(key):
    assert FindingKey.from_record(key.record()) == key


@pytest.mark.parametrize(
    'value',
    [
        None,
        'dead_owner',
        {'kind': 'exploded'},
        {'kind': 'dead_owner', 'scope': 'inline'},
        {'kind': 'dead_owner', 'scope': 7, 'owner': {'type': 'task', 'id': 1}},
        {'kind': 'tighten', 'owner': {'type': 'task', 'id': 1}},
    ],
)
def test_a_value_that_is_no_key_reads_as_none(value):
    assert FindingKey.from_record(value) is None


@pytest.mark.parametrize(
    ('kind', 'scope', 'owner'),
    [
        (FindingKind.DEAD_OWNER, None, TaskRef(1)),
        (FindingKind.DEAD_OWNER, 'inline', None),
        (FindingKind.TIGHTEN, None, TaskRef(1)),
        (FindingKind.OVERFLOW, 'inline', None),
    ],
)
def test_a_key_of_the_wrong_shape_is_refused_naming_its_values(kind, scope, owner):
    with pytest.raises(ValueError) as excinfo:
        FindingKey(kind, scope, owner)
    message = str(excinfo.value)
    assert str(kind) in message
    assert repr(scope) in message
    assert repr(owner) in message


def test_escalation_ids_are_distinct_per_key_and_stable():
    keys = [
        DEAD_INLINE,
        FindingKey(FindingKind.DEAD_OWNER, INLINE_SCOPE, TaskRef(6761)),
        FindingKey(FindingKind.DEAD_OWNER, 'pkg.mod.LIST', TaskRef(6760)),
        FindingKey(FindingKind.DEAD_OWNER, INLINE_SCOPE, TICKET),
        TIGHTEN_KEY,
        OVERFLOW_KEY,
    ]
    ids = [key.escalation_task_id() for key in keys]
    assert len(set(ids)) == len(keys)
    assert ids == [key.escalation_task_id() for key in keys]


def test_only_a_dead_owner_finding_escalates_once_closed():
    assert FindingKind.DEAD_OWNER.escalates_when_closed
    assert not FindingKind.TIGHTEN.escalates_when_closed
    assert not FindingKind.OVERFLOW.escalates_when_closed


def test_the_metadata_contract_field_names():
    """Consumed by the documentation task and the delivered check: a rename breaks both."""
    assert FINDING_KEY_FIELD == 'x_inv12_finding'
    assert OWNS_NOTE_FIELD == 'x_inv12_owns'


# ---------------------------------------------------------------------------
# (3) Wiring against the real register.

#: One tracked module whose unused import is suppressed and owned by task 77.
OWNED_MODULE = 'import os  # noqa: F401  # debt: task 77\n'


def register_tree(root: Path, files: dict[str, str]) -> Path:
    """A fixture checkout the register measures: a ruff config, an empty table, *files*."""
    track_fixture_tree(
        root, {'pyproject.toml': RUFF_CONFIG, exception_register.TABLE_PATH: '[]\n', **files}
    )
    return root


def test_the_real_register_feeds_the_sweep_its_citations(tmp_path: Path):
    root = register_tree(tmp_path / 'tree', {'m.py': OWNED_MODULE})
    view = parse_register(read_register(root, tmp_path / 'absent.json'))
    assert view == RegisterView(
        citations=(Citation(INLINE_SCOPE, TaskRef(77), ('m.py:1',), ('m.py',)),), slack_total=0
    )


def test_a_failing_register_is_a_register_failure(tmp_path: Path):
    root = tmp_path / 'tree'
    track_fixture_tree(root, {'pyproject.toml': RUFF_CONFIG, 'm.py': OWNED_MODULE})
    with pytest.raises(SweepFailure) as excinfo:
        read_register(root, tmp_path / 'absent.json')
    assert excinfo.value.stage is FailureStage.REGISTER


# ---------------------------------------------------------------------------
# The scan/act harness: a real temp tasks.db and an in-process StubMcp.

NOW = datetime.datetime(2026, 10, 17, 6, 0, tzinfo=datetime.UTC)

#: Where the stub's curator inserts the tasks it creates: clear of every seeded id.
FIRST_CREATED_ID = 9001


class StubMcp:
    """Both MCP servers, in process, at the sweep's ToolCall port signature.

    Records every call as ``(server, tool, args)``.  On the fused-memory side it
    plays the curator: a submitted candidate resolves per ``submit_outcome``,
    and a created task is INSERTED into the temp store with the submitted
    metadata.  The stub writes; the sweep only reads.
    """

    def __init__(
        self,
        db: Path,
        *,
        owner_tickets: Mapping[str, dict] | None = None,
        submit_outcome: str = 'created',
        combine_into: int | None = None,
        fail_on: frozenset[str] = frozenset(),
    ) -> None:
        self.db = db
        self.owner_tickets = dict(owner_tickets or {})
        self.submit_outcome = submit_outcome
        self.combine_into = combine_into
        self.fail_on = fail_on
        self.calls: list[tuple[str, str, dict]] = []
        self.pending: list[dict] = []
        self.submitted: dict[str, dict] = {}
        self.next_task_id = FIRST_CREATED_ID

    def tool_calls(self, tool: str) -> list[dict]:
        return [args for _, called, args in self.calls if called == tool]

    def _record(self, server: str, tool: str, args: Mapping[str, object]) -> None:
        self.calls.append((server, tool, dict(args)))
        if tool in self.fail_on:
            raise ConnectionError(f'{tool}: connection refused')

    def fused_memory(self, tool: str, args: Mapping[str, object]) -> Mapping[str, object]:
        self._record('fused-memory', tool, args)
        if tool == 'submit_task':
            if self.submit_outcome == 'error':
                return {'error': 'curator queue closed', 'error_type': 'RuntimeError'}
            ticket = f'tkt_S{len(self.submitted) + 1}'
            self.submitted[ticket] = dict(args)
            return {'ticket': ticket}
        if tool == 'resolve_ticket':
            ticket = str(args['ticket'])
            if ticket in self.owner_tickets:
                return self.owner_tickets[ticket]
            return self._resolve_submitted(self.submitted[ticket])
        if tool == 'update_task':
            return self._update(int(str(args['id'])), args['metadata'])
        raise AssertionError(f'unexpected fused-memory tool {tool}')

    def _resolve_submitted(self, args: dict) -> dict:
        if self.submit_outcome == 'created':
            task_id = self.next_task_id
            self.next_task_id += 1
            with sqlite3.connect(self.db) as conn:
                conn.execute(
                    "INSERT INTO tasks (tag, id, title, status, metadata, updated_at) "
                    "VALUES ('master', ?, ?, 'pending', ?, '2026-10-17T06:00:00+00:00')",
                    (task_id, args['title'], json.dumps(args['metadata'])),
                )
            return {'status': 'created', 'task_id': str(task_id)}
        if self.submit_outcome == 'combined':
            return {'status': 'combined', 'task_id': str(self.combine_into)}
        if self.submit_outcome == 'refused':
            return {'status': 'refused', 'reason': 'premise_cancelled'}
        if self.submit_outcome == 'pending':
            return {'status': 'failed', 'reason': 'timeout', 'task_id': None}
        raise AssertionError(f'unexpected submit_outcome {self.submit_outcome}')

    def _update(self, task_id: int, metadata: object) -> dict:
        assert isinstance(metadata, dict)
        with sqlite3.connect(self.db) as conn:
            (raw,) = conn.execute(
                "SELECT metadata FROM tasks WHERE tag = 'master' AND id = ?", (task_id,)
            ).fetchone()
            merged = {**json.loads(raw or '{}'), **metadata}
            conn.execute(
                "UPDATE tasks SET metadata = ? WHERE tag = 'master' AND id = ?",
                (json.dumps(merged), task_id),
            )
        return {'id': task_id, 'updated': True}

    def escalation(self, tool: str, args: Mapping[str, object]) -> Mapping[str, object]:
        self._record('escalation', tool, args)
        if tool == 'get_pending_escalations':
            return {'result': [{'task_id': entry['task_id']} for entry in self.pending]}
        if tool == 'escalate_info':
            self.pending.append(dict(args))
            return {'id': f'esc-{len(self.pending)}', 'status': 'queued', 'level': 0}
        raise AssertionError(f'unexpected escalation tool {tool}')


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    root = tmp_path / 'proj'
    (root / '.taskmaster' / 'tasks').mkdir(parents=True)
    return root


@pytest.fixture
def seed(make_tasks_db, project_root: Path):
    """Seed the project's tasks.db with *rows*; return its path."""

    def _seed(rows: list[dict]) -> Path:
        return make_tasks_db(rows, directory=project_root / '.taskmaster' / 'tasks')

    return _seed


def ports_for(project_root: Path, stub: StubMcp) -> Ports:
    return Ports(
        project_root=project_root,
        db=tasks_db_path(project_root),
        fused_memory=stub.fused_memory,
        escalation=stub.escalation,
    )


def sweep_once(register: RegisterView, ports: Ports) -> ActOutcome:
    return act(scan(register, ports, now=NOW), ports, now=NOW)


def view(*citations: Citation, slack_total: int = 0) -> RegisterView:
    return RegisterView(citations=citations, slack_total=slack_total)


def set_status(db: Path, task_id: int, status: str) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE tasks SET status = ? WHERE tag = 'master' AND id = ?", (status, task_id))


LIST_SCOPE = 'pkg.mod.LIST'


def test_scenario_16_a_dead_owner_is_filed_or_escalated_once_by_scope(project_root, seed):
    db = seed([{'id': 2733, 'status': 'done'}])
    stub = StubMcp(db)
    ports = ports_for(project_root, stub)
    register = view(
        Citation(INLINE_SCOPE, TaskRef(2733), ('m.py:1',)),
        Citation(LIST_SCOPE, TaskRef(2733), ('default',)),
    )

    first = sweep_once(register, ports)

    (submitted,) = stub.tool_calls('submit_task')
    inline_key = FindingKey(FindingKind.DEAD_OWNER, INLINE_SCOPE, TaskRef(2733))
    assert submitted['metadata'][FINDING_KEY_FIELD] == inline_key.record()
    assert 'planning_mode' not in submitted
    assert len(stub.tool_calls('resolve_ticket')) == 1
    (escalated,) = stub.tool_calls('escalate_info')
    list_key = FindingKey(FindingKind.DEAD_OWNER, LIST_SCOPE, TaskRef(2733))
    assert escalated['task_id'] == list_key.escalation_task_id()
    assert (first.findings, first.filed, first.escalated) == (2, 1, 1)

    second = sweep_once(register, ports)

    assert len(stub.tool_calls('submit_task')) == 1
    assert len(stub.tool_calls('escalate_info')) == 1
    assert second.skipped_existing == 2
    assert (second.filed, second.escalated) == (0, 0)


def test_scenario_17_every_kind_of_dead_owner_is_one_finding(project_root, seed):
    db = seed([{'id': 20, 'status': 'cancelled'}])
    failed, created = TicketRef('tkt_0F'), TicketRef('tkt_0C')
    stub = StubMcp(
        db,
        owner_tickets={
            failed.id: {'status': 'failed', 'reason': 'worker_dead'},
            created.id: {'status': 'created', 'task_id': '20'},
        },
    )
    register = view(
        Citation(INLINE_SCOPE, TaskRef(99), ('a.py:1',)),
        Citation(INLINE_SCOPE, failed, ('b.py:1',)),
        Citation(INLINE_SCOPE, created, ('c.py:1',)),
    )

    outcome = sweep_once(register, ports_for(project_root, stub))

    assert (outcome.checked, outcome.findings, outcome.filed) == (3, 3, 3)
    keys = [args['metadata'][FINDING_KEY_FIELD] for args in stub.tool_calls('submit_task')]
    assert keys == [
        FindingKey(FindingKind.DEAD_OWNER, INLINE_SCOPE, owner).record()
        for owner in (TaskRef(99), failed, created)
    ]


def test_scenario_19_a_closed_finding_task_escalates_and_is_never_refiled(project_root, seed):
    key = FindingKey(FindingKind.DEAD_OWNER, INLINE_SCOPE, TaskRef(2733))
    db = seed(
        [
            {'id': 2733, 'status': 'done'},
            {'id': 5000, 'status': 'done', 'metadata': {FINDING_KEY_FIELD: key.record()}},
        ]
    )
    stub = StubMcp(db)

    outcome = sweep_once(view(Citation(INLINE_SCOPE, TaskRef(2733), ('m.py:1',))), ports_for(project_root, stub))

    assert stub.tool_calls('submit_task') == []
    (escalated,) = stub.tool_calls('escalate_info')
    assert escalated['task_id'] == key.escalation_task_id()
    assert outcome.escalated == 1


def test_scenario_22_an_owner_that_dies_between_scan_and_act_is_filed(project_root, seed):
    db = seed([{'id': 2733, 'status': 'pending'}])
    stub = StubMcp(db)
    ports = ports_for(project_root, stub)
    register = view(Citation(INLINE_SCOPE, TaskRef(2733), ('m.py:1',)))

    measured = scan(register, ports, now=NOW)
    set_status(db, 2733, 'done')
    outcome = act(measured, ports, now=NOW)

    assert len(stub.tool_calls('submit_task')) == 1
    assert outcome.filed == 1


def test_scenario_22_an_owner_revived_between_scan_and_act_is_not_filed(project_root, seed):
    db = seed([{'id': 2733, 'status': 'done'}])
    stub = StubMcp(db)
    ports = ports_for(project_root, stub)
    register = view(Citation(INLINE_SCOPE, TaskRef(2733), ('m.py:1',)))

    measured = scan(register, ports, now=NOW)
    set_status(db, 2733, 'pending')
    outcome = act(measured, ports, now=NOW)

    assert stub.tool_calls('submit_task') == []
    assert stub.tool_calls('escalate_info') == []
    assert (outcome.findings, outcome.filed) == (0, 0)


def test_live_owners_direct_or_via_a_successor_produce_no_output(project_root, seed):
    db = seed(
        [
            {'id': 10, 'status': 'pending'},
            {'id': 11, 'status': 'done', 'metadata': {'x_coalesced_into': 12}},
            {'id': 12, 'status': 'in-progress'},
        ]
    )
    stub = StubMcp(db)
    register = view(
        Citation(INLINE_SCOPE, TaskRef(10), ('a.py:1',)),
        Citation(LIST_SCOPE, TaskRef(11), ('default',)),
    )

    outcome = sweep_once(register, ports_for(project_root, stub))

    assert stub.tool_calls('submit_task') == []
    assert stub.tool_calls('escalate_info') == []
    assert (outcome.checked, outcome.findings, outcome.via_successor) == (2, 0, 1)


def test_scan_is_read_only(project_root, seed):
    db = seed([{'id': 2733, 'status': 'done'}])
    stub = StubMcp(db)

    scan(view(Citation(INLINE_SCOPE, TaskRef(2733), ('m.py:1',))), ports_for(project_root, stub), now=NOW)

    assert [tool for _, tool, _ in stub.calls] == ['get_pending_escalations']


# ---------------------------------------------------------------------------
# D11: the curator is not the dedup mechanism.  A finding counts as filed only
# when its ticket yields a LIVE task carrying the key; otherwise it escalates.

DEAD_2733 = FindingKey(FindingKind.DEAD_OWNER, INLINE_SCOPE, TaskRef(2733))
INLINE_2733 = view(Citation(INLINE_SCOPE, TaskRef(2733), ('m.py:1',)))


def test_a_created_task_carrying_the_key_is_filed(project_root, seed):
    stub = StubMcp(seed([{'id': 2733, 'status': 'done'}]), submit_outcome='created')

    outcome = sweep_once(INLINE_2733, ports_for(project_root, stub))

    assert (outcome.filed, outcome.escalated) == (1, 0)
    assert stub.tool_calls('escalate_info') == []


def test_a_finding_combined_onto_a_closed_task_escalates_naming_the_outcome(project_root, seed):
    stub = StubMcp(
        seed([{'id': 2733, 'status': 'done'}, {'id': 4000, 'status': 'done'}]),
        submit_outcome='combined',
        combine_into=4000,
    )
    ports = ports_for(project_root, stub)

    outcome = sweep_once(INLINE_2733, ports)

    assert (outcome.filed, outcome.escalated) == (0, 1)
    (escalated,) = stub.tool_calls('escalate_info')
    assert escalated['task_id'] == DEAD_2733.escalation_task_id()
    detail = escalated['detail']
    for fact in ('tkt_S1', 'combined', 'task 4000', "'done'"):
        assert fact in detail

    again = sweep_once(INLINE_2733, ports)

    assert len(stub.tool_calls('submit_task')) == 1
    assert len(stub.tool_calls('escalate_info')) == 1
    assert again.skipped_existing == 1


@pytest.mark.parametrize(
    ('submit_outcome', 'combine_into'),
    [('combined', 4001), ('refused', None), ('pending', None)],
)
def test_a_submit_that_yields_no_live_task_carrying_the_key_escalates(
    project_root, seed, submit_outcome, combine_into
):
    stub = StubMcp(
        seed([{'id': 2733, 'status': 'done'}, {'id': 4001, 'status': 'pending'}]),
        submit_outcome=submit_outcome,
        combine_into=combine_into,
    )

    outcome = sweep_once(INLINE_2733, ports_for(project_root, stub))

    assert (outcome.filed, outcome.escalated) == (0, 1)
    (escalated,) = stub.tool_calls('escalate_info')
    assert escalated['task_id'] == DEAD_2733.escalation_task_id()


def test_a_submit_error_is_a_fused_memory_failure(project_root, seed):
    stub = StubMcp(seed([{'id': 2733, 'status': 'done'}]), submit_outcome='error')

    with pytest.raises(SweepFailure) as excinfo:
        sweep_once(INLINE_2733, ports_for(project_root, stub))

    assert excinfo.value.stage is FailureStage.FUSED_MEMORY
