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
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

import exception_register
import pytest
from _task_db_scan import tasks_db_path
from exception_owner_liveness import Liveness
from exception_owner_sweep import (
    FAILURE_TASK_ID,
    FINDING_KEY_FIELD,
    INLINE_SCOPE,
    MAX_OUTPUTS_PER_RUN,
    OVERFLOW_KEY,
    OWNS_NOTE_FIELD,
    REPORT_COMMAND,
    TIGHTEN_COMMAND,
    TIGHTEN_KEY,
    Action,
    ActOutcome,
    Citation,
    FailureStage,
    FindingKey,
    FindingKind,
    Ports,
    RegisterView,
    RunConfig,
    SweepFailure,
    SweepResult,
    act,
    escalation_url,
    parse_register,
    read_register,
    run,
    scan,
)
from inline_suppression_fixtures import RUFF_CONFIG, SUBPROCESS_TIMEOUT_SECS, track_fixture_tree
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


# ---------------------------------------------------------------------------
# INV-4: the storm bound, and the slack tighten task (scenarios 20 and 24).


def dead_owners(first: int, count: int) -> list[dict]:
    return [{'id': first + index, 'status': 'done'} for index in range(count)]


def inline_citations(first: int, count: int) -> tuple[Citation, ...]:
    return tuple(
        Citation(INLINE_SCOPE, TaskRef(first + index), (f'm{index}.py:1',)) for index in range(count)
    )


def keys_of(stub: StubMcp) -> list[dict]:
    return [args['metadata'][FINDING_KEY_FIELD] for args in stub.tool_calls('submit_task')]


def test_scenario_20_outputs_stop_at_the_bound_and_one_roll_up_reports_the_rest(project_root, seed):
    stub = StubMcp(seed(dead_owners(3001, 14)))
    ports = ports_for(project_root, stub)
    register = view(*inline_citations(3001, 14))

    first = sweep_once(register, ports)

    assert MAX_OUTPUTS_PER_RUN == 10
    assert keys_of(stub) == [
        FindingKey(FindingKind.DEAD_OWNER, INLINE_SCOPE, TaskRef(3001 + index)).record()
        for index in range(10)
    ]
    (roll_up,) = stub.tool_calls('escalate_info')
    assert roll_up['task_id'] == OVERFLOW_KEY.escalation_task_id()
    assert '4' in roll_up['detail']
    assert REPORT_COMMAND in roll_up['detail']
    assert (first.filed, first.overflow, first.findings) == (10, 4, 14)

    second = sweep_once(register, ports)

    assert len(stub.tool_calls('submit_task')) == 14
    assert len(stub.tool_calls('escalate_info')) == 1
    assert (second.skipped_existing, second.filed, second.overflow) == (10, 4, 0)


def test_no_second_roll_up_while_the_first_is_pending(project_root, seed):
    stub = StubMcp(seed(dead_owners(3001, 11)))
    stub.pending.append({'task_id': OVERFLOW_KEY.escalation_task_id()})

    outcome = sweep_once(view(*inline_citations(3001, 11)), ports_for(project_root, stub))

    assert outcome.overflow == 1
    assert stub.tool_calls('escalate_info') == []


def test_skips_never_consume_the_budget(project_root, seed):
    carried = [
        {
            'id': 8000 + index,
            'status': 'pending',
            'metadata': {
                FINDING_KEY_FIELD: FindingKey(FindingKind.DEAD_OWNER, INLINE_SCOPE, TaskRef(3001 + index)).record()
            },
        }
        for index in range(10)
    ]
    stub = StubMcp(seed(dead_owners(3001, 11) + carried))

    outcome = sweep_once(view(*inline_citations(3001, 11)), ports_for(project_root, stub))

    assert (outcome.skipped_existing, outcome.filed, outcome.overflow) == (10, 1, 0)


def test_scenario_24_slack_files_one_tighten_task_across_runs(project_root, seed):
    stub = StubMcp(seed([]))
    ports = ports_for(project_root, stub)
    register = view(slack_total=54)

    sweep_once(register, ports)
    sweep_once(register, ports)

    (submitted,) = stub.tool_calls('submit_task')
    assert submitted['metadata'][FINDING_KEY_FIELD] == TIGHTEN_KEY.record()
    assert TIGHTEN_COMMAND == 'uv run --project shared python scripts/inline_suppressions.py --tighten'
    assert TIGHTEN_COMMAND in submitted['description']


def test_no_slack_files_no_tighten_task(project_root, seed):
    stub = StubMcp(seed([]))

    sweep_once(view(slack_total=0), ports_for(project_root, stub))

    assert stub.tool_calls('submit_task') == []


def test_a_closed_tighten_task_is_followed_by_a_new_one_never_an_escalation(project_root, seed):
    stub = StubMcp(seed([{'id': 5000, 'status': 'done', 'metadata': {FINDING_KEY_FIELD: TIGHTEN_KEY.record()}}]))

    sweep_once(view(slack_total=3), ports_for(project_root, stub))

    assert keys_of(stub) == [TIGHTEN_KEY.record()]
    assert stub.tool_calls('escalate_info') == []


def test_the_tighten_task_comes_after_findings_and_inside_the_bound(project_root, seed):
    stub = StubMcp(seed(dead_owners(3001, 10)))

    outcome = sweep_once(view(*inline_citations(3001, 10), slack_total=54), ports_for(project_root, stub))

    assert TIGHTEN_KEY.record() not in keys_of(stub)
    assert (outcome.filed, outcome.overflow) == (10, 1)
    (roll_up,) = stub.tool_calls('escalate_info')
    assert roll_up['task_id'] == OVERFLOW_KEY.escalation_task_id()


# ---------------------------------------------------------------------------
# INV-9: the x_inv12_owns pointer note.  The declarations and markers are the
# home of who-cites-what; the note on each live cited owner only points there,
# and every run reconciles it.

LATER = NOW + datetime.timedelta(days=1)


def stored_metadata(db: Path, task_id: int) -> dict:
    with sqlite3.connect(db) as conn:
        (raw,) = conn.execute(
            "SELECT metadata FROM tasks WHERE tag = 'master' AND id = ?", (task_id,)
        ).fetchone()
    return json.loads(raw or '{}')


def note(*scopes: str, on: datetime.datetime = NOW) -> dict[str, object]:
    return {'scopes': list(scopes), 'as_of': on.date().isoformat(), 'report': REPORT_COMMAND}


def note_updates(stub: StubMcp) -> list[tuple[str, object]]:
    """Each ``update_task`` as (id, the note it merges); every one must be a shallow merge."""
    updates = stub.tool_calls('update_task')
    for args in updates:
        assert args.get('metadata_mode', 'merge') == 'merge'
        assert set(args['metadata']) == {OWNS_NOTE_FIELD}
    return [(args['id'], args['metadata'][OWNS_NOTE_FIELD]) for args in updates]


def cited_5149(*scopes: str) -> RegisterView:
    return view(*(Citation(scope, TaskRef(5149), ('a.py:1',)) for scope in scopes))


def test_a_live_cited_owner_carries_a_note_naming_every_citing_scope(project_root, seed):
    db = seed([{'id': 5149, 'status': 'pending', 'metadata': {'source': 'kept'}}])
    stub = StubMcp(db)

    sweep_once(cited_5149(LIST_SCOPE, INLINE_SCOPE), ports_for(project_root, stub))

    assert note_updates(stub) == [('5149', note(INLINE_SCOPE, LIST_SCOPE))]
    (update,) = stub.tool_calls('update_task')
    assert update['project_root'] == str(project_root)
    assert stored_metadata(db, 5149) == {'source': 'kept', OWNS_NOTE_FIELD: note(INLINE_SCOPE, LIST_SCOPE)}


def test_the_note_is_a_pointer_holding_no_disposition_or_verdict(project_root, seed):
    db = seed([{'id': 5149, 'status': 'pending'}])

    sweep_once(cited_5149(INLINE_SCOPE), ports_for(project_root, StubMcp(db)))

    assert set(stored_metadata(db, 5149)[OWNS_NOTE_FIELD]) == {'scopes', 'as_of', 'report'}


def test_an_unchanged_note_is_not_rewritten_nor_its_date_refreshed(project_root, seed):
    db = seed([{'id': 5149, 'status': 'pending'}])
    stub = StubMcp(db)
    ports = ports_for(project_root, stub)

    sweep_once(cited_5149(INLINE_SCOPE), ports)
    act(scan(cited_5149(INLINE_SCOPE), ports, now=LATER), ports, now=LATER)

    assert len(stub.tool_calls('update_task')) == 1
    assert stored_metadata(db, 5149)[OWNS_NOTE_FIELD] == note(INLINE_SCOPE)


def test_a_scope_added_rewrites_the_note_once(project_root, seed):
    db = seed([{'id': 5149, 'status': 'pending'}])
    stub = StubMcp(db)
    ports = ports_for(project_root, stub)
    sweep_once(cited_5149(INLINE_SCOPE), ports)

    widened = cited_5149(INLINE_SCOPE, LIST_SCOPE)
    act(scan(widened, ports, now=LATER), ports, now=LATER)

    assert note_updates(stub) == [
        ('5149', note(INLINE_SCOPE)),
        ('5149', note(INLINE_SCOPE, LIST_SCOPE, on=LATER)),
    ]


@pytest.mark.parametrize('status', ['pending', 'done'])
def test_a_note_nothing_cites_any_more_is_cleared_once(project_root, seed, status):
    db = seed([{'id': 6000, 'status': status, 'metadata': {OWNS_NOTE_FIELD: note(INLINE_SCOPE)}}])
    stub = StubMcp(db)
    ports = ports_for(project_root, stub)

    sweep_once(view(), ports)
    sweep_once(view(), ports)

    assert note_updates(stub) == [('6000', None)]
    assert stored_metadata(db, 6000)[OWNS_NOTE_FIELD] is None


def test_a_dead_cited_owners_note_is_left_as_it_is(project_root, seed):
    stale = note(INLINE_SCOPE, LIST_SCOPE)
    db = seed([{'id': 2733, 'status': 'done', 'metadata': {OWNS_NOTE_FIELD: stale}}])
    stub = StubMcp(db)

    outcome = sweep_once(INLINE_2733, ports_for(project_root, stub))

    assert outcome.filed == 1
    assert stub.tool_calls('update_task') == []
    assert stored_metadata(db, 2733)[OWNS_NOTE_FIELD] == stale


def test_the_note_lands_on_a_tickets_task_and_never_on_a_successor(project_root, seed):
    created = TicketRef('tkt_0C')
    db = seed(
        [
            {'id': 7001, 'status': 'pending'},
            {'id': 11, 'status': 'done', 'metadata': {'x_coalesced_into': 12}},
            {'id': 12, 'status': 'in-progress'},
        ]
    )
    stub = StubMcp(db, owner_tickets={created.id: {'status': 'created', 'task_id': '7001'}})
    register = view(
        Citation(INLINE_SCOPE, created, ('c.py:1',)),
        Citation(LIST_SCOPE, TaskRef(11), ('default',)),
    )

    sweep_once(register, ports_for(project_root, stub))

    assert note_updates(stub) == [('7001', note(INLINE_SCOPE))]


def test_note_writes_never_consume_the_output_budget(project_root, seed):
    live = [{'id': 100 + index, 'status': 'pending'} for index in range(3)]
    stub = StubMcp(seed(live + dead_owners(3001, 10)))
    register = view(
        *(Citation(INLINE_SCOPE, TaskRef(100 + index), (f'l{index}.py:1',)) for index in range(3)),
        *inline_citations(3001, 10),
    )

    outcome = sweep_once(register, ports_for(project_root, stub))

    assert (outcome.filed, outcome.overflow) == (10, 0)
    assert [task_id for task_id, _ in note_updates(stub)] == ['100', '101', '102']


def test_an_update_task_fault_is_a_fused_memory_failure(project_root, seed):
    stub = StubMcp(seed([{'id': 5149, 'status': 'pending'}]), fail_on=frozenset({'update_task'}))

    with pytest.raises(SweepFailure) as excinfo:
        sweep_once(cited_5149(INLINE_SCOPE), ports_for(project_root, stub))

    assert excinfo.value.stage is FailureStage.FUSED_MEMORY


# ---------------------------------------------------------------------------
# run(): the stamp, --dry-run and fail-loud (scenarios 21 and 23), and the CLI.

T0 = NOW - datetime.timedelta(hours=20)

ONE_DEAD = register_document(debt=(_site('m.py', 1, TaskRef(2733)),))


def stamp_path(project_root: Path) -> Path:
    return project_root / exception_register.SWEEP_STATE_PATH


def write_prior_stamp(project_root: Path) -> Path:
    """A stamp from a successful run at ``T0``, as the register's reader accepts it."""
    path = stamp_path(project_root)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'last_run': T0.isoformat(), 'ok': True}), encoding='utf-8')
    return path


def run_once(
    project_root: Path,
    stub: StubMcp,
    document: Mapping[str, object] | Callable[[], Mapping[str, object]],
    *,
    dry_run: bool = False,
) -> SweepResult:
    config = RunConfig(root=project_root, project_root=project_root, dry_run=dry_run)
    read = document if callable(document) else lambda: document
    return run(config, ports_for(project_root, stub), read_register=read, now=NOW)


def stamped(project_root: Path) -> dict:
    return json.loads(stamp_path(project_root).read_text(encoding='utf-8'))


def test_a_clean_run_stamps_a_result_the_register_reads_as_fresh(project_root, seed):
    stub = StubMcp(seed([{'id': 2733, 'status': 'done'}]))

    result = run_once(project_root, stub, ONE_DEAD)

    assert result.ok
    record = stamped(project_root)
    assert record.keys() >= {
        'checked', 'findings', 'filed', 'escalated', 'skipped_existing', 'overflow', 'slack', 'last_run', 'ok'
    }
    assert (record['ok'], record['last_run'], record['filed']) == (True, NOW.isoformat(), 1)
    reading = exception_register.read_sweep(stamp_path(project_root), now=NOW)
    assert (reading.status, reading.ok) == (exception_register.SweepStatus.FRESH, True)


def test_scenario_23_a_dry_run_reports_what_it_would_do_and_writes_nothing(project_root, seed):
    stub = StubMcp(seed([{'id': 2733, 'status': 'done'}, {'id': 5149, 'status': 'pending'}]))
    document = register_document(
        declarations=(_declaration(LIST_SCOPE, default=Debt(TaskRef(2733))),),
        debt=(_site('m.py', 1, TaskRef(2733)), _site('n.py', 2, TaskRef(5149))),
        slack_total=54,
    )

    result = run_once(project_root, stub, document, dry_run=True)

    assert result.ok
    assert {tool for _, tool, _ in stub.calls} <= {'get_pending_escalations', 'resolve_ticket'}
    assert not stamp_path(project_root).exists()
    assert [(r.scope, r.owner, r.verdict.liveness, r.action) for r in result.reports] == [
        (INLINE_SCOPE, TaskRef(2733), Liveness.DEAD, Action.FILE_TASK),
        (INLINE_SCOPE, TaskRef(5149), Liveness.LIVE, None),
        (LIST_SCOPE, TaskRef(2733), Liveness.DEAD, Action.FILE_ESCALATION),
    ]
    assert result.tighten is Action.FILE_TASK
    assert (result.checked, result.findings, result.filed, result.slack) == (3, 2, 0, 54)


def test_no_slack_plans_no_tighten(project_root, seed):
    result = run_once(project_root, StubMcp(seed([])), register_document(slack_total=0), dry_run=True)

    assert result.tighten is None


def test_a_report_line_names_the_successor_that_keeps_an_owner_live(project_root, seed):
    stub = StubMcp(
        seed(
            [
                {'id': 11, 'status': 'done', 'metadata': {'x_coalesced_into': 12}},
                {'id': 12, 'status': 'in-progress'},
            ]
        )
    )
    document = register_document(declarations=(_declaration(LIST_SCOPE, default=Debt(TaskRef(11))),))

    (report,) = run_once(project_root, stub, document, dry_run=True).reports

    line = report.line()
    assert 'via successor' in line
    assert '11' in line
    assert '12' in line


def test_scenario_21_an_unreadable_store_files_nothing_and_says_so_once(project_root):
    prior = write_prior_stamp(project_root)
    stub = StubMcp(tasks_db_path(project_root))

    result = run_once(project_root, stub, ONE_DEAD)

    assert stub.tool_calls('submit_task') == []
    assert stub.tool_calls('update_task') == []
    (escalated,) = stub.tool_calls('escalate_info')
    assert (escalated['task_id'], escalated['category']) == (FAILURE_TASK_ID, 'infra_issue')
    assert escalated['evidence'][0]['measured_at'] == NOW.isoformat()
    assert result.ok is False
    assert result.failure is not None
    assert result.failure.stage is FailureStage.STORE
    record = json.loads(prior.read_text(encoding='utf-8'))
    assert (record['ok'], record['last_run']) == (False, T0.isoformat())


def test_a_failure_with_no_prior_stamp_creates_none(project_root):
    result = run_once(project_root, StubMcp(tasks_db_path(project_root)), ONE_DEAD)

    assert result.ok is False
    assert not stamp_path(project_root).exists()


def _register_exited() -> Mapping[str, object]:
    raise SweepFailure(FailureStage.REGISTER, 'exception_register.py exited 2')


@pytest.mark.parametrize(
    'document', [_register_exited, {**register_document(), 'schema_version': 2}], ids=['exited', 'unreadable']
)
def test_a_register_failure_fails_the_run_before_any_store_or_mcp_read(project_root, seed, document):
    stub = StubMcp(seed([]))

    result = run_once(project_root, stub, document)

    assert result.ok is False
    assert result.failure is not None
    assert result.failure.stage is FailureStage.REGISTER
    assert [tool for _, tool, _ in stub.calls] == ['escalate_info']


def test_an_unreachable_escalation_server_fails_loud_without_raising(project_root, seed):
    stub = StubMcp(
        seed([{'id': 2733, 'status': 'done'}]),
        fail_on=frozenset({'get_pending_escalations', 'escalate_info'}),
    )

    result = run_once(project_root, stub, ONE_DEAD)

    assert result.ok is False
    assert result.failure is not None
    assert result.failure.stage is FailureStage.ESCALATION
    assert len(stub.tool_calls('escalate_info')) == 1
    assert 'escalate_info' in result.failure.detail
    assert stub.tool_calls('submit_task') == []


def test_a_failure_mid_act_keeps_what_landed_and_does_not_advance_the_stamp(project_root, seed):
    prior = write_prior_stamp(project_root)
    stub = StubMcp(
        seed([{'id': 2733, 'status': 'done'}, {'id': 5149, 'status': 'pending'}]),
        fail_on=frozenset({'update_task'}),
    )
    document = register_document(debt=(_site('m.py', 1, TaskRef(2733)), _site('n.py', 2, TaskRef(5149))))

    result = run_once(project_root, stub, document)

    assert result.failure is not None
    assert (result.filed, result.ok, result.failure.stage) == (1, False, FailureStage.FUSED_MEMORY)
    record = json.loads(prior.read_text(encoding='utf-8'))
    assert (record['ok'], record['last_run'], record['filed']) == (False, T0.isoformat(), 1)


@pytest.mark.parametrize('prior', [True, False], ids=['prior-stamp', 'no-stamp'])
def test_a_failed_dry_run_still_escalates_nothing_and_writes_nothing(project_root, prior):
    before = write_prior_stamp(project_root).read_bytes() if prior else None
    stub = StubMcp(tasks_db_path(project_root))

    result = run_once(project_root, stub, ONE_DEAD, dry_run=True)

    assert result.ok is False
    assert stub.tool_calls('escalate_info') == []
    if before is None:
        assert not stamp_path(project_root).exists()
    else:
        assert stamp_path(project_root).read_bytes() == before


def test_the_escalation_url_is_the_projects_configured_port(tmp_path: Path):
    root = tmp_path / 'proj'
    root.mkdir()
    (root / 'dark-factory-orchestrator.yaml').write_text('escalation:\n  port: 8102\n', encoding='utf-8')

    assert escalation_url(root) == 'http://127.0.0.1:8102/mcp'


def test_a_project_with_no_escalation_config_is_a_config_failure(tmp_path: Path):
    with pytest.raises(SweepFailure) as excinfo:
        escalation_url(tmp_path)

    assert excinfo.value.stage is FailureStage.CONFIG


SWEEP_SCRIPT = Path(__file__).resolve().parents[1] / 'exception_owner_sweep.py'


def unbound_port() -> int:
    """A loopback port nothing listens on: bound, read, then released."""
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def test_the_cli_dry_run_against_unreachable_servers_is_exit_two_and_writes_nothing(
    tmp_path: Path, project_root, seed
):
    seed([{'id': 77, 'status': 'done'}])
    tree = register_tree(tmp_path / 'tree', {'m.py': OWNED_MODULE})
    port = unbound_port()

    completed = subprocess.run(
        [
            sys.executable,
            str(SWEEP_SCRIPT),
            '--dry-run',
            '--root',
            str(tree),
            '--project-root',
            str(project_root),
            '--fused-memory-url',
            f'http://127.0.0.1:{port}',
            '--escalation-url',
            f'http://127.0.0.1:{port}/mcp',
        ],
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECS,
        check=False,
    )

    assert completed.returncode == 2, completed.stderr
    assert json.loads(completed.stdout.splitlines()[-1])['ok'] is False
    assert 'Traceback' not in completed.stderr
    assert not stamp_path(project_root).exists()


def test_a_missing_import_is_exit_two_and_never_a_traceback(tmp_path: Path):
    """Only a real process can prove it: the fault happens before main exists.

    The script is copied alone, so its first sibling import fails; ``-S``
    hides the editable install, which an emptied ``PYTHONPATH`` alone cannot.
    """
    copy = tmp_path / SWEEP_SCRIPT.name
    shutil.copyfile(SWEEP_SCRIPT, copy)
    env = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}

    completed = subprocess.run(
        [sys.executable, '-S', str(copy), '--dry-run'],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECS,
        env=env,
        check=False,
    )

    assert completed.returncode == 2, completed.stderr
    assert 'uv run --project shared' in completed.stderr
    assert 'Traceback' not in completed.stderr
