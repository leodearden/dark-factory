"""Tests for ``scripts/exception_owner_sweep.py`` — INV-12's nightly owner sweep.

Under test, through public names only: the register input boundary and the
finding key; the scan/act core against a real temp tasks.db and an in-process
stub of both MCP servers (``StubMcp``); filing confirmation, the storm bound,
the ``x_inv12_owns`` pointer note, the run stamp, ``--dry-run``, fail-loud and
the CLI.  ``plans/inv12-exceptions-owned-or-ratified-prd.md`` D11, Contract
"Sweep", scenarios 16-24.  Fixture markers live in string literals only.
"""

from __future__ import annotations

from pathlib import Path

import exception_register
import pytest
from exception_owner_sweep import (
    FINDING_KEY_FIELD,
    INLINE_SCOPE,
    OVERFLOW_KEY,
    OWNS_NOTE_FIELD,
    TIGHTEN_KEY,
    Citation,
    FailureStage,
    FindingKey,
    FindingKind,
    RegisterView,
    SweepFailure,
    parse_register,
    read_register,
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
    assert parse_register(document).citations == (
        Citation(INLINE_SCOPE, TaskRef(6760), ('a.py:9', 'a.py:10', 'b.py:12')),
        Citation(INLINE_SCOPE, TICKET, ('c.py:1',)),
        Citation('pkg.mod.LIST', TaskRef(5149), ('k3',)),
        Citation('pkg.mod.LIST', TaskRef(6760), ('default', 'k1')),
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
    assert view == RegisterView(citations=(Citation(INLINE_SCOPE, TaskRef(77), ('m.py:1',)),), slack_total=0)


def test_a_failing_register_is_a_register_failure(tmp_path: Path):
    root = tmp_path / 'tree'
    track_fixture_tree(root, {'pyproject.toml': RUFF_CONFIG, 'm.py': OWNED_MODULE})
    with pytest.raises(SweepFailure) as excinfo:
        read_register(root, tmp_path / 'absent.json')
    assert excinfo.value.stage is FailureStage.REGISTER
