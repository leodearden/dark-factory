"""No dispatched role may call ``record_defect_outcome``.

Authority over a defect's proof and ruling is by tool exposure
(``plans/defect-tests-prd.md`` decision 7): fused-memory has no authenticated
caller identity, so the orchestrator workflow and ``scripts/prove_defect.py``
are its only callers, and no role's ``allowed_tools`` may grant it. A grant can
be the tool's own name, the whole ``mcp__fused-memory`` server, or a glob that
matches it; the planted cases below show the predicate catches each form, so
the guard over ``ROLES`` is not vacuous.
"""

from fnmatch import fnmatchcase

import pytest

from orchestrator.agents.roles import ROLES

RECORD_DEFECT_OUTCOME = 'mcp__fused-memory__record_defect_outcome'


def _server_of(tool: str) -> str:
    return '__'.join(tool.split('__', 2)[:2])


def _grants(entry: str, tool: str) -> bool:
    return entry == tool or entry == _server_of(tool) or fnmatchcase(tool, entry)


@pytest.mark.parametrize(
    'entry',
    [
        RECORD_DEFECT_OUTCOME,
        'mcp__fused-memory',
        'mcp__fused-memory__*',
        'mcp__fused-memory__record_*',
        '*',
    ],
)
def test_each_grant_form_counts_as_a_grant(entry: str) -> None:
    assert _grants(entry, RECORD_DEFECT_OUTCOME)


@pytest.mark.parametrize(
    'entry',
    [
        'mcp__fused-memory__declare_defect',
        'mcp__fused-memory__get_task',
        'mcp__escalation__*',
        'mcp__escalation',
        'mcp__fused-memory__record_defect',
        'Read',
    ],
)
def test_other_entries_do_not_count_as_a_grant(entry: str) -> None:
    assert not _grants(entry, RECORD_DEFECT_OUTCOME)


def test_no_role_grants_record_defect_outcome() -> None:
    assert ROLES
    grants = [
        (name, entry)
        for name, role in ROLES.items()
        for entry in role.allowed_tools
        if _grants(entry, RECORD_DEFECT_OUTCOME)
    ]
    assert not grants, (
        f'{RECORD_DEFECT_OUTCOME} is reserved for the orchestrator workflow and '
        f'scripts/prove_defect.py, but these (role, allowed_tools entry) pairs grant it: '
        f'{grants}'
    )
