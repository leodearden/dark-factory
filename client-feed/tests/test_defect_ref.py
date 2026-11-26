"""B24: the one ref rule for a client defect report, minted and read back."""

import pytest
from shared.defects import DefectOrigin, MalformedDefectRef, parse_ref

from client_feed.defect_ref import (
    PortalReport,
    check_ref,
    parse_portal_ref,
    portal_ref,
    portal_ref_prefix,
)


def test_a_client_report_mints_its_portal_ref() -> None:
    assert portal_ref('acme-widgets', 2, 3) == 'portal:acme-widgets:2:3'


def test_the_minted_ref_is_the_shared_grammars_client_ref() -> None:
    parsed = parse_ref(portal_ref('acme-widgets', 2, 3))

    assert parsed.origin is DefectOrigin.CLIENT
    assert parsed.components == (('project', 'acme-widgets'), ('block', '2'), ('n', '3'))


@pytest.mark.parametrize(
    ('project_id', 'block', 'report'),
    [
        ('acme-widgets', 0, 1),
        ('acme-widgets', -1, 1),
        ('acme-widgets', True, 1),
        ('acme-widgets', 2, 0),
        ('acme-widgets', 2, True),
        ('', 2, 1),
        ('Bad Id', 2, 1),
    ],
    ids=[
        'block-zero',
        'block-negative',
        'block-bool',
        'report-zero',
        'report-bool',
        'empty-project-id',
        'project-id-not-a-token',
    ],
)
def test_a_component_off_the_rule_is_refused(project_id: str, block: int, report: int) -> None:
    with pytest.raises(MalformedDefectRef):
        portal_ref(project_id, block, report)


@pytest.mark.parametrize(('block', 'report'), [(1, 1), (2, 3), (12, 1), (7, 40), (105, 2048)])
def test_a_minted_ref_reads_back_as_its_report(block: int, report: int) -> None:
    ref = portal_ref('acme-widgets', block, report)

    assert parse_portal_ref(ref, project_id='acme-widgets') == PortalReport(block=block, report=report)


@pytest.mark.parametrize(
    'ref',
    [
        'portal:other-client:2:3',
        'portal:acme-widgets-2:2:3',
        'review:7:0123456789ab',
        'portal:acme-widgets:check:1',
        'portal:acme-widgets:02:3',
        'portal:acme-widgets:0:3',
        'portal:acme-widgets:2',
        'not a ref',
    ],
    ids=[
        'another-project',
        'prefix-sharing-project',
        'another-origin',
        'block-token',
        'leading-zero-block',
        'zero-block',
        'wrong-arity',
        'not-a-ref',
    ],
)
def test_a_ref_the_project_would_not_mint_reads_as_no_report(ref: str) -> None:
    assert parse_portal_ref(ref, project_id='acme-widgets') is None


def test_the_prefix_is_the_text_every_ref_of_the_project_begins_with() -> None:
    assert portal_ref_prefix('acme-widgets') == 'portal:acme-widgets:'
    for project_id, block, report in [('acme-widgets', 2, 3), ('acme', 1, 1), ('a.b_c', 30, 400)]:
        assert portal_ref(project_id, block, report).startswith(portal_ref_prefix(project_id))


def test_the_prefix_of_one_project_never_selects_a_prefix_sharing_project() -> None:
    assert not portal_ref('acme-widgets', 2, 3).startswith(portal_ref_prefix('acme'))
    assert not portal_ref('acme', 2, 3).startswith(portal_ref_prefix('acme-widgets'))


def test_the_check_ref_is_the_projects_block_token_ref() -> None:
    assert check_ref('acme-widgets') == 'portal:acme-widgets:check:1'


def test_the_check_ref_is_a_client_ref_of_the_shared_grammar() -> None:
    assert parse_ref(check_ref('acme-widgets')).origin is DefectOrigin.CLIENT


def test_the_check_ref_names_no_portal_report() -> None:
    assert parse_portal_ref(check_ref('acme-widgets'), project_id='acme-widgets') is None


def test_the_check_ref_begins_with_the_projects_prefix() -> None:
    assert check_ref('acme-widgets').startswith(portal_ref_prefix('acme-widgets'))
