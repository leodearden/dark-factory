"""B24: the one ref rule for a client defect report."""

import pytest
from client_feed.defect_ref import portal_ref
from shared.defects import DefectOrigin, MalformedDefectRef, parse_ref


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
