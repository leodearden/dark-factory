"""The one ref rule for a client's defect report (plans/client-feed-prd.md §14.2 decision 11), kept out of the vendorable contract."""

from __future__ import annotations

from shared.defects import DefectOrigin, MalformedDefectRef, mint_portal_ref


def portal_ref(project_id: str, block: int, report: int) -> str:
    """`portal:<project_id>:<block>:<report>`; the filer and the publisher both mint through this function."""
    if type(block) is not int or block < 1:
        origin = DefectOrigin.CLIENT
        raise MalformedDefectRef(
            f'{origin} ref component block={block!r} must be an int of at least 1'
            ' (bool is excluded even though it is an int subclass)',
            origin=origin,
            value=block,
        )
    return mint_portal_ref(project_id, str(block), report)
